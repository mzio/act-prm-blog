"""
SFTTrainer — supervised fine-tuning trainer.

Subclass of :class:`strl.pytorch.trainer.rl.RLTrainer`. Inherits the rollout /
eval / checkpointing loop and overrides only ``compute_loss``: plain
cross-entropy on the action span with optional per-token advantage weights.
With ``advantage = 1.0`` per action token (the RAAWR-API outcome-SFT default),
this collapses to standard maximum-likelihood SFT — no PPO-style importance
ratio.
"""

import hashlib
import json
import logging
import math
import os
from typing import Any

import torch
from torch.nn import functional as F

from ..utils import action_start_token
from .rl import RLTrainer

logger = logging.getLogger(__name__)


# Shared with prepare_minibatch's label mask so the tokens trained under
# --train_action_only are exactly the tokens scored by eval_actiononly_*.
_action_start_token = action_start_token


class SFTTrainer(RLTrainer):
    """
    Supervised fine-tuning trainer.

    Inherits ``__init__``, ``train()``, ``prepare_minibatch()``,
    ``generate_and_save_trajectories()`` and the trajectory-saving helpers
    from :class:`RLTrainer`. The only behavioral difference is
    ``compute_loss``, which drops the PPO importance-sampling ratio that
    RLTrainer applies and computes plain (advantage-weighted) CE on the
    label / action span.

    Trajectories are still pulled from rollouts the same way as RL — the
    expectation is that for SFT the EpisodeSteps carry ``advantage = 1.0`` on
    every action token (set at construction time, e.g. by the outcome-SFT
    collector in ``state_action_raawr_api``). Non-action tokens are zeroed
    out by ``label_mask`` regardless.
    """

    @torch.no_grad()
    def eval_extra_metrics(
        self,
        trajectories: Any,
        split: str = "eval",
        checkpoint_name: str | None = None,
    ) -> dict[str, float]:
        """Offline SFT eval metrics, teacher-forced over the eval trajectories.

        Two spans are scored from the SAME forward pass (one forward per step):

        - Whole target span ``state_action_tokens[state_len:]`` (thought+action):
          ``eval_action_ppl``          = exp(mean CE over target tokens)
          ``eval_action_accuracy``     = fraction of target tokens argmax==gold
        - Action **sub-span** only (the ``<tool_call>...</tool_call>`` block or a
          ``Final Answer:`` suffix; == whole target when there's no reasoning
          prefix, e.g. an actions_only target). An action-token mask is applied to
          the already-computed logits — no second forward:
          ``eval_actiononly_ppl``      = exp(mean CE over action-only tokens)
          ``eval_actiononly_accuracy`` = fraction of action-only tokens argmax==gold

        The action span is isolated with the same suffix logic as
        ``scripts/eval_action_subspan.py`` (see ``_action_start_token``), applied to
        each step's rendered target ``step.action["content"]``. The
        ``eval_actiononly_*`` keys match the airline box exactly for cross-domain
        comparability. Keyed to the rollout-metric prefix so ``best_metric`` can
        early-stop on either ppl and a notebook can plot the per-eval curve.
        Returns {} if there are no scorable tokens.
        """
        trajs = trajectories.get("policy") if isinstance(trajectories, dict) else trajectories
        if not trajs:
            return {}

        model = self.llm.model
        device = model.device
        tokenizer = self.hf_tokenizer
        was_training = model.training
        model.eval()

        # Whole-target span accumulators (unchanged).
        total_ce = 0.0
        total_correct = 0
        total_tokens = 0
        # Action-only sub-span accumulators.
        act_ce = 0.0
        act_correct = 0
        act_tokens = 0
        # Per-step records, so any SUBSET of the eval turns can be scored offline without
        # re-running. Motivation: the arms disagree about which turns they can supervise --
        # the expert has reasoning on only ~50% of turns (31% on finance), Act-PRM has it on
        # all of them, actions_only on none. Restricting the comparison to turns where the
        # EXPERT reasoned is a fair, on-distribution question, but it cannot be answered from
        # inside a single arm's run: actions_only has no thoughts in its own targets and so
        # cannot identify the subset. Join these records across arms on `action_key` (see
        # below) to get the restricted table for every arm at once.
        step_records: list[dict[str, Any]] = []
        for traj in trajs:
            for step in traj.episode_steps:
                ids = getattr(step, "state_action_tokens", None)
                state_len = getattr(step, "state_len", None)
                if not ids or state_len is None or state_len >= len(ids):
                    continue
                input_ids = torch.tensor([ids], device=device)
                # Next-token-shifted, exactly as compute_loss / prepare_minibatch.
                logits = model(input_ids=input_ids, use_cache=False).logits[0, :-1, :]
                labels = input_ids[0, 1:]
                start = max(0, state_len - 1)  # first target-token prediction position
                tgt_logits = logits[start:]
                tgt_labels = labels[start:]
                if tgt_labels.numel() == 0:
                    continue
                tgt_logits_f = tgt_logits.float()
                tgt_argmax = tgt_logits_f.argmax(dim=-1)
                total_ce += F.cross_entropy(
                    tgt_logits_f, tgt_labels, reduction="sum"
                ).item()
                total_correct += int((tgt_argmax == tgt_labels).sum().item())
                total_tokens += int(tgt_labels.numel())

                # Action sub-span: reuse the SAME shifted logits/labels, just index
                # from the action's first prediction position (no second forward).
                target_content = None
                action_msg = getattr(step, "action", None)
                if isinstance(action_msg, dict):
                    target_content = action_msg.get("content")
                a_start = _action_start_token(tokenizer, ids, state_len, target_content)
                a_pos = max(0, a_start - 1)  # first action-token prediction position
                # Offset into the already-sliced target tensors.
                a_off = a_pos - start
                if 0 <= a_off < tgt_labels.numel():
                    a_labels = tgt_labels[a_off:]
                    a_argmax = tgt_argmax[a_off:]
                    _ce = F.cross_entropy(tgt_logits_f[a_off:], a_labels, reduction="sum").item()
                    _ok = int((a_argmax == a_labels).sum().item())
                    _n = int(a_labels.numel())
                    act_ce += _ce
                    act_correct += _ok
                    act_tokens += _n
                    # Content key. (sample_id, timestep) is NOT stable across arms: the
                    # thought pools carry a different number of trajectories than the
                    # actions_only pool (finance eval: 27/398 steps vs 25/363), so the
                    # sample numbering shifts and only ~90% of nominally-shared turns even
                    # agree on action length. The gold ACTION tokens, by contrast, are
                    # identical across arms for the same turn -- only the thought prefix
                    # differs -- so they identify the turn. Paired with the trailing state
                    # tokens to disambiguate a repeated identical tool call.
                    _akey = hashlib.sha1(
                        ",".join(str(int(t)) for t in a_labels.tolist()).encode()
                    ).hexdigest()[:16]
                    _skey = hashlib.sha1(
                        ",".join(str(int(t)) for t in ids[max(0, state_len - 64) : state_len]).encode()
                    ).hexdigest()[:16]
                    step_records.append(
                        {
                            "action_key": _akey,
                            "state_key": _skey,
                            "sample_id": getattr(step, "sample_id", None),
                            "timestep": getattr(step, "timestep", None),
                            "action_ce": _ce,
                            "action_correct": _ok,
                            "action_tokens": _n,
                            # a_start > state_len <=> this arm's target carries a reasoning
                            # prefix on this turn. False for every actions_only step.
                            "has_thought": bool(a_start > state_len),
                            "thought_tokens": int(max(0, a_start - state_len)),
                        }
                    )

        if was_training:
            model.train()
        if total_tokens == 0:
            return {}

        prefix = f"{checkpoint_name}_{split}" if checkpoint_name is not None else split
        out = {
            f"{prefix}/eval_action_ppl": math.exp(total_ce / total_tokens),
            f"{prefix}/eval_action_accuracy": total_correct / total_tokens,
            f"{prefix}/eval_action_loss": total_ce / total_tokens,  # mean CE (nats) = log(ppl)
        }
        if act_tokens > 0:
            # Action-subspan (tool_call only) — the metric that actually reflects
            # action fit independent of thought verbosity.
            out[f"{prefix}/eval_actiononly_ppl"] = math.exp(act_ce / act_tokens)
            out[f"{prefix}/eval_actiononly_accuracy"] = act_correct / act_tokens
            # Same subspan, restricted to the turns where THIS arm has a reasoning prefix.
            # Reported for transparency only: it is NOT cross-arm comparable, because each
            # arm's thought-bearing turns are a different subset (and empty for
            # actions_only). Use the per-step dump + a common turn set for that.
            _sub = [r for r in step_records if r["has_thought"]]
            _sn = sum(r["action_tokens"] for r in _sub)
            if _sn > 0:
                out[f"{prefix}/eval_actiononly_ppl_ownthoughtsub"] = math.exp(
                    sum(r["action_ce"] for r in _sub) / _sn
                )
                out[f"{prefix}/eval_actiononly_frac_turns_with_thought"] = len(_sub) / max(
                    1, len(step_records)
                )
        if step_records and self.log_path:
            try:
                os.makedirs(self.log_path, exist_ok=True)
                with open(os.path.join(self.log_path, "eval_step_records.jsonl"), "a") as _f:
                    for _r in step_records:
                        _f.write(json.dumps({"split": split, "prefix": prefix, **_r}) + "\n")
            except Exception as _e:  # never let bookkeeping kill an eval
                logger.warning("could not write eval_step_records.jsonl: %s", _e)
        return out

    def compute_loss(
        self,
        model: torch.nn.Module,
        batch: dict[str, torch.Tensor],
        fp32_loss: bool | None = None,
    ) -> dict[str, Any]:
        """Plain (weighted) CE on the label-mask span; no PPO ratio.

        ``loss = -(advantages * new_logprobs * label_mask).sum() / num_label_tokens``
        """
        fp32_loss = fp32_loss or self.fp32_loss
        device = model.device

        # ``advantages`` is reused as a per-token weight (1.0 ⇒ plain CE).
        # ``label_mask`` zeroes out non-action tokens.
        advantages = batch["advantages"].to(device)
        label_mask = batch["label_mask"].to(device)

        model_inputs = {k: v.to(device) for k, v in batch.items() if k in ("input_ids", "attention_mask")}
        # Next-token-shifted logits / labels (matches prepare_minibatch).
        logits = model(**model_inputs, use_cache=False).logits[:, :-1, :]
        labels = model_inputs["input_ids"][:, 1:]
        dtype = torch.float32 if fp32_loss else model.dtype

        # log p(label_t | x_<t)  — shape (B, L-1)
        new_logprobs = -F.cross_entropy(
            logits.transpose(1, 2).to(dtype=dtype),  # (B, V, L-1)
            labels,
            reduction="none",
        ).to(dtype=logits.dtype)

        num_label_tokens = label_mask.sum().clamp_min(1)
        loss = -(new_logprobs * label_mask * advantages).sum() / num_label_tokens

        # Logging metrics (parallel to RLTrainer.compute_loss for dashboard parity)
        ppl = torch.exp(-(new_logprobs * label_mask).sum() / num_label_tokens).item()
        # Next-action-token accuracy on the supervised span: fraction of target
        # tokens whose greedy argmax matches the gold token (train-split analogue of
        # eval_action_accuracy). Detached — no grad needed for the metric.
        with torch.no_grad():
            token_accuracy = (
                ((logits.argmax(dim=-1) == labels).to(new_logprobs.dtype) * label_mask).sum()
                / num_label_tokens
            ).item()
        # Action-only TRAIN metrics. The loss above is over the full label_mask
        # (thought + action); these report the same span eval_actiononly_* reports on
        # the eval split, so train and eval curves are directly comparable. action_mask
        # is metrics-only and never enters the loss. Falls back to label_mask when the
        # collator didn't supply one (older replay buffers / non-SFT callers).
        _amask = batch.get("action_mask")
        if _amask is not None:
            _amask = _amask.to(device)
            _n_act = _amask.sum().clamp_min(1)
            with torch.no_grad():
                actiononly_ppl = torch.exp(-(new_logprobs * _amask).sum() / _n_act).item()
                actiononly_accuracy = (
                    ((logits.argmax(dim=-1) == labels).to(new_logprobs.dtype) * _amask).sum() / _n_act
                ).item()
                action_token_frac = (_amask.sum() / label_mask.sum().clamp_min(1)).item()
        else:
            actiononly_ppl = ppl
            actiononly_accuracy = token_accuracy
            action_token_frac = 1.0

        _n_label_tokens_out = num_label_tokens.item() if hasattr(num_label_tokens, "item") else num_label_tokens
        _n_action_tokens_out = (
            _amask.sum().item() if _amask is not None else _n_label_tokens_out
        )
        mean_advantage = (advantages.sum() / num_label_tokens).item()
        per_seq_gen_lens = label_mask.sum(dim=-1).tolist()
        num_gen_tokens = sum(per_seq_gen_lens) / len(per_seq_gen_lens) if per_seq_gen_lens else 0.0

        del advantages, label_mask, model_inputs, logits, labels, num_label_tokens, _amask
        torch.cuda.empty_cache()

        return {
            "loss": loss,
            "ppl": ppl,
            "actiononly_ppl": actiononly_ppl,
            "actiononly_accuracy": actiononly_accuracy,
            "action_token_frac": action_token_frac,
            "action_accuracy": token_accuracy,
            # Token counts so a caller aggregating over batches can do a TOKEN-WEIGHTED
            # mean. Averaging per-batch means is wrong here: action spans range ~57-973
            # tokens, so short batches would count as much as long ones.
            "n_label_tokens": float(_n_label_tokens_out),
            "n_action_tokens": float(_n_action_tokens_out),
            "advantage": mean_advantage,
            "num_gen_tokens": num_gen_tokens,
        }


__all__ = ["SFTTrainer"]
