"""
SFTTrainer — supervised fine-tuning trainer.

Subclass of :class:`strl.pytorch.trainer.rl.RLTrainer`. Inherits the rollout /
eval / checkpointing loop and overrides only ``compute_loss``: plain
cross-entropy on the action span with optional per-token advantage weights.
With ``advantage = 1.0`` per action token (the RAAWR-API outcome-SFT default),
this collapses to standard maximum-likelihood SFT — no PPO-style importance
ratio.
"""

import math
from typing import Any

import torch
from torch.nn import functional as F

from act_prm.environments.act_prm_traces.data import extract_action

from .rl import RLTrainer


def _action_start_token(
    tokenizer: Any, ids: list[int], state_len: int, target_content: str | None
) -> int:
    """First token index (into ``ids``) at which the explicit action begins within
    the target span ``ids[state_len:]``.

    Mirrors ``scripts/eval_action_subspan.py::action_start_token`` but operates
    directly on the *already-tokenized* ``state_action_tokens`` (no re-render / no
    second forward). The action (``<tool_call>...</tool_call>`` block or a
    ``Final Answer:`` suffix) is always a **suffix** of the assistant content, so we
    find the largest token index ``k >= state_len`` such that the decoded tail
    ``decode(ids[k:])`` still fully contains the action string — that token is the
    action's first token. Returns ``state_len`` (whole target == action) when there
    is no separable reasoning prefix, matching the subspan script's fallback.
    """
    action_str = extract_action(target_content or "")
    if not action_str:
        return state_len  # no separable action -> whole target span is the action

    def _tail_has_action(k: int, needle: str) -> bool:
        return needle in tokenizer.decode(ids[k:])

    # Sanity: the action must appear somewhere in the target tail. If the exact
    # extracted string can't be located (chat-template / whitespace artifacts),
    # fall back to a looser marker, else to the whole-target span.
    if not _tail_has_action(state_len, action_str):
        for marker in ("<tool_call>", "Final Answer:"):
            if _tail_has_action(state_len, marker):
                action_str = marker
                break
        else:
            return state_len

    a_start = state_len
    for k in range(state_len, len(ids)):
        if _tail_has_action(k, action_str):
            a_start = k
        else:
            break
    return a_start


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
                    act_ce += F.cross_entropy(
                        tgt_logits_f[a_off:], a_labels, reduction="sum"
                    ).item()
                    act_correct += int((a_argmax == a_labels).sum().item())
                    act_tokens += int(a_labels.numel())

        if was_training:
            model.train()
        if total_tokens == 0:
            return {}

        prefix = f"{checkpoint_name}_{split}" if checkpoint_name is not None else split
        out = {
            f"{prefix}/eval_action_ppl": math.exp(total_ce / total_tokens),
            f"{prefix}/eval_action_accuracy": total_correct / total_tokens,
        }
        if act_tokens > 0:
            out[f"{prefix}/eval_actiononly_ppl"] = math.exp(act_ce / act_tokens)
            out[f"{prefix}/eval_actiononly_accuracy"] = act_correct / act_tokens
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
        mean_advantage = (advantages.sum() / num_label_tokens).item()
        per_seq_gen_lens = label_mask.sum(dim=-1).tolist()
        num_gen_tokens = sum(per_seq_gen_lens) / len(per_seq_gen_lens) if per_seq_gen_lens else 0.0

        del advantages, label_mask, model_inputs, logits, labels, num_label_tokens
        torch.cuda.empty_cache()

        return {
            "loss": loss,
            "ppl": ppl,
            "advantage": mean_advantage,
            "num_gen_tokens": num_gen_tokens,
        }


__all__ = ["SFTTrainer"]
