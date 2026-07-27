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

from .rl import RLTrainer


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
        """Offline SFT eval metrics over the target (thought+action) span, computed
        teacher-forced over the eval trajectories:

        - ``eval_action_ppl``      = exp(mean CE over target tokens)
        - ``eval_action_accuracy`` = fraction of target tokens whose argmax logit
                                     equals the gold token

        The target span is ``state_action_tokens[state_len:]`` — the same span
        ``prepare_minibatch`` / ``compute_loss`` supervise. Keyed to match the
        rollout-metric prefix so ``best_metric: eval_action_ppl`` early-stops on it
        and a notebook can plot it. Returns {} if there are no scorable tokens.
        """
        trajs = trajectories.get("policy") if isinstance(trajectories, dict) else trajectories
        if not trajs:
            return {}

        model = self.llm.model
        device = model.device
        was_training = model.training
        model.eval()

        from act_prm.environments.act_prm_traces.data import extract_action

        total_ce = 0.0
        total_correct = 0
        total_tokens = 0
        # ACTION-SUBSPAN metrics: score ONLY the logged action tokens (the
        # <tool_call> block), excluding the (possibly verbose) thought — so the
        # whole-span ppl isn't dominated by hard-to-predict reasoning tokens
        # (e.g. expert_thoughts). This isolates whether thoughts help ACTION fit.
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
                total_ce += F.cross_entropy(
                    tgt_logits.float(), tgt_labels, reduction="sum"
                ).item()
                total_correct += int((tgt_logits.argmax(dim=-1) == tgt_labels).sum().item())
                total_tokens += int(tgt_labels.numel())

                # Action subspan = the trailing action tokens of the target span.
                act = getattr(step, "action", None)
                content = act.get("content") if isinstance(act, dict) else None
                action_str = extract_action(content or "") if content else None
                if action_str:
                    n_act = len(self.hf_tokenizer(action_str, add_special_tokens=False)["input_ids"])
                    n_act = min(n_act, int(tgt_labels.numel()))
                    if n_act > 0:
                        a_logits = tgt_logits[-n_act:]
                        a_labels = tgt_labels[-n_act:]
                        act_ce += F.cross_entropy(a_logits.float(), a_labels, reduction="sum").item()
                        act_correct += int((a_logits.argmax(dim=-1) == a_labels).sum().item())
                        act_tokens += n_act

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
        mean_advantage = (advantages.sum() / num_label_tokens).item()
        per_seq_gen_lens = label_mask.sum(dim=-1).tolist()
        num_gen_tokens = sum(per_seq_gen_lens) / len(per_seq_gen_lens) if per_seq_gen_lens else 0.0

        del advantages, label_mask, model_inputs, logits, labels, num_label_tokens
        torch.cuda.empty_cache()

        return {
            "loss": loss,
            "ppl": ppl,
            "action_accuracy": token_accuracy,
            "advantage": mean_advantage,
            "num_gen_tokens": num_gen_tokens,
        }


__all__ = ["SFTTrainer"]
