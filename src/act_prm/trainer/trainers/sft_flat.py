"""
SFTFlatTrainer — SFT over a CORPUS-WIDE FLAT dataset of (state, action) steps.

Difference from :class:`act_prm.trainer.trainers.sft.SFTTrainer` (which inherits
``RLTrainer.train``): the sampling unit. Both flatten to one row per logged action and
both minimise the same weighted CE (advantage 1.0), but

  sft.py      : per outer batch, pick ``batch_size`` TRAJECTORIES, expand them into their
                ~64 steps, shuffle within that set, accumulate, one optimizer step. Every
                update therefore sees steps that are correlated -- all from the same 4
                episodes. Re-derives EpisodeSteps each batch by calling the generator's
                ``_score_action_only``: ONE MODEL FORWARD PER STEP PER EPOCH, purely to
                fill ``old_logprobs`` that the CE loss never reads (there is no importance
                ratio -- verified: train/advantage is exactly 1.0 on every arm).

  sft_flat.py : tokenise every step ONCE at startup (no forward at all), shuffle the flat
                step list corpus-wide, and draw each batch from the whole corpus, so an
                update sees ~N steps from ~N different trajectories.

What this file deliberately REUSES rather than re-implements:
  compact_observations  (environments/act_prm_traces/data.py) -- the hide-observations /
                        obs_max_chars state construction. Skipping it would silently train
                        on full context, so the env args are read exactly as
                        generator/act_prm/base.py:477-482 reads them.
  get_batch_model_inputs(generator/utils.py) -- messages -> right-padded input_ids/mask
  prepare_minibatch     (trainer/train.py)   -- EpisodeSteps -> rows -> DataLoader, and it
                        builds label_mask / action_mask / advantages
  SFTTrainer.compute_loss                    -- the loss AND all the action-span metrics
  action_start_token                         -- via prepare_minibatch / compute_loss

Eval runs the SAME compute_loss over the eval steps under no_grad, replacing sft.py's two
unbatched forwards per step (generator scoring + eval_extra_metrics) with one batched pass.
"""

import os
import time
from os.path import join
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig
from torch.optim import Optimizer
from tqdm import tqdm

from act_prm.environments.act_prm_traces.data import compact_observations
from act_prm.environments.base import Environment
from act_prm.generator.utils import get_batch_model_inputs
from act_prm.llm_handlers.huggingface import HuggingFaceLLM
from act_prm.lora import save_lora
from act_prm.replay_buffer.types import EpisodeStep, Trajectory

from act_prm.utils.logging import timed

from ..train import prepare_minibatch
from ..utils import rotate_stale_run_artifacts
from .rl import _lower_is_better, is_better
from .sft import SFTTrainer


def _n_tokens(hf_tokenizer: Any, messages: list[dict[str, str]], enable_thinking: bool,
              **template_kwargs: Any) -> int:
    """Rendered-chat token count. Mirrors ActPrmGenerator._n_tokens so ``state_len``
    lands on exactly the same boundary the generator path produces."""
    out = hf_tokenizer.apply_chat_template(
        messages, tokenize=True, enable_thinking=enable_thinking, **template_kwargs
    )
    if not isinstance(out, (list, tuple)):  # newer transformers -> BatchEncoding
        out = out["input_ids"]
        if out and isinstance(out[0], (list, tuple)):
            out = out[0]
    return len(out)


def build_flat_steps(
    env: Environment,
    split: str,
    hf_tokenizer: Any,
    enable_thinking: bool = False,
    max_steps_per_traj: int | None = None,
    require_thought: bool = False,
    require_thought_eval: bool = False,
) -> list[EpisodeStep]:
    """Tokenise every logged (state, action) pair in ``env.datasets[split]`` into an
    EpisodeStep. NO model forward -- this is the whole point of the file.

    ``old_logprobs`` is a zero vector of the ACTION-token length, not []: prepare_minibatch
    sizes padded_logprobs / padded_advantages / padded_mask from ``len(old_logprobs)``
    (train.py:496-509), so an empty list would yield an empty label mask and train on
    nothing. The values are unused (the CE loss reads new_logprobs); only the length matters.
    """
    # Read exactly what generator/act_prm/base.py:477-482 reads, or hide-obs silently
    # stops applying and every arm trains on full context.
    obs_max_chars = getattr(env, "obs_max_chars", 2000)
    first_obs_to_show = getattr(env, "first_obs_to_show", 1)
    last_obs_to_show = getattr(env, "last_obs_to_show", 1)
    hide_middle = getattr(env, "hide_observations", False)

    pool = env.datasets[split]
    steps: list[EpisodeStep] = []
    for sample_id, traj in enumerate(
        tqdm(pool, desc=f"build flat steps [{split}]", colour="cyan")
    ):
        messages = traj["messages"]
        system_prompt = traj.get("system_prompt", "")
        action_indices = [i for i, m in enumerate(messages) if m.get("role") == "assistant"]
        # require_thought: train ONLY on assistant turns carrying reasoning before the
        # action -- the expert_thoughts_all arm. Implemented in the GENERATOR
        # (generator/act_prm/base.py:512), which sft_flat bypasses entirely, so without
        # this the flag reached the config, was silently ignored, and expert_thoughts_all
        # ran as an exact duplicate of expert_thoughts (identical PPL and action_frac).
        # TRAIN ONLY by default: filtering eval too would score the arm on a different,
        # harder subset than the other arms.
        if require_thought and (split == "train" or require_thought_eval):
            from act_prm.environments.act_prm_traces.data import extract_action as _xa

            def _has_thought(i: int) -> bool:
                c = messages[i].get("content") or ""
                a = _xa(c)
                if not a or c.find(a) < 0:
                    return False
                return len(c[: c.find(a)].strip()) >= 10

            action_indices = [i for i in action_indices if _has_thought(i)]
        if max_steps_per_traj:
            action_indices = action_indices[:max_steps_per_traj]
        for t, idx in enumerate(action_indices):
            state = compact_observations(
                messages[:idx], obs_max_chars,
                first_to_show=first_obs_to_show,
                last_to_show=last_obs_to_show,
                hide_middle=hide_middle,
            )
            x_t = messages[idx].get("content") or ""
            if not x_t.strip():
                continue
            scoring_state = [{"role": "system", "content": system_prompt}] + [
                m for m in state if m["role"] != "system"
            ]
            state_len = _n_tokens(
                hf_tokenizer, scoring_state, enable_thinking, add_generation_prompt=True
            )
            full_msgs = scoring_state + [{"role": "assistant", "content": x_t}]
            model_inputs, _ = get_batch_model_inputs(
                input_messages=[full_msgs],
                tools=None,
                hf_tokenizer=hf_tokenizer,
                padding_side="right",
                enable_thinking=enable_thinking,
                add_generation_prompt=False,
                continue_final_message=False,
            )
            ids = model_inputs["input_ids"][0]
            mask = model_inputs["attention_mask"][0].bool()
            sa_tokens = ids[mask].tolist()
            n_action = len(sa_tokens) - state_len
            if n_action <= 0:  # template collapsed the target; nothing to supervise
                continue
            step = EpisodeStep(
                state=scoring_state,
                action={"role": "assistant", "content": x_t},
                next_obs=[],
                state_action_tokens=sa_tokens,
                state_len=state_len,
                old_logprobs=[0.0] * n_action,   # length matters, values do not
                temperature=1.0,
                reward=0.0,
                done=True,
                truncated=False,
                timestep=t,
                try_step=0,
                batch_id=0,
                sample_id=sample_id,
                generation_id=0,
                split=split,
            )
            step.advantage = 1.0
            step.advantage_is_computed = True
            steps.append(step)
    return steps


def _as_single_step_trajectories(steps: list[EpisodeStep]) -> list[Trajectory]:
    """prepare_minibatch consumes Trajectories; one step each keeps rows independent."""
    return [
        Trajectory(episode_steps=[s], try_step=0, discount_factor=1.0, final_reward=0.0)
        for s in steps
    ]


class SFTFlatTrainer(SFTTrainer):
    """SFT with corpus-wide step-level shuffling. Inherits compute_loss (and its
    action-span metrics) from SFTTrainer; overrides only the training loop."""

    def _eval_flat(self, llm, eval_steps: list[EpisodeStep], cfg, micro_bs: int) -> dict[str, float]:
        """Teacher-forced eval over the flat eval steps, via the SAME compute_loss used in
        training. Replaces sft.py's generator-scoring pass + eval_extra_metrics (two
        unbatched forwards per step) with one batched pass."""
        llm.model.eval()
        loader, _ = prepare_minibatch(
            new_trajectories=_as_single_step_trajectories(eval_steps),
            hf_tokenizer=self.hf_tokenizer,
            batch_size=micro_bs,
            max_seq_len=cfg.get("max_seq_len", 32768),
            train_action_only=cfg.get("train_action_only", False),
            shuffle=False,
        )
        # TOKEN-WEIGHTED aggregation, matching eval_extra_metrics' total_ce/total_tokens.
        # Averaging per-batch means would be wrong: action spans run ~57-973 tokens, so a
        # short batch would count as much as a long one. compute_loss gives per-batch means
        # plus the token counts, so recover the sums:  ce_sum = log(ppl) * n_tokens.
        import math

        ce_lab = ce_act = corr_lab = corr_act = 0.0
        n_lab = n_act = 0.0
        frac_num = 0.0
        with torch.no_grad():
            for mb in loader:
                m = self.compute_loss(llm.model, mb, fp32_loss=self.fp32_loss)
                nl, na = float(m["n_label_tokens"]), float(m["n_action_tokens"])
                ce_lab += math.log(max(float(m["ppl"]), 1e-12)) * nl
                ce_act += math.log(max(float(m["actiononly_ppl"]), 1e-12)) * na
                corr_lab += float(m["action_accuracy"]) * nl
                corr_act += float(m["actiononly_accuracy"]) * na
                frac_num += na
                n_lab += nl
                n_act += na
        llm.model.train()
        if not n_lab:
            return {}
        return {
            "eval/eval_ppl": math.exp(ce_lab / n_lab),
            "eval/eval_action_accuracy": corr_lab / n_lab,
            "eval/eval_actiononly_ppl": math.exp(ce_act / max(n_act, 1.0)),
            "eval/eval_actiononly_accuracy": corr_act / max(n_act, 1.0),
            "eval/eval_action_token_frac": frac_num / n_lab,
            "eval/eval_n_label_tokens": n_lab,
            "eval/eval_n_action_tokens": n_act,
        }

    def train(
        self,
        llm: HuggingFaceLLM | None = None,
        optimizer: Optimizer | Any | None = None,
        cfg: DictConfig | None = None,
        env: Environment | None = None,
        eval_env: Environment | None = None,
        eval_every: int | None = None,
        num_steps: int | None = None,
        num_substeps: int | None = None,
        checkpoint_name: str | None = None,
        name_or_identifier: str | None = None,
        **kwargs: Any,
    ) -> HuggingFaceLLM:
        llm = llm or self.llm
        optimizer = optimizer or self.optimizer
        cfg = cfg or self.cfg
        rotate_stale_run_artifacts(cfg, self.checkpoint_path)
        env = env or self.env
        eval_env = eval_env or self.eval_env
        hf_tokenizer = self.hf_tokenizer
        eval_every = eval_every or cfg.eval_every
        num_steps = num_steps or cfg.get("num_steps", None) or cfg.num_batches

        # steps_per_batch = the EFFECTIVE gradient batch, in STEPS. sft.py's is implicit
        # (~64-70: every step of 4 trajectories); here it is explicit and comparable.
        steps_per_batch = int(cfg.get("steps_per_batch", 32))
        micro_bs = int(cfg.get("dataloader_batch_size", 1))
        enable_thinking = bool(self.generator_cfg.get("enable_thinking", False)) \
            if getattr(self, "generator_cfg", None) else False

        # ---- build once, no forwards ------------------------------------------------
        t0 = time.time()
        _rt = bool(cfg.get("require_thought", False))
        _rte = bool(cfg.get("require_thought_eval", False))
        train_steps = build_flat_steps(env, "train", hf_tokenizer, enable_thinking,
                                       cfg.get("max_steps_per_traj", None), _rt, _rte)
        eval_steps = build_flat_steps(eval_env, "eval", hf_tokenizer, enable_thinking,
                                      cfg.get("max_steps_per_traj", None), _rt, _rte)
        print(f"[sft_flat] built {len(train_steps)} train / {len(eval_steps)} eval steps "
              f"in {time.time() - t0:.1f}s (no model forwards); "
              f"steps_per_batch={steps_per_batch} micro_bs={micro_bs}")

        rng = np.random.default_rng(cfg.get("seed", 0))
        order = rng.permutation(len(train_steps))
        cursor = 0
        epoch = 0

        for batch_idx in range(num_steps):
            metrics: dict[str, Any] = {
                "progress/batch": batch_idx,
                "optim/lr": cfg.learning_rate,
                "progress/done_frac": (batch_idx + 1) / num_steps,
                "progress/epoch": epoch,
            }
            _is_last = batch_idx == num_steps - 1

            # ---- eval ----------------------------------------------------------------
            _is_eval = (eval_every > 0 and batch_idx % eval_every == 0) or _is_last
            if batch_idx == 0 and cfg.get("no_initial_eval", False) and not _is_last:
                _is_eval = False
            if _is_eval and eval_steps:
                with timed("run_evals_eval", metrics):
                    metrics.update(self._eval_flat(llm, eval_steps, cfg, micro_bs))
                # best_metric_name is e.g. "eval_actiononly_ppl" and our keys are
                # "eval/eval_actiononly_ppl", so match by substring as rl.py:294 does
                # rather than gluing prefixes (which double-prefixed to eval/eval_eval_*).
                _cands = [k for k in metrics if self.best_metric_name in k and k.startswith("eval/")]
                cur = metrics[_cands[0]] if _cands else None
                if cur is not None and is_better(cur, self.best_metric, self.best_metric_name):
                    self.best_metric, self.best_metric_step = cur, batch_idx
                    save_lora(llm.model, join(self.checkpoint_path, "step_best"))
                    metrics[f"eval/{self.best_metric_name}_best"] = cur
                    metrics[f"eval/{self.best_metric_name}_best_step"] = batch_idx
                    self._no_improve_evals = 0
                elif cur is not None:
                    self._no_improve_evals += 1
                metrics["eval/no_improve_evals"] = self._no_improve_evals

            # ---- corpus-wide sampling ------------------------------------------------
            if cursor + steps_per_batch > len(order):
                epoch += 1
                # Re-permute with a DIFFERENT stream each epoch. environments/base.py used
                # to re-seed with a constant, making every epoch's order identical; keep
                # this generator stateful so successive permutations actually differ.
                order = rng.permutation(len(train_steps))
                cursor = 0
            batch_steps = [train_steps[i] for i in order[cursor:cursor + steps_per_batch]]
            cursor += steps_per_batch

            llm.model.train()
            with timed("train_update", metrics):
                loader, _mb = prepare_minibatch(
                    new_trajectories=_as_single_step_trajectories(batch_steps),
                    hf_tokenizer=hf_tokenizer,
                    batch_size=micro_bs,
                    batch_idx=batch_idx,
                    max_seq_len=cfg.get("max_seq_len", 32768),
                    train_action_only=cfg.get("train_action_only", False),
                    shuffle=True,
                )
                accum = max(1, len(loader))  # one optimizer step per batch of steps
                optimizer.zero_grad()
                agg: dict[str, float] = {}
                for mb in loader:
                    lm = self.compute_loss(llm.model, mb, fp32_loss=self.fp32_loss)
                    (lm["loss"] / accum).backward()
                    for k, v in lm.items():
                        agg[k] = agg.get(k, 0.0) + float(v) / accum
                optimizer.step()
                optimizer.zero_grad()
                metrics.update({f"train/{k}": v for k, v in agg.items()})
                metrics["train/n_steps_in_batch"] = len(batch_steps)

            if cfg.get("keep_step_checkpoints", True) and (batch_idx + 1) % cfg.get("save_every", 20) == 0:
                snap = join(self.checkpoint_path, f"step_{batch_idx + 1:04d}")
                os.makedirs(snap, exist_ok=True)
                save_lora(llm.model, snap)

            self.ml_logger.log_metrics(dict(metrics))

            patience = int(cfg.get("early_stop_patience", 0) or 0)
            if patience > 0 and self._no_improve_evals >= patience and not _is_last:
                print(f"[sft_flat] EARLY STOP at batch {batch_idx}: "
                      f"{self.best_metric_name} not improved for {self._no_improve_evals} evals "
                      f"(best={self.best_metric:.4f} @ step {self.best_metric_step})")
                break

        save_lora(llm.model, join(self.checkpoint_path, "step_last"))
        return llm
