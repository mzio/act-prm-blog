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
            "advantage": mean_advantage,
            "num_gen_tokens": num_gen_tokens,
        }

    def train(
        self,
        llm: HuggingFaceLLM | None = None,
        optimizer: Optimizer | Any | None = None,
        cfg: DictConfig | None = None,
        env: Environment | None = None,
        eval_env: Environment | None = None,
        eval_every: int | None = None,
        # Specify training duration
        num_steps: int | None = None,
        num_substeps: int | None = None,
        # Other identifiers
        checkpoint_name: str | None = None,
        name_or_identifier: str | None = None,
        **generate_and_save_trajectories_kwargs: Any,
    ) -> HuggingFaceLLM:
        """
        Implement SFT training loop while adhering to RLTrainer interface.

        Assumes `env` is an ActPrmTracesEnv with `keep_expert_trajectories` is True
        for ActPRM (`thoughts_policy`, `thoughts_base`) and `expert_thoughts` runs,
        and False for `actions_only` runs.


        Given the Act-PRM environment with full (thought)-action trajectories, we:
        1. 
        
        entire Policy Gradient training loop for Hugging Face Transformer (PEFT) model (llm.model)
        """
        llm = llm or self.llm
        optimizer = optimizer or self.optimizer

        cfg = cfg or self.cfg
        env = env or self.env
        eval_env = eval_env or self.eval_env
        # Evaluation
        hf_tokenizer = self.hf_tokenizer
        # Batch iterations to evaluate on
        eval_every = eval_every or cfg.eval_every

        # 1. Determine mechanical dataset batch size and number of epochs
        num_steps = num_steps or cfg.get("num_steps", None) or cfg.num_batches
        num_substeps = (
            num_substeps or cfg.num_substeps
        )  # number of effective gradient updates per sampling batch
        dataloader_batch_size = (
            1 if cfg.get("group_size", 1) == 1 else 2
        )  # HF behavior w/ batches and padding, also GPU poor
        # MZ 03/07/2026: just set this to 1 for now
        dataloader_batch_size = 1

        # 0, not len(train): initialising to the pool size meant the condition
        # `rl_start_idx + batch_size > wen_shuffle` stayed false for the whole FIRST
        # epoch, so epoch 1 always ran in raw on-disk pool order and the first shuffle
        # only fired at the epoch-1/2 boundary.
        wen_shuffle = 0

        for batch_idx in range(0, num_steps):
            metrics = {
                "progress/batch": batch_idx,
                "optim/lr": cfg.learning_rate,
                "progress/done_frac": (batch_idx + 1) / num_steps,
            }
            t_start = time.time()

            # Run evaluations (skip the step-0 eval when --no_initial_eval, unless it's
            # also the final step). Final step always evals.
            _is_last = batch_idx == num_steps - 1
            _is_eval_step = (eval_every > 0 and batch_idx % eval_every == 0) or _is_last
            if batch_idx == 0 and cfg.get("no_initial_eval", False) and not _is_last:
                _is_eval_step = False
            if _is_eval_step:
                llm.model.eval()
                eval_rollout_metrics = {}
                for eval_split in getattr(eval_env, "eval_splits", ["eval"]):
                    with timed(f"run_evals_{eval_split}", metrics):
                        eval_env.split = eval_split

                        # Collect eval "rollouts" by selecting from env trajectories
                        _split_trajs = []
                        for _, sample_id in enumerate(range(0, len(eval_env))):
                            _traj = eval_env.get_trajectory(sample_id, eval_split)
                            _split_trajs.append(_traj)
                        #^Doesn't matter for eval bc we're computing offline eval metrics
                        # over the entire eval-set. But for training, we should consider
                        # shuffling over each possible state-action (step)

                        # TODO: incorporate the below into `compute_loss` override here,
                        # so we also have the compute_loss metrics (to maybe select with)
                        # Extra offline eval metrics (SFT: action-token PPL + accuracy
                        # over the target span). Best-effort: never crash the eval.
                        try:
                            _extra = self.eval_extra_metrics(
                                _split_trajs, split=eval_split, checkpoint_name=checkpoint_name
                            )
                            if _extra:
                                _split_metrics.update(_extra)
                                eval_rollout_metrics.update(_extra)
                        except Exception as _ee:  # noqa: BLE001
                            logger.warning(
                                "eval_extra_metrics failed: %s: %s", type(_ee).__name__, _ee
                            )
                        display_metrics(
                            _split_metrics,
                            title=f"Rollout {eval_split.title()} Metrics, Step {batch_idx}",
                            style="bright_yellow",
                        )
                metrics.update(eval_rollout_metrics)

                # Save best checkpoints
                best_metric_key = [k for k in eval_rollout_metrics.keys() if self.best_metric_name in k][0]
                last_metric = eval_rollout_metrics[best_metric_key]
                if is_better(last_metric, self.best_metric, self.best_metric_name):
                    self.best_metric = last_metric
                    self.best_metric_step = batch_idx
                    save_lora(llm.model, self.best_checkpoint_path)
                    logger.info(
                        f"RL EVAL (Step {batch_idx}): Updated best metric to {last_metric} at step {batch_idx}"
                    )
                    metrics.update(
                        {
                            f"eval/{self.best_metric_name}": last_metric,
                            f"eval/{self.best_metric_name}_best": self.best_metric,
                            f"eval/{self.best_metric_name}_best_step": self.best_metric_step,
                        }
                    )
                    try:  # Saving replay buffer
                        self.replay_buffer.save_hf_dataset_to_disk(self.best_replay_buffer_path)
                        logger.info(
                            "Saved best replay buffer to %s",
                            self.best_replay_buffer_path,
                        )
                    except SchemaInferenceError:
                        logger.warning(
                            "Failed to save best replay buffer to %s\nIs replay buffer empty?",
                            self.best_replay_buffer_path,
                        )

                # Track consecutive non-improving evals for early stopping.
                _patience = int(cfg.get("early_stop_patience", 0) or 0)
                if _patience > 0:
                    if self.best_metric_step == batch_idx:
                        self._no_improve_evals = 0
                    else:
                        self._no_improve_evals += 1
                    metrics["eval/no_improve_evals"] = self._no_improve_evals

                # Early flush: persist eval metrics now so a crash later in
                # this batch doesn't lose the eval snapshot. The end-of-batch
                # log_metrics call appends a second row with train/loss + timing
                # on the happy path; downstream analysis should dedupe by
                # progress/batch and keep the last entry.
                try:
                    self.ml_logger.log_metrics(dict(metrics))
                except Exception as _flush_exc:  # noqa: BLE001
                    logger.warning(
                        "early eval-metrics flush failed: %s: %s",
                        type(_flush_exc).__name__,
                        _flush_exc,
                    )

                # Early stop: eval best_metric hasn't improved for `early_stop_patience`
                # evals. step_best is already saved above, so we lose nothing by stopping.
                if _patience > 0 and self._no_improve_evals >= _patience and not _is_last:
                    logger.info(
                        "EARLY STOP at step %d: eval %s not improved for %d evals "
                        "(best=%.4f @ step %d).",
                        batch_idx, self.best_metric_name, self._no_improve_evals,
                        float(self.best_metric), self.best_metric_step,
                    )
                    break

            # Generate and save trajectories to a HF Dataset
            _save_rollouts_every = cfg.get("save_rollouts_every", num_steps)
            do_save_rollouts = _save_rollouts_every > 0 and (
                (batch_idx + 1) % _save_rollouts_every == 0 or (batch_idx + 1 == num_steps)
            )
            if do_save_rollouts:
                self.generate_and_save_trajectories(
                    cfg=cfg,
                    save_batch_idx=batch_idx,
                    save_generator_constructor=self.rl_generator_constructor,
                    **generate_and_save_trajectories_kwargs,
                )

            # NOTE: no_train (relabel / rollout-only) is handled AFTER the train-set
            # rollouts are generated + saved below — otherwise `continue` here would
            # skip train-set generation and the relabel would export ONLY eval
            # trajectories (corpus train:0). See the split-coverage fix.
            # 1. Sample rollouts for training
            # (no_train is handled AFTER generation below, so relabel/rollout-only mode
            #  still generates + saves TRAIN rollouts to generations.jsonl — it must not
            #  skip this, or the exported corpus would contain only eval trajectories.)
            env.split = "train"
            rl_start_idx = batch_idx * cfg.batch_size
            # Do NOT shuffle in no_train (relabel/export) mode: export_sft_corpus maps
            # each generation back via pool[sample_id % n] on the UNSHUFFLED pool, so a
            # mid-run shuffle would graft thoughts onto the wrong task's observations.
            # Keeping order aligned lets any num_batches produce a correct corpus
            # (wrap-around across tasks just yields clean augmentation). Training runs
            # (no_train=False) still shuffle as before.
            if not cfg.get("no_train", False) and rl_start_idx + cfg.batch_size > wen_shuffle:
                env.shuffle(split="train")
                wen_shuffle += len(env.datasets["train"])

            with timed("train_rollouts", metrics):
                train_rollout_metrics, new_trajectories = self._dispatch_rollouts(
                    llm=llm,
                    hf_tokenizer=hf_tokenizer,
                    generator_constructor=self.rl_generator_constructor,
                    env=env,
                    cfg=cfg,
                    batch_id=batch_idx,
                    checkpoint_name=checkpoint_name,
                    split="train",
                    num_tries=cfg.num_tries,
                    start_idx=rl_start_idx,
                    tasks_per_update=cfg.batch_size,
                    name_or_identifier=name_or_identifier,
                )
            metrics.update(train_rollout_metrics)
            display_metrics(
                train_rollout_metrics,
                title=f"Rollout Training Metrics, Step {batch_idx}",
                style="bright_cyan",
            )

            self.replay_buffer.save_hf_dataset_to_disk(self.last_replay_buffer_path)

            # Relabel / rollout-only mode: TRAIN rollouts are now generated AND saved
            # to generations.jsonl with this fixed checkpoint — skip only the optimizer
            # step (so the exported SFT corpus covers train + eval, not eval-only).
            if cfg.get("no_train", False):
                continue

            # 2. Update policy LLM with generated rollouts
            _t_optim = time.time()
            llm.model.train()

            train_trajectories = []  # Get the trajectories we'll train on
            for k, v in new_trajectories.items():
                if k.startswith("policy"):
                    train_trajectories.extend(v)

            train_loader, _minibatch_metrics = self.prepare_minibatch(
                new_trajectories=train_trajectories,
                hf_tokenizer=hf_tokenizer,
                batch_size=dataloader_batch_size,
                shuffle=True,
                batch_idx=batch_idx,  # for debugging
                max_seq_len=cfg.get("max_seq_len", 32768),
                drop_zero_advantage=cfg.get("drop_zero_advantage", False),
                train_action_only=cfg.get("train_action_only", False),
            )
            metrics.update(_minibatch_metrics)  # empty {} for now

            # For now, auto-calculate gradient accumulation steps based on num_substeps
            gradient_accumulation_steps = max(1, len(train_loader) // num_substeps)
            pbar_substep = tqdm(total=num_substeps, desc="Number of substeps", colour="blue", position=2)
            pbar_dataloader = tqdm(train_loader, desc="Dataloader batches", colour="cyan", position=3)

            for mini_batch_idx, mini_batch in enumerate(pbar_dataloader):
                # Sanity-check model inputs
                if mini_batch_idx == 0 or (mini_batch_idx + 1) % 10 == 0:
                    self._check_model_inputs(mini_batch, hf_tokenizer, cfg)

                loss_metrics = self.compute_loss(llm.model, mini_batch, fp32_loss=self.fp32_loss)
                loss = loss_metrics["loss"]
                loss = loss / gradient_accumulation_steps
                loss.backward()

                if (mini_batch_idx + 1) % gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    pbar_substep.update(1)

                loss_metrics = {f"train/{k}": get_item(v) for k, v in loss_metrics.items()}
                metrics.update(loss_metrics)
                pbar_dataloader.set_postfix(**loss_metrics)

            metrics["time/optim"] = time.time() - _t_optim

            # Periodic rolling LoRA checkpoint for stop/restart (every save_every steps; the
            # replay buffer is already saved every step above). Best checkpoint still saved on
            # eval-improvement. Resume by starting from step_last (--lora_checkpoint_path).
            _save_every = int(cfg.get("save_every", 0) or 0)
            if _save_every > 0 and ((batch_idx + 1) % _save_every == 0 or _is_last):
                save_lora(llm.model, self.last_checkpoint_path)
                logger.info(
                    "Saved rolling checkpoint (step %d) -> %s", batch_idx, self.last_checkpoint_path
                )
                # ALSO keep a numbered snapshot. step_last is overwritten every save_every
                # batches, and step_best is only meaningful if evals are frequent -- with
                # eval_every == num_batches they both end up being the FINAL model. That cost
                # us the batch-30 checkpoint of the 08-26 retail run, whose generations show
                # 0% degenerate thoughts at batch 30 versus 41% by batch 79: the good model
                # existed and was overwritten by the hacked one.
                # ~253MB per snapshot at r32; set keep_step_checkpoints=false to disable.
                if cfg.get("keep_step_checkpoints", True):
                    _snap = join(self.checkpoint_path, f"step_{batch_idx + 1:04d}")
                    os.makedirs(_snap, exist_ok=True)
                    save_lora(llm.model, _snap)
                    logger.info("Saved numbered snapshot -> %s", _snap)

            # Log metrics
            try:
                metrics["time/total"] = time.time() - t_start
                self.ml_logger.log_metrics(metrics)  # increments each time
            except Exception as e:
                _error_class = e.__class__.__name__
                _error_message = str(e)
                rich_print(f"[red]Error logging metrics: {_error_class}: {_error_message}[/red]")
                for k, v in metrics.items():
                    print(k, type(v))
                # headless-safe: a metric-logging failure must not kill training
            torch.cuda.empty_cache()

        # Load best model checkpoint
        llm.model = load_lora(llm.model, self.best_checkpoint_path)
        return llm

    def _dispatch_rollouts(
        self,
        env: Environment,
        cfg: DictConfig,
        batch_id: int,
        checkpoint_name: str | None = None,
        split: str = "train",
        num_tries: int = 1,
        start_idx: int = 0,
        tasks_per_update: int | None = None,  # i.e., batch_size
        name_or_identifier: str | None = None,
        **kwargs
    ):
        """
        For SFT traiining, we assume env is an ActPrmTracesEnv, so we have full (thought)-action
        trajectories already for training. We "generate rollouts" by sampling from these
        """
        trajectory = env.get_trajectory(start_idx, split)

    def run_rollouts(
    llm: HuggingFaceLLM,
    hf_tokenizer: PreTrainedTokenizerBase,
    generator_constructor: Callable[..., HuggingFaceGenerator],
    env: Environment,
    cfg: DictConfig,
    batch_id: int,
    checkpoint_name: str | None = None,
    split: str = "train",
    num_tries: int = 1,
    start_idx: int = 0,
    tasks_per_update: int | None = None,  # i.e., batch_size
    name_or_identifier: str | None = None,
    # Overrides for generation
    max_tokens: int | None = None,
    temperature: float | None = None,
    pbar_position: int = 0,
    num_return_sequences: int | None = None,
) -> tuple[dict[str, Any], dict[str, list[Trajectory]]]:
    """
    Run rollouts for a single batch, e.g., by generating rollouts and grading them

    Returns:
    - final_metrics: Metrics for the batch, keyed by "{split}/{try_idx}/{metric}"
    - new_trajectories: Trajectories for the batch, keyed by an identifier (default "policy")
    """
    num_tries = num_tries or 1  # guard: eval/train configs may leave (eval_)num_tries unset
    _has_torch_model = hasattr(llm, "model") and hasattr(getattr(llm, "model", None), "training")
    if _has_torch_model:
        was_training = llm.model.training
    else:
        was_training = False

    ctx = torch.no_grad() if _has_torch_model else contextlib.nullcontext()
    with ctx:
        if _has_torch_model:
            llm.model.eval()
        env.split = split  # Select task split

        generator = generator_constructor(
            llm=llm,
            hf_tokenizer=hf_tokenizer,
            env=env,
            cfg=cfg,
            enable_thinking=cfg.get("enable_thinking", False),
            name_or_identifier=name_or_identifier,
        )
        batch_size = tasks_per_update or len(env)  # len(env) is the number of tasks or problems
        num_return_sequences = num_return_sequences or (
            cfg.group_size if split == "train" else cfg.eval_group_size
        )
        all_eval_metrics = {}
        keys_for_correct = []
        eval_metric_keys = [
            "final_reward",
            "first_return",
            "action_prob",
            "last_state_len",
            "timesteps",
            "correct",
            "match_rate",
            "total",
        ]
        # Store new trajectories to return
        new_trajectories: dict[str, list[Trajectory]] = {}
        all_trajectory_groups: list[dict[str, list[TrajectoryGroup]]] = []

        for try_idx in range(num_tries):
            pbar_desc = f"Generating {num_return_sequences} rollouts for sample {start_idx + 1} / {start_idx + batch_size}"
            sample_pbar = tqdm(
                range(start_idx, start_idx + batch_size),
                desc=pbar_desc,
                colour="blue",
                leave=True,
                position=pbar_position,
            )
            n_success_so_far = 0
            n_total_so_far = 0
            for _, sample_id in enumerate(sample_pbar):
                group_dict = generator.do_group_rollout(
                    env=env,
                    sample_id=sample_id,
                    batch_id=batch_id,
                    split=split,
                    try_step=try_idx,
                    num_return_sequences=num_return_sequences,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    pbar_position=pbar_position + 1,
                )
                all_trajectory_groups.append(group_dict)
                # Tally successes for the live success/total pbar postfix.
                # Use the primary trajectory key ("policy" if present, else the
                # first one returned). "correct" is an int/bool on Trajectory.
                _primary_key = "policy" if "policy" in group_dict else next(iter(group_dict), None)
                if _primary_key is not None:
                    for _tg in group_dict.get(_primary_key, []) or []:
                        for _traj in getattr(_tg, "trajectories", []) or []:
                            n_total_so_far += 1
                            if int(getattr(_traj, "correct", 0) or 0) >= 1:
                                n_success_so_far += 1
                pbar_desc = f"Generating {num_return_sequences} rollouts for sample {start_idx + 1} / {start_idx + batch_size}"
                sample_pbar.set_description(pbar_desc)
                sample_pbar.set_postfix(success=f"{n_success_so_far}/{n_total_so_far}")
                # Per-task rollout record (one line per trajectory). Lets us slice a
                # run's eval by task subset, bootstrap CIs, and diff arms task-by-task
                # -- the aggregate metrics below throw all of that away.
                _dump_per_task_records(
                    cfg=cfg,
                    env=env,
                    group_dict=group_dict,
                    sample_id=sample_id,
                    split=split,
                    batch_id=batch_id,
                    try_idx=try_idx,
                    checkpoint_name=checkpoint_name,
                )
            # End-of-try checkpoint: push the running rollouts buffer
            # to the hub (covers every rollout collected for try_idx).
            try:
                generator._maybe_save_rollouts()
            except Exception:
                pass  # already best-effort inside _maybe_save_rollouts

        # Save metrics and samples
        trajectory_keys = all_trajectory_groups[0].keys()
        _metric_prefix = f"{checkpoint_name}_{split}" if checkpoint_name is not None else split

        for _key in trajectory_keys:
            for trajectory_groups in all_trajectory_groups:  # list of list of trajectory groups
                for traj_group in trajectory_groups[_key]:  # len(trajectory_groups) usually 1,
                    for trajectory in traj_group.trajectories:  # can be >1, e.g., if step-wise adv
                        if _key == "policy":
                            # Only store metrics for the default "policy" trajectory group
                            _try_step = trajectory.try_step
                            for metric_key in eval_metric_keys:
                                _metric_key = f"{_metric_prefix}/try_{_try_step}/{metric_key}"
                                if metric_key == "correct":
                                    keys_for_correct.append(_metric_key)
                                if _metric_key not in all_eval_metrics:
                                    all_eval_metrics[_metric_key] = []
                                val = getattr(trajectory, metric_key, 1)  # 1 for total samples
                                all_eval_metrics[_metric_key].append(val)
                            # Also log flat env metrics surfaced on the trajectory
                            # (eval env: task_completion, task_personalization, num_respond_user).
                            for _mk, _mv in getattr(trajectory, "metrics", {}).items():
                                all_eval_metrics.setdefault(f"{_metric_prefix}/try_{_try_step}/{_mk}", []).append(_mv)
                        # Add trajectory to list of new trajectories
                        if _key not in new_trajectories:
                            new_trajectories[_key] = []
                        new_trajectories[_key].append(trajectory)

    final_metrics = {}  # return these metrics for the batch
    # 1. Compute aggregate metrics
    for k, v in all_eval_metrics.items():
        if "correct" in k or "total" in k:
            final_metrics[k] = np.sum(v).item()  # convert to float for json.dumps
        else:
            final_metrics[k] = np.mean(v).item()
        final_metrics[f"{k}_std"] = np.std(v).item()
        final_metrics[f"{k}_max"] = np.max(v).item()
    # 2. Add accuracy (dummy for Act-PRM training rollouts)
    for k in keys_for_correct:
        total_v = final_metrics[k.replace("correct", "total")]
        final_metrics[k.replace("correct", "accuracy")] = final_metrics[k] / total_v

    # 3. Add generator usage metrics (token counts)
    if hasattr(generator, "get_usage_metrics"):
        usage = generator.get_usage_metrics()
        for k, v in usage.items():
            final_metrics[f"usage/{k}"] = v

    if _has_torch_model and was_training:
        llm.model.train()

    return final_metrics, new_trajectories


__all__ = ["SFTTrainer"]
