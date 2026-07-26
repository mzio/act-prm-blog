"""
PyTorch RL Trainer for Hugging Face Transformers (PEFT / LoRA) models.

Ported from the pre-refactor ``strl.pytorch.trainer.rl`` with imports updated
to the current package layout: generators now live in ``strl.generator``
(``HuggingFaceGenerator`` / ``StrlHuggingFaceGenerator``), and the rollout +
LoRA helpers are siblings in ``strl.trainer``.
"""

import logging
import os
import random  # noqa: F401 -- parity with prior import surface
import time
from copy import copy
from os.path import join
from typing import Any, Callable

import torch
from datasets import Dataset
from datasets.arrow_writer import SchemaInferenceError
from omegaconf import DictConfig
from rich import print as rich_print
from rich.console import Console
from torch.nn import functional as F  # noqa: F401 -- used by SFTTrainer subclass
from torch.optim import Optimizer
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase

from act_prm.environments import Environment
from act_prm.generator.huggingface.base import HuggingFaceGenerator
from act_prm.llm_handlers import HuggingFaceLLM
from act_prm.lora import load_lora, save_lora
from act_prm.replay_buffer.types import Trajectory
from act_prm.utils.display import display_metrics
from act_prm.utils.logging import timed

from ..train import run_rollouts
from .pg_base import BaseTrainer

console = Console()
logger = logging.getLogger(__name__)


def get_item(x: Any) -> int | float:
    """
    Get scalar from torch tensor, numpy array, or scalar
    """
    try:
        return x.item() if hasattr(x, "item") else x
    except Exception:  # undetached Torch tensor?
        assert hasattr(x, "item")
        return x.detach().cpu().item()


def _lower_is_better(metric: str) -> bool:
    """True for metrics where smaller is better (loss / perplexity)."""
    m = metric.lower()
    return "loss" in m or "ppl" in m or "perplexity" in m


def is_better(x: float, y: float, metric: str) -> bool:
    """
    Determine if x is better than y for a given metric
    """
    return x <= y if _lower_is_better(metric) else x >= y


class RLTrainer(BaseTrainer):
    """
    PyTorch trainer for policy gradient with Hugging Face Transformers models
    """

    def __init__(
        self,
        cfg: DictConfig,
        checkpoint_path: str | None = None,
        log_path: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(cfg=cfg, checkpoint_path=checkpoint_path, log_path=log_path, **kwargs)

        # Checkpointing and best metrics
        self.best_checkpoint_path = join(self.checkpoint_path, "step_best")
        if not os.path.exists(self.best_checkpoint_path):
            os.makedirs(self.best_checkpoint_path)
        # Rolling latest checkpoint (saved every `save_every` steps) for stop/restart.
        self.last_checkpoint_path = join(self.checkpoint_path, "step_last")
        if not os.path.exists(self.last_checkpoint_path):
            os.makedirs(self.last_checkpoint_path)
        self.best_replay_buffer_path = join(self.checkpoint_path, "replay_buffer_best")
        self.last_replay_buffer_path = join(self.checkpoint_path, "replay_buffer")

        self.best_metric_name = cfg.best_metric
        # Best direction depends on the metric: loss / perplexity are lower-is-better
        # (e.g. best_metric: eval_action_ppl for SFT early-stop), else higher-is-better.
        self.best_metric = float("inf") if _lower_is_better(self.best_metric_name) else float("-inf")
        self.best_metric_step = -1

    def compute_loss(
        self,
        model: torch.nn.Module,
        batch: dict[str, torch.Tensor],
        fp32_loss: bool | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Compute policy gradient loss for a batch of model inputs
        -> Assumes there is always a non-zero "state" or "prefix" length, i.e.,
           we compute loss on the first "action" or "target" token
        """
        fp32_loss = fp32_loss or self.fp32_loss
        device = model.device

        # Get advantages, logprobs, label mask (advantages are 0 for non-label tokens)
        # -> Next-token shifted as targets, see ./train.py prepare_minibatch()
        advantages = batch["advantages"].to(device)
        old_logprobs = batch["logprobs"].to(device) if "logprobs" in batch else None
        label_mask = batch["label_mask"].to(device)

        # Compute on-policy logprobs
        model_inputs = {k: v.to(device) for k, v in batch.items() if k in ["input_ids", "attention_mask"]}
        logits = model(**model_inputs, use_cache=False).logits[:, :-1, :]
        labels = model_inputs["input_ids"][:, 1:]
        dtype = torch.float32 if fp32_loss else model.dtype
        # Xent needs (N, C, d) inputs and (N, d) labels,
        # -> So we transpose from (B, L-1, V) -> (B, V, L-1)
        new_logprobs = -F.cross_entropy(
            logits.transpose(1, 2).to(dtype=dtype),
            labels,
            reduction="none",
        ).to(dtype=logits.dtype)

        # Get importance-weighted loss; non-surrogate form for clarity
        try:
            if old_logprobs is not None:
                ratio = torch.exp(new_logprobs.detach() - old_logprobs.to(device))
            else:
                ratio = 1.0  # torch.exp(0)
        except Exception as e:
            print(f"{e.__class__.__name__}: {e}")
            print(f"new_logprobs: {new_logprobs.shape}")
            print(f"old_logprobs: {old_logprobs.shape}")
            breakpoint()
        num_label_tokens = label_mask.sum().clamp_min(1)
        loss = -(ratio * new_logprobs * advantages).sum() / num_label_tokens

        # Compute additional logging metrics
        ppl = torch.exp(-(new_logprobs * label_mask).sum() / num_label_tokens).item()
        mean_advantage = (advantages.sum() / num_label_tokens).item()  # advantages already padded
        num_gen_tokens = label_mask.sum(dim=-1).tolist()  # average generated tokens per sample
        num_gen_tokens = sum(num_gen_tokens) / len(num_gen_tokens)

        del advantages, old_logprobs, model_inputs, logits, labels, num_label_tokens
        torch.cuda.empty_cache()

        return {
            "loss": loss,
            "ppl": ppl,
            "advantage": mean_advantage,
            "num_gen_tokens": num_gen_tokens,
        }

    def eval_extra_metrics(
        self,
        trajectories: Any,
        split: str = "eval",
        checkpoint_name: str | None = None,
    ) -> dict[str, float]:
        """Hook for extra offline eval metrics beyond the rollout metrics.

        Default: none. :class:`SFTTrainer` overrides this to add action-token
        perplexity + accuracy over the eval target span. Returned keys are merged
        into the eval metrics dict (and so are eligible for ``best_metric``).
        """
        return {}

    def _dispatch_rollouts(self, **kwargs):
        """Hook for subclasses: which run_rollouts variant to call.

        Default forwards to :func:`strl.trainer.train.run_rollouts` (per-sample
        loop). :class:`strl.trainer.rl_batch.RLBatchTrainer` overrides this to
        use ``run_batch_rollouts`` (sample-batched). Same return signature:
        ``(metrics_dict, trajectories_dict)``.
        """
        return run_rollouts(**kwargs)

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
        Implement entire Policy Gradient training loop for Hugging Face Transformer (PEFT) model (llm.model)
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

        wen_shuffle = len(env.datasets["train"])

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
                        _split_metrics, _split_trajs = self._dispatch_rollouts(
                            llm=llm,
                            hf_tokenizer=hf_tokenizer,
                            generator_constructor=self.rl_generator_constructor,
                            env=eval_env,
                            split=eval_split,
                            cfg=cfg,
                            batch_id=batch_idx,
                            checkpoint_name=checkpoint_name,
                            # eval_num_tries may be unset in the trainer config
                            # (and --num_tries only sets num_tries); fall back so
                            # eval doesn't crash on range(None).
                            num_tries=cfg.eval_num_tries or cfg.num_tries or 1,
                            start_idx=0,
                            tasks_per_update=len(eval_env),
                            name_or_identifier=name_or_identifier,
                        )
                        eval_rollout_metrics.update(_split_metrics)
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

            # 1. Sample rollouts for training
            # (no_train is handled AFTER generation below, so relabel/rollout-only mode
            #  still generates + saves TRAIN rollouts to generations.jsonl — it must not
            #  skip this, or the exported corpus would contain only eval trajectories.)
            env.split = "train"
            rl_start_idx = batch_idx * cfg.batch_size
            if rl_start_idx + cfg.batch_size > wen_shuffle:
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

            # Relabel / rollout-only mode: train rollouts have now been generated AND saved
            # (generations.jsonl) with this fixed checkpoint — skip only the optimizer step.
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
                breakpoint()
            torch.cuda.empty_cache()

        # Load best model checkpoint
        llm.model = load_lora(llm.model, self.best_checkpoint_path)
        return llm

    def generate_and_save_trajectories(
        self,
        save_generator_constructor: Callable[..., HuggingFaceGenerator],
        save_batch_idx: int,
        llm: HuggingFaceLLM | None = None,
        save_env: Environment | None = None,
        cfg: DictConfig | None = None,
        hf_tokenizer: PreTrainedTokenizerBase | None = None,
        save_name_or_identifier: str | None = None,
        trajectory_key: str = "policy",
        dataset_prefix: str = "mzio/strl-sft_rollouts",
        dataset_suffix: str = "",
        split: str | None = None,
    ) -> list[list[Trajectory]]:
        """
        Generate trajectories for all tasks in an environment
        -> We create a "TrajectoryGroup" of `group_size` for each task
        """
        llm = llm or self.llm
        env = save_env or self.env
        cfg = cfg or self.cfg
        hf_tokenizer = hf_tokenizer or self.hf_tokenizer

        if split is not None:
            dataset_suffix = f"{dataset_suffix}-{split}" if len(dataset_suffix) > 0 else f"-{split}"
        split = split or "train"

        env.split = split
        was_training = copy(llm.model.training)
        llm.model.eval()

        logger.info("Generating trajectories for all %d tasks in the environment", len(env))
        _, new_trajectories = self._dispatch_rollouts(
            llm=llm,
            hf_tokenizer=hf_tokenizer,
            generator_constructor=save_generator_constructor,
            env=env,
            cfg=cfg,
            batch_id=save_batch_idx,
            checkpoint_name="sft_gen",
            split=split,
            num_tries=cfg.num_tries,
            start_idx=0,  # Sequentially generate trajectories for all tasks
            tasks_per_update=len(env),
            name_or_identifier=save_name_or_identifier,
        )
        # Build ds_identifier from run_name, extracting abbreviated key=value pairs:
        #   E<env_config>-G<generator_config>-S<seed>-R<replicate>
        ds_identifier = "-".join(
            [
                f"{label}{self.run_name.split(delim)[-1].split('-')[0]}"
                for label, delim in [
                    ("E", "ec="),  # --env_config
                    ("G", "gc="),  # --generator_config
                    ("S", "s="),  # --seed
                    ("R", "r="),  # --replicate
                ]
            ]
        )
        if "joint" in self.run_name:
            ds_identifier += "-joint"
        # Always shorten common long substrings in the identifier
        ds_identifier = (
            ds_identifier.replace("act_prm_", "aprm_").replace("treasure_", "tr_").replace("coin_", "cc_")
        )
        ds_name = f"{dataset_prefix}-{ds_identifier}{dataset_suffix}-b{save_batch_idx:03d}"
        if len(ds_name) > 96:  # additional shortening if still too long
            ds_name = ds_name.replace("tau_bench", "tau").replace("act_prm", "aprm").replace("qwen3", "qw3")
        ds_name = _sanitize_hf_repo_name(ds_name)
        cfg.dataset_url_sft = f"https://huggingface.co/datasets/{ds_name}"

        # Build metadata for the dataset card
        _trajectories = new_trajectories[trajectory_key]
        metadata = {
            "run_name": self.run_name,
            "run_cmd": self.run_cmd,
            "env_config": cfg.get("env_config", None),
            "generator_config": cfg.get("generator_config", None),
            "trainer_config": cfg.get("trainer_config", None),
            "model_name": cfg.get("model_name", None),
            "seed": cfg.get("seed", None),
            "replicate": cfg.get("replicate", None),
            "batch_idx": save_batch_idx,
            "split": split,
            "trajectory_key": trajectory_key,
            "num_trajectories": len(_trajectories),
            "num_episodes": sum(len(t.episode_steps) for t in _trajectories),
            "group_size": cfg.get("group_size", None),
            "batch_size": cfg.get("batch_size", None),
        }
        try:
            _save_trajectories_to_hf_dataset(_trajectories, ds_name, metadata=metadata)
            logger.info("Saved trajectories to HF Dataset: %s", cfg.dataset_url_sft)
        except Exception as e:
            _error_text = f"({type(e).__name__}: {e})"
            logger.error("Failed to save trajectories to HF Dataset: %s", _error_text)
            breakpoint()

        if was_training:
            llm.model.train()

        return new_trajectories[trajectory_key]


def _sanitize_hf_repo_name(name: str, max_length: int = 96) -> str:
    """
    Sanitize a HuggingFace repo name to comply with Hub restrictions:
    - Only alphanumeric, '-', '_', '.' allowed
    - '--' and '..' are forbidden
    - Cannot start or end with '-' or '.'
    - Max length is 96
    """
    import re

    # Keep only allowed characters
    name = re.sub(r"[^a-zA-Z0-9/_\-.]", "_", name)
    # Collapse '--' and '..' sequences
    while "--" in name:
        name = name.replace("--", "-")
    while ".." in name:
        name = name.replace("..", ".")
    # Strip leading/trailing '-' and '.' from the repo name part (after namespace/)
    if "/" in name:
        namespace, repo = name.split("/", 1)
        repo = repo.strip("-.")
        name = f"{namespace}/{repo}"
    else:
        name = name.strip("-.")
    # Truncate to max length
    if len(name) > max_length:
        name = name[:max_length].rstrip("-.")
    return name


def _save_trajectories_to_hf_dataset(
    trajectories: list[Trajectory],
    dataset_name: str,
    exclude_keys: list[str] | None = None,
    private: bool = False,
    metadata: dict[str, Any] | None = None,
) -> None:
    """
    Save a list of trajectories to a HF Dataset, with optional metadata in the README
    """
    exclude_keys = exclude_keys or ["state_action_tokens", "old_logprobs"]
    ds_samples = [
        {k: v for k, v in vars(step).items() if k not in exclude_keys}
        for trajectory in trajectories
        for step in trajectory.episode_steps
    ]  # Flatten to get dicts from list of EpisodeSteps in each Trajectory
    try:
        Dataset.from_list(ds_samples).push_to_hub(dataset_name, private=private)
        if metadata:
            _push_dataset_card(dataset_name, metadata)
    except Exception as e:
        _error_text = f"({type(e).__name__}: {e})"
        logger.error("Failed to save trajectories to HF Dataset: %s", _error_text)
        dataset_name = dataset_name.replace("/", "-")
        Dataset.from_list(ds_samples).save_to_disk(dataset_name)
        logger.info("Saved trajectories to local directory: %s", dataset_name)


def _push_dataset_card(
    dataset_name: str,
    metadata: dict[str, Any],
) -> None:
    """
    Push a DatasetCard (README.md) with metadata to the HuggingFace Hub
    """
    from huggingface_hub import DatasetCard, DatasetCardData

    # Standard HF card fields
    card_data = DatasetCardData(
        license="mit",
        tags=["strl", "rollouts"],
    )
    # Build card body with run metadata
    body_lines = ["# Show-Don't-Tell RL Rollout Dataset\n"]
    body_lines.append("## Run Metadata\n")
    for key, value in metadata.items():
        if value is not None:
            body_lines.append(f"- **{key}**: `{value}`")
    body = "\n".join(body_lines)

    card = DatasetCard.from_template(card_data, template_str=body)
    card.push_to_hub(dataset_name)
