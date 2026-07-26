"""
Parent class for synchronous PyTorch (LoRA) policy-gradient trainers.

This is the trainable HuggingFace counterpart to the async ``BaseTrainer`` in
``base.py`` (which drives the API / Tinker generators). The concrete trainers
``RLTrainer`` / ``RLBatchTrainer`` / ``SFTTrainer`` subclass this and are wired
through ``get_trainer`` (see ``__init__.py``) from ``main_pytorch.py``.

Ported from the pre-refactor ``strl.pytorch.trainer.base`` with imports updated
to the current package layout (generators now live in ``strl.generator``).
"""

import logging
import sys
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Callable

import torch
from omegaconf import DictConfig
from rich import print as rich_print
from rich.console import Console
from tinker_cookbook.utils import ml_log
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase

from act_prm.environments import Environment
from act_prm.generator import get_generator_constructor
from act_prm.generator.huggingface.base import HuggingFaceGenerator
from act_prm.llm_handlers import HuggingFaceLLM
from act_prm.lora.checkpoint import push_lora_to_hub
from act_prm.replay_buffer import ReplayBuffer

from ..train import hide_observations, prepare_minibatch

logger = logging.getLogger(__name__)
console = Console()


class BaseTrainer(ABC):
    """
    Parent class for synchronous PyTorch trainers (Hugging Face Transformers / PEFT models)
    """

    def __init__(
        self,
        cfg: DictConfig,
        llm: HuggingFaceLLM,
        optimizer: Optimizer | Any | None,  # None for API-only models
        generator_cfg: DictConfig,
        replay_buffer: ReplayBuffer,
        env: Environment,
        eval_env: Environment,
        ml_logger: ml_log.Logger,
        hf_tokenizer: PreTrainedTokenizerBase | None = None,
        checkpoint_path: str | None = None,
        log_path: str | None = None,
        **kwargs: Any,
    ) -> None:
        self.cfg = cfg
        self.llm = llm
        self.optimizer = optimizer
        self.generator_cfg = generator_cfg
        self.replay_buffer = replay_buffer
        self.env = env
        self.eval_env = eval_env
        self.ml_logger = ml_logger
        self.hf_tokenizer = hf_tokenizer
        self.fp32_loss = cfg.get("fp32_loss", False)

        # If True, we hide observations other than the last one to avoid context blow-up
        # -> See hide_observations() in ./train.py for more details
        self.hide_observations = cfg.get("hide_observations", False)
        self.hidden_obs_content = cfg.get("hidden_obs_content", "...")

        # RL / Evaluation generator: does standard rollouts, see ../generator/pytorch/base.py
        self.rl_generator_constructor = self.get_generator_constructor(**self.generator_cfg)
        self.best_metric = 1e8 if "loss" in cfg.best_metric else -1e8

        self.run_name = cfg.run_name
        self.run_url = ml_logger.get_logger_url() if ml_logger is not None else None
        self.run_cmd = " ".join(sys.argv)

        # Logging and checkpointing
        self.checkpoint_path = checkpoint_path or cfg.checkpoint_path
        self.log_path = log_path or cfg.log_path

        # Hub push config
        self.push_to_hub = cfg.get("push_to_hub", True)
        self.hub_model_prefix = cfg.get("hub_model_prefix", "mzio")

    def get_generator_constructor(self, **kwargs: Any) -> Callable[..., HuggingFaceGenerator]:
        """
        Get a (partially initialized) Hugging Face Generator constructor by name
        """
        return get_generator_constructor(
            **kwargs, ml_logger=self.ml_logger, cfg=self.cfg, replay_buffer=self.replay_buffer
        )

    def save_and_push_best_lora(
        self,
        model: "torch.nn.Module",
        checkpoint_path: str,
        step: int,
        metric_name: str,
        metric_value: float,
        tag: str = "best",
    ) -> str | None:
        """
        Save LoRA checkpoint locally and optionally push to HuggingFace Hub.
        Returns the hub URL if pushed, else None.
        """
        from act_prm.lora import save_lora

        save_lora(model, checkpoint_path)
        if not self.push_to_hub:
            return None

        # Build hub repo ID from run_name
        # e.g., mzio/strl-lora-Eaprm_tw_tr_medium-S42-R0-best
        try:
            run_name = self.run_name
            parts = []
            for label, delim in [("E", "ec="), ("S", "s="), ("R", "r=")]:
                val = run_name.split(delim)[-1].split("-")[0]
                val = val.replace("act_prm_", "aprm_").replace("treasure_", "tr_").replace("coin_", "cc_")
                parts.append(f"{label}{val}")
            identifier = "-".join(parts)
        except Exception:
            identifier = "unknown"
        repo_id = f"{self.hub_model_prefix}/strl-lora-{identifier}-{tag}"

        # Sanitize
        repo_id = repo_id.replace("__", "_")[:96]

        return push_lora_to_hub(
            model,
            repo_id=repo_id,
            commit_message=f"Step {step}: {metric_name}={metric_value:.4f}",
        )

    def maybe_hide_observations(
        self,
        messages: list[dict[str, str]],
        hidden_obs_content: str | None = None,
        first_obs_to_show: int = 2,  # e.g., to keep prompt
        last_obs_to_show: int = 1,  # e.g., to keep last observation
    ) -> list[dict[str, str]]:
        """
        Maybe hide past observations from messages
        """
        if not self.hide_observations:
            return messages

        hidden_obs_content = hidden_obs_content or self.hidden_obs_content
        return hide_observations(messages, hidden_obs_content, first_obs_to_show, last_obs_to_show)

    def prepare_minibatch(self, **kwargs: Any) -> tuple[DataLoader, dict[str, Any]]:
        """
        Prepare a minibatch of trajectories for training
        """
        return prepare_minibatch(**kwargs)

    def _check_model_inputs(
        self,
        batch: dict[str, torch.Tensor],
        hf_tokenizer: PreTrainedTokenizerBase | None = None,
        cfg: DictConfig | None = None,
    ) -> None:
        """
        Sanity-check model inputs by rich printing them
        """
        hf_tokenizer = hf_tokenizer or self.hf_tokenizer
        cfg = cfg or self.cfg

        decoded_inputs = hf_tokenizer.batch_decode(batch["input_ids"][:, 1:])
        _labels = deepcopy(batch["labels"][:, 1:])
        _labels[_labels == -100] = 0  # -100 will cause tokenization errors
        decoded_labels = hf_tokenizer.batch_decode(_labels)
        advantages = batch.get("advantages")
        is_icls = batch.get("is_icl", [False] * len(decoded_inputs))
        for idx, decoded_input in enumerate(decoded_inputs):
            _is_icl = is_icls[idx].item()
            console.print(f"[cyan]Input {idx} (ICL: {_is_icl}):\n{decoded_input}\n[/cyan]")
            console.print(f"[green]Label {idx} (ICL: {_is_icl}):\n{decoded_labels[idx]}\n[/green]")
            if advantages is not None:
                _adv = advantages[idx]
                _adv = _adv.tolist() if isinstance(_adv, torch.Tensor) else _adv
                console.print(f"[yellow]Advantage {idx} (ICL: {_is_icl}):\n{_adv}\n[/yellow]")
            console.print("=" * 100)
        # Keep run url and cmd in display
        rich_print(f"[bold]Run url: [link={self.run_url}]{self.run_url}[/link][/bold]")
        rich_print(f"[bold]Run cmd: [bright_cyan]{self.run_cmd}[/bright_cyan][/bold]")

    def compute_loss(
        self,
        model: torch.nn.Module,
        batch: dict[str, torch.Tensor],
        fp32_loss: bool | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Compute loss for a batch of model inputs
        -> Simple mini-batch cross-entropy loss (no weights)
        """
        fp32_loss = fp32_loss or self.fp32_loss
        device = model.device
        model_inputs = {k: v.to(device) for k, v in batch.items() if k in ["input_ids", "attention_mask"]}
        weight = batch.get("weight", 1.0)  # ignored but report it
        logits = model(**model_inputs, use_cache=False).logits[:, :-1, :]
        labels = batch["labels"][:, 1:]
        vocab_size = logits.shape[-1]
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, vocab_size).to(dtype=torch.float32 if fp32_loss else logits.dtype),
            labels.view(-1).to(device),
            reduction="mean",
        ).to(dtype=logits.dtype)
        ppl = torch.exp(loss).detach().cpu()
        return {"loss": loss, "ppl": ppl, "weight": weight}

    @abstractmethod
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
        mini_batch_size: int | None = None,
        gradient_accumulation_steps: int | None = None,  # 1 if not specified here or in cfg
        # Other identifiers
        checkpoint_name: str | None = None,
        name_or_identifier: str | None = None,
        **kwargs: Any,
    ) -> HuggingFaceLLM:
        """
        Implement entire training loop
        """
        raise NotImplementedError
