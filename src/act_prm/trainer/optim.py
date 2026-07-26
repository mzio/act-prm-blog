"""
Optimizers and schedulers
"""

from typing import Any

import torch
import torch.nn as nn


def get_optimizer(
    model: nn.Module,
    name: str = "sgd",
    learning_rate: float = 4e-5,
    **kwargs: Any,
) -> Any:  # torch.optim.Optimizer | HuggingFaceOptimizer
    """
    Return PyTorch or Hugging Face optimizer
    """
    _parameters = [p for p in model.parameters() if p.requires_grad]
    kwargs.update({"lr": learning_rate})  # reconcile Tinker convention with PyTorch
    if name == "sgd":
        return torch.optim.SGD(_parameters, **kwargs)
    elif name == "adam":
        return torch.optim.Adam(_parameters, **kwargs)
    elif name in ["adamw", "adamw_torch"]:
        return torch.optim.AdamW(_parameters, **kwargs)
    elif name == "adamw_torch_fused":
        return torch.optim.AdamW(_parameters, **kwargs, fused=True)
    elif name == "adafactor":
        from transformers.optimization import Adafactor

        kwargs["relative_step"] = False  # for now
        return Adafactor(_parameters, **kwargs)
    else:
        raise NotImplementedError(f"Sorry, {name} optimizer not implemented.")


def get_scheduler(
    optimizer: Any,
    name: str = "none",
    **kwargs: Any,
) -> Any:  # torch.optim.lr_scheduler.LRScheduler | HuggingFaceScheduler
    """
    Return PyTorch or Hugging Face scheduler
    """
    if name in ["plateau", "reduce_lr_on_plateau"]:
        from torch.optim.lr_scheduler import ReduceLROnPlateau

        return ReduceLROnPlateau(optimizer=optimizer, **kwargs)

    elif name == "cosine_warmup":
        from transformers.optimization import get_cosine_schedule_with_warmup

        num_warmup_steps = kwargs.pop("num_warmup_steps", None)
        num_training_steps = kwargs.pop("num_training_steps", None)
        assert num_warmup_steps is not None and num_training_steps is not None, (
            "num_warmup_steps and num_training_steps must be provided"
        )
        return get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            **kwargs,
        )

    elif name in ["linear_warmup", "linear"]:
        from transformers.optimization import get_linear_schedule_with_warmup

        num_warmup_steps = kwargs.pop("num_warmup_steps", None)
        num_training_steps = kwargs.pop("num_training_steps", None)
        assert num_warmup_steps is not None and num_training_steps is not None, (
            "num_warmup_steps and num_training_steps must be provided"
        )
        return get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            **kwargs,
        )

    else:
        return None
