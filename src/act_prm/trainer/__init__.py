"""
PyTorch trainers for Hugging Face Transformer (PEFT / LoRA) models.

Mirrors the pre-refactor ``strl.pytorch`` entrypoints (``get_trainer`` /
``get_optimizer`` / ``get_scheduler``). Imports are lazy so that merely
importing ``strl.trainer`` does not pull in torch / peft until a trainer or
optimizer is actually constructed (the GPU-side path, driven by
``main_pytorch.py``).
"""

from typing import Any


def get_trainer(name: str, **kwargs: Any) -> Any:
    """
    Get a (synchronous, LoRA) PyTorch trainer by name.
    """
    if name == "rl":
        from .trainers.rl import RLTrainer

        return RLTrainer(**kwargs)

    if name == "rl_batch":
        from .trainers.rl_batch import RLBatchTrainer

        return RLBatchTrainer(**kwargs)

    if name == "sft":
        from .trainers.sft import SFTTrainer

        return SFTTrainer(**kwargs)

    if name == "sft_flat":
        # Corpus-wide step-level SFT (see trainers/sft_flat.py). Kept as a SEPARATE
        # trainer so the 12 Stage-2 arms produced on the "sft" path stay reproducible.
        from .trainers.sft_flat import SFTFlatTrainer

        return SFTFlatTrainer(**kwargs)

    raise NotImplementedError(f"Trainer {name} not implemented")


def get_optimizer(*args: Any, **kwargs: Any) -> Any:
    """Return a PyTorch / Hugging Face optimizer (see ``optim.get_optimizer``)."""
    from .optim import get_optimizer as _get_optimizer

    return _get_optimizer(*args, **kwargs)


def get_scheduler(*args: Any, **kwargs: Any) -> Any:
    """Return a PyTorch / Hugging Face scheduler (see ``optim.get_scheduler``)."""
    from .optim import get_scheduler as _get_scheduler

    return _get_scheduler(*args, **kwargs)


__all__ = [
    "get_trainer",
    "get_optimizer",
    "get_scheduler",
]
