"""
Helper functions for Hugging Face PEFT implementation of LoRA
"""

from typing import Any

# Enable and disable adapters
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from peft.tuners.tuners_utils import BaseTunerLayer
from peft.utils import ModulesToSaveWrapper
from rich import print as rich_print
from transformers import PreTrainedModel
from transformers.utils.peft_utils import check_peft_version

# Minimum PEFT version supported for the integration
MIN_PEFT_VERSION = "0.5.0"

TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def get_lora_model(model: PreTrainedModel, **lora_kwargs: Any) -> PeftModel:
    """
    Get a PEFT model from a base model and LoRA configuration
    """
    model = get_peft_model(model, peft_config=get_lora_config(**lora_kwargs))
    display_trainable_parameter_count(model)
    return model


def get_lora_config(
    r: int = 32,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules=TARGET_MODULES,
) -> LoraConfig:
    """
    Get a default LoRA config object for PEFT models
    """
    return LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        target_modules=list(target_modules),
    )


def count_parameters(model: PreTrainedModel, trainable_only: bool = False) -> int:
    """
    Counts the total number of parameters in a PyTorch model.
    If trainable_only is True, only counts trainable parameters.
    """
    return sum(
        p.numel() for p in model.parameters() if ((p.requires_grad and trainable_only) or not trainable_only)
    )


def display_trainable_parameter_count(model: PreTrainedModel) -> None:
    """
    Displays the total number of trainable parameters in a PyTorch model.
    """
    trainable_count = count_parameters(model, trainable_only=True)
    total_count = count_parameters(model)
    trainable_ratio = trainable_count / total_count
    _text = "\n".join(
        [
            f"-> [bright_blue]Trainable parameters:[/bright_blue] {trainable_count}",
            f"-> [bright_red]Total parameters:[/bright_red]     {total_count}",
            f"-> [bright_magenta]Trainable percentage:[/bright_magenta] {trainable_ratio * 100:.2f}%",
        ]
    )
    rich_print(f"[bold]LoRA Parameter Counts[/bold]\n{_text}")
    return trainable_count, total_count, trainable_ratio


# Modified from https://github.com/huggingface/transformers/blob/878562b68d06536b475a61496e3c2a26fdb95af1/src/transformers/integrations/peft.py#L365
def disable_adapters_(
    model: PreTrainedModel,
    adapter_names: list[str] | None = None,
) -> None:
    """
    If you are not familiar with adapters and PEFT methods, we invite you to read more about them
    on the PEFT official documentation: https://huggingface.co/docs/peft

    In-place disables all adapters that are attached to the model.
    -> This leads to inferring with the base model only.
    """
    check_peft_version(min_version=MIN_PEFT_VERSION)

    for name, module in model.named_modules():
        if adapter_names is not None and name not in adapter_names:
            continue
        if isinstance(module, (BaseTunerLayer, ModulesToSaveWrapper)):
            # Recent versions of PEFT need to call `enable_adapters` instead
            if hasattr(module, "enable_adapters"):
                module.enable_adapters(enabled=False)
            else:
                setattr(module, "disable_adapters", True)


# Modified from https://github.com/huggingface/transformers/blob/878562b68d06536b475a61496e3c2a26fdb95af1/src/transformers/integrations/peft.py#L388
def enable_adapters_(
    model: PreTrainedModel,
    adapter_names: list[str] | None = None,
) -> None:
    """
    If you are not familiar with adapters and PEFT methods, we invite you to read more about them
    on the PEFT official documentation: https://huggingface.co/docs/peft

    In-place enables all adapters that are attached to the model.
    """
    check_peft_version(min_version=MIN_PEFT_VERSION)

    for name, module in model.named_modules():
        if adapter_names is not None and name not in adapter_names:
            continue
        if isinstance(module, BaseTunerLayer):
            # Recent versions of PEFT need to call `enable_adapters` instead
            if hasattr(module, "enable_adapters"):
                module.enable_adapters(enabled=True)
            else:
                setattr(module, "disable_adapters", False)
