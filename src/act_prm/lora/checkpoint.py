"""
LoRA checkpointing helper functions
"""

import logging
from collections import OrderedDict
from os.path import join
from typing import Any

import torch
from peft import PeftModel, set_peft_model_state_dict
from safetensors.torch import load_file

logger = logging.getLogger(__name__)


# -----------------
# Default / One GPU
# -----------------
def save_trainable_weights(model: Any) -> OrderedDict:
    """
    Save all trainable weights (i.e., LoRA weights) of a standard nn.Module model

    If saved to `state_dict`, should load later with:
    `model.load_state_dict(state_dict, strict=False)`
    """
    with torch.no_grad():
        state_dict = OrderedDict()
        for n, p in model.named_parameters():
            if p.requires_grad:
                state_dict[n] = p.cpu()
    return state_dict


def save_lora(model: Any, out_dir: str):
    """
    Save a model with LoRA weights
    """
    model.save_pretrained(out_dir)


def load_lora(model: Any, checkpoint_path: str, is_trainable: bool = True) -> Any:
    """
    Load LoRA weights from checkpoint to a model
    """
    if isinstance(model, PeftModel):
        _checkpoint_path = join(checkpoint_path, "adapter_model.safetensors")
        _adapter_name = model.active_adapter
        state = load_file(_checkpoint_path)  # state_dict-like mapping from safetensors
        set_peft_model_state_dict(model, state, adapter_name=_adapter_name)
    else:
        model = PeftModel.from_pretrained(model, checkpoint_path, is_trainable=is_trainable)
    if is_trainable:
        model.train()
    return model


def push_lora_to_hub(
    model: Any,
    repo_id: str,
    commit_message: str | None = None,
    private: bool = False,
) -> str | None:
    """
    Push LoRA adapter weights to HuggingFace Hub.

    Args:
        model: PeftModel with LoRA adapters
        repo_id: HuggingFace Hub repo ID (e.g., "mzio/aprm-lora-best-...")
        commit_message: Optional commit message
        private: Whether the repo should be private

    Returns:
        URL of the pushed model, or None if push failed
    """
    try:
        model.push_to_hub(
            repo_id,
            commit_message=commit_message or "Update LoRA checkpoint",
            private=private,
        )
        logger.info("Pushed LoRA checkpoint to: https://huggingface.co/%s", repo_id)
        return f"https://huggingface.co/{repo_id}"
    except Exception as e:
        logger.warning(
            "Failed to push LoRA to hub (%s: %s). Continuing with local save only.",
            type(e).__name__,
            e,
        )
        return None
