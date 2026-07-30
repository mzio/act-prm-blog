"""
LLM classes and types.

Lean fork: only the local HuggingFace Transformers policy is kept (the PyTorch
training path). The Claude / OpenAI / Meta / MetaGen / Tinker handlers from the
upstream ``strl`` project are intentionally dropped.
"""

from typing import Any

from .action_utils import get_actions
from .base import LLM
from .huggingface import HuggingFaceLLM
from .types import ActionFromLLM


def load_llm(
    name: str,
    model_config: dict[str, Any],
    is_async: bool = False,
    **kwargs: Any,
) -> LLM:
    """
    Load an LLM by name. Only ``hf_transformer`` (local HuggingFace policy) is
    supported in this lean Act-PRM fork.
    """
    if name == "hf_transformer":
        return HuggingFaceLLM(model_config=model_config, **kwargs)

    if name in ("claude_agent_sdk", "claude_query"):
        # LLM-judge backend for the graders (Stage-3 RL). LAZY import: the Claude
        # Agent SDK lives in the RL/grader venv (.venv-tau2 / a claude-sdk venv),
        # NOT the base training .venv — importing inside the branch keeps the base
        # venv working without the SDK installed. ``model_config`` carries the
        # ClaudeQueryLLM kwargs (model, max_turns, permission_mode, ...).
        from .claude_agent_sdk import ClaudeQueryLLM

        return ClaudeQueryLLM(**(model_config or {}), **kwargs)

    raise ValueError(
        f"Invalid model name: {name!r}. This Act-PRM fork supports 'hf_transformer' "
        f"and 'claude_agent_sdk' (LLM-judge grader; needs the SDK venv)."
    )


__all__ = [
    "get_actions",
    "load_llm",
    "ActionFromLLM",
    "LLM",
    "HuggingFaceLLM",
]
