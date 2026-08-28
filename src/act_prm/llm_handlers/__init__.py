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

        # LLM-judge backend for the graders (the snorkel_finance gym scores

        # finqa_reasoning answers with it). LAZY import: the Claude Agent SDK

        # lives in the grader venv, not the base training .venv, so importing

        # inside the branch keeps the base venv working without it installed.

        from .claude_agent_sdk import ClaudeQueryLLM


        return ClaudeQueryLLM(**(model_config or {}), **kwargs)


    raise ValueError(
        f"Invalid model name: {name!r}. This Act-PRM fork only supports 'hf_transformer'."
    )


__all__ = [
    "get_actions",
    "load_llm",
    "ActionFromLLM",
    "LLM",
    "HuggingFaceLLM",
]
