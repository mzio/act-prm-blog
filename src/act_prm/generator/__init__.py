"""
Generators (rollout harnesses) for LLM-based training.

Lean Act-PRM fork of ``strl.generator``: the registry keeps only the local
HuggingFace paths — the base env-rollout generator (``hf``) and the Act-PRM EM
harness (``act_prm``).
"""

from collections.abc import Callable
from functools import partial
from typing import Any


def get_generator_constructor(
    name: str,
    **kwargs: Any,
) -> Callable[..., Any]:
    """
    Get a (partially initialized) generator (harness) constructor by name.

    Usage:

    ```python
    generator_ctor = get_generator_constructor(name, **generator_cfg)
    generator = generator_ctor(llm=..., env=..., cfg=..., ...)
    ```
    """
    if name in ["act_prm", "act_prm_lenpen"]:
        from .act_prm.base import ActPrmGenerator

        return partial(ActPrmGenerator, **kwargs)

    if name in ["huggingface", "hf"]:
        from .huggingface.base import HuggingFaceGenerator

        return partial(HuggingFaceGenerator, **kwargs)

    raise NotImplementedError(f"Sorry, generator {name!r} is not implemented in the act-prm fork.")


__all__ = [
    "get_generator_constructor",
]
