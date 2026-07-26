"""
Environments.

Lean Act-PRM fork: the only environment is ``act_prm_traces`` — a source of
logged, action-only demonstration trajectories (no interactive stepping). The
Act-PRM generator reads these traces and infers the latent thoughts behind each
logged action.
"""

from typing import Any

from .base import Environment
from .types import EnvironmentState, EnvironmentStateWithAnswer, EnvironmentStepResult


def get_env(name: str, is_async: bool = True, **kwargs: Any) -> Environment:
    """
    Get environment by name.
    """
    if name in ["act_prm_traces", "act_prm"]:
        from .act_prm_traces.env import ActPrmTracesEnv

        return ActPrmTracesEnv(**kwargs)

    raise NotImplementedError(f"Sorry, invalid environment: '{name}'.")


def load_env(name: str, **kwargs: Any) -> Environment:
    """Alias for :func:`get_env`."""
    return get_env(name, **kwargs)


__all__ = [
    "get_env",
    "load_env",
    "Environment",
    "EnvironmentState",
    "EnvironmentStateWithAnswer",
    "EnvironmentStepResult",
]
