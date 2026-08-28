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

    if name == "tau2bench":
        if is_async:
            from .tau2bench import AsyncTau2BenchEnv

            return AsyncTau2BenchEnv(**kwargs)
        from .tau2bench import Tau2BenchEnv

        return Tau2BenchEnv(**kwargs)

    if name == "snorkel_insurance":
        # Tool-calling commercial underwriting over the snorkel benchmark tables,
        # LLM-judge graded. Ported alongside the finance env so the insurance domain has a
        # gym for Stage-3 rollout eval on its 41 held-out questions.
        if is_async:
            from .snorkel_insurance import AsyncSnorkelInsuranceEnv

            return AsyncSnorkelInsuranceEnv(**kwargs)
        from .snorkel_insurance import SnorkelInsuranceEnv

        return SnorkelInsuranceEnv(**kwargs)

    if name == "snorkel_finance":
        # Tool-calling financial QA over 10-K filings, LLM-judge graded. Ported from the
        # recovered mz-airline branch so the finance domain has a gym for rollout eval.
        if is_async:
            from .snorkel_finance import AsyncSnorkelFinanceEnv

            return AsyncSnorkelFinanceEnv(**kwargs)
        else:
            from .snorkel_finance import SnorkelFinanceEnv

            return SnorkelFinanceEnv(**kwargs)

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
