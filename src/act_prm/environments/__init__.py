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

    if name == "snorkel_finance":
        # Tool-calling financial QA over 10-K filings; LLM-judge grading.
        # See environments/snorkel_finance/ and graders/snorkel_finance.py.
        if is_async:
            from .snorkel_finance import AsyncSnorkelFinanceEnv

            return AsyncSnorkelFinanceEnv(**kwargs)
        else:
            from .snorkel_finance import SnorkelFinanceEnv

            return SnorkelFinanceEnv(**kwargs)

    if name == "snorkel_insurance":
        # Tool-calling insurance-underwriting QA over a SQLite backend;
        # LLM-judge grading. See environments/snorkel_insurance/.
        if is_async:
            from .snorkel_insurance import AsyncSnorkelInsuranceEnv

            return AsyncSnorkelInsuranceEnv(**kwargs)
        else:
            from .snorkel_insurance import SnorkelInsuranceEnv

            return SnorkelInsuranceEnv(**kwargs)

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
