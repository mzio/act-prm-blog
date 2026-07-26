"""
Environment objects and types
"""

from typing import Any

from datasets import Dataset
from datasets.iterable_dataset import IterableColumn, IterableDataset
from pydantic import BaseModel

# HuggingFace Datasets types
type HuggingFaceDatasetOrColumn = Dataset | IterableDataset | IterableColumn


class EnvironmentState(BaseModel):
    """
    State of the environment after a step
    """

    system_prompt: str
    new_messages: list[dict[str, Any]]
    model_response: dict[str, Any] | Any | None
    prior_messages: list[dict[str, Any]]
    tools: list[dict[str, Any]]
    # Other metadata / info
    sample_id: int = 0
    generation_id: int = 0
    batch_id: int = 0
    timestep: int = 0
    try_step: int = 0
    metadata: dict[str, Any] | None = None
    user_id: str = "user"  # helpful for personalization / filtering
    
    # Number of past observations to show
    first_obs_to_show: int | None = None
    last_obs_to_show: int | None = None

    # Other metadata helpful for ICL or retrieval
    task_prompt: str | None = None  # e.g., the content in the initial user message
    default_context: list[dict[str, Any]] | None = None  # e.g., list of few-shot examples

    prior_context: list[dict[str, Any]] | list[str] | None = None
    prior_rewards: list[float] | None = None
    prior_returns: list[float] | None = None
    prior_advantages: list[float] | None = None

    # Optional planted assistant response. When set on the current state,
    # the generator may skip llm.generate and use this string as the
    # model's reply for this step (used for self-distillation / planted
    # trajectories — e.g. IFBench's `model_init_reply`).
    ground_truth_response: str | None = None


class EnvironmentStateWithAnswer(EnvironmentState):
    """
    State of the environment after a step for QA tasks
    """

    question: str | None
    answer: str | None
    grading_rubric: list[dict[str, Any]] | None = None


class EnvironmentStepResult(BaseModel):
    """
    Result of a step in the environment
    - Note: we may not use, as most Gym conventions just return these as a tuple
    """

    state: EnvironmentState
    reward: float
    done: bool
    truncated: bool
    info: dict[str, Any] | None = None


# Alternative to above:
# We can use this type suggestion for environment.step() return value
type StepResult = tuple[
    EnvironmentState | EnvironmentStateWithAnswer,
    float,
    bool,
    bool,
    dict[str, Any],
]
