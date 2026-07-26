"""
``ActPrmTracesEnv`` — an offline source of logged, action-only trajectories.

Unlike the interactive environments in the upstream project, this one is not
stepped token-by-token: the Act-PRM generator (:mod:`act_prm.generator.act_prm`)
pulls a whole logged trajectory via :meth:`get_trajectory` and runs its own EM
E-step (sample + score candidate thoughts) over the logged actions. ``reset`` /
``step`` are implemented to satisfy the :class:`Environment` ABC but are not part
of the Act-PRM rollout path.
"""

import logging
from pathlib import Path
from typing import Any

from ..base import Environment
from ..types import EnvironmentState, EnvironmentStepResult
from .data import (
    DATASET,
    load_synthetic,
    load_trajectories,
    load_trajectories_split,
)

logger = logging.getLogger(__name__)


class ActPrmTracesEnv(Environment):
    """Logged action-only demonstration trajectories for Act-PRM."""

    def __init__(
        self,
        dataset: str = DATASET,
        split_file: str | None = None,
        num_trajectories: int = 64,
        eval_trajectories: int = 8,
        max_traj_timestep: int = 8,
        max_steps_per_traj: int = 3,
        obs_max_chars: int = 2000,
        system_prompt_file: str | None = None,
        synthetic: bool = False,
        **kwargs: Any,
    ) -> None:
        # max_turns bounds the per-trajectory step count (used for progress bars).
        kwargs.setdefault("max_turns", max_steps_per_traj)
        super().__init__(**kwargs)

        self.dataset = dataset
        self.max_steps_per_traj = max_steps_per_traj
        self.obs_max_chars = obs_max_chars
        self.eval_splits = ["eval"]

        if synthetic or dataset in ("synthetic", "debug"):
            pool = load_synthetic()
            n_train = max(1, min(num_trajectories, len(pool)))
            train_pool = pool[:n_train]
            eval_pool = pool[n_train:] or pool[:1]
            logger.info("ActPrmTracesEnv: using %d synthetic trajectories", len(pool))
        elif split_file:
            logger.info("ActPrmTracesEnv: loading trajectories via split %s", split_file)
            train_pool, eval_pool = load_trajectories_split(split_file, dataset)
            if num_trajectories:
                train_pool = train_pool[:num_trajectories]
            if eval_trajectories:
                eval_pool = eval_pool[:eval_trajectories]
        else:
            total = num_trajectories + eval_trajectories
            logger.info(
                "ActPrmTracesEnv: streaming %d trajectories from %s (%d train / %d eval)",
                total,
                dataset,
                num_trajectories,
                eval_trajectories,
            )
            trajs = load_trajectories(total, max_traj_timestep, dataset)
            train_pool = trajs[:num_trajectories]
            eval_pool = trajs[num_trajectories:]

        # Optional system-prompt override (match the RL env's system prompt exactly).
        if system_prompt_file:
            sp = Path(system_prompt_file).read_text().strip()
            for t in train_pool + eval_pool:
                t["system_prompt"] = sp
            logger.info("ActPrmTracesEnv: system prompt overridden from %s (%d chars)", system_prompt_file, len(sp))

        self.datasets: dict[str, list[dict[str, Any]]] = {
            "train": train_pool,
            "eval": eval_pool,
        }
        logger.info(
            "ActPrmTracesEnv ready: %d train / %d eval trajectories",
            len(train_pool),
            len(eval_pool),
        )

    # ------------------------------------------------------------------
    # Act-PRM data access (used by the generator)
    # ------------------------------------------------------------------
    def get_trajectory(self, sample_id: int, split: str | None = None) -> dict[str, Any]:
        """Return the logged action-only trajectory at ``sample_id`` (wrapping)."""
        split = split or self.split
        ds = self.datasets[split]
        if not ds:
            raise RuntimeError(f"ActPrmTracesEnv split {split!r} is empty")
        return ds[sample_id % len(ds)]

    # ------------------------------------------------------------------
    # Environment ABC contract (not exercised by the Act-PRM E-step)
    # ------------------------------------------------------------------
    def reset(
        self,
        sample_id: int,
        generation_id: int,
        try_step: int = 0,
        batch_idx: int = 0,
    ) -> EnvironmentState:
        traj = self.get_trajectory(sample_id)
        return EnvironmentState(
            system_prompt=traj["system_prompt"],
            new_messages=[],
            model_response=None,
            prior_messages=[],
            tools=[],
            sample_id=sample_id,
            generation_id=generation_id,
            try_step=try_step,
            metadata={"trajectory": traj},
        )

    def step(self, **kwargs: Any) -> EnvironmentStepResult:
        """No-op terminal step. Act-PRM is offline EM; this exists only for the ABC."""
        current_state = kwargs.get("current_state")
        if current_state is None:
            current_state = EnvironmentState(
                system_prompt="",
                new_messages=[],
                model_response=None,
                prior_messages=[],
                tools=[],
            )
        return EnvironmentStepResult(state=current_state, reward=0.0, done=True, truncated=False)
