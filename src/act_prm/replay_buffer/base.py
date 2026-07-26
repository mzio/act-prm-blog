"""
Default replay buffer for storing training samples
"""

import asyncio
import logging
from copy import deepcopy
from typing import Any

import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset

from .types import EpisodeStep, Trajectory, TrajectoryGroup

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


# Only consider the following columns to identify duplicates
# -> We may remove keys based on this?
DEFAULT_DUPLICATE_KEYS = [
    "state_len",
    "done",
    "truncated",
    "split",
    "batch_id",  # "timestep", "try_step",
    "sample_id",
    "generation_id",
    "return_is_computed",
    "advantage_is_computed",
    "constant_reward_group",
    "is_train",
    "is_icl",
]


def convert_df_features_to_list(df: pd.DataFrame, iterable_keys: list[str]) -> pd.DataFrame:
    """
    Convert all iterable features in df to lists.
    Rows where the value is None/NaN are left as-is.
    """
    for k in iterable_keys:
        if k not in df.columns:
            continue
        df[k] = df[k].apply(lambda x: x.tolist() if hasattr(x, "tolist") else x)
    return df


class ReplayBuffer:
    """
    Replay buffer for storing state, action, next_obs, return, advantage
    """

    def __init__(
        self,
        max_size: int = 1e10,
        remove_duplicates: bool = False,
        duplicate_keys: list[str] | None = None,
        debug: bool = False,
    ) -> None:
        self.max_size = int(max_size)
        self.remove_duplicates = remove_duplicates
        self.duplicate_keys = duplicate_keys or DEFAULT_DUPLICATE_KEYS

        self.buffer: list[dict[str, Any]] = []
        self.embeddings: list[list[float]] = []  # will convert these to torch.Tensor later
        # The only things we need to filter on are:
        # - split, batch_id, try_step, data_sample_id ?
        self.hf_ds_buffer: Dataset | None = None  # Initialize after saving or loading from disk
        self.pd_df_buffer: pd.DataFrame | None = None  # Optionally use for quick filtering
        self.iterable_keys: list[str] = []  # Updated in `self.add_episode_step`
        self.debug = debug

    # Kept in-memory on the Trajectory (wandb logs them from there), but EXCLUDED from the
    # serialized buffer / uploaded HF dataset: per-step env metrics are a free-form dict
    # whose key set varies row to row, which breaks the Hub dataset viewer's per-shard cast.
    _BUFFER_EXCLUDE_KEYS = frozenset({"metrics"})

    def add_episode_step(self, episode_step: EpisodeStep) -> None:
        """
        Add an episode step to the replay buffer
        """
        dict_step = {
            k: v for k, v in episode_step.model_dump().items()
            if v is not None and k not in self._BUFFER_EXCLUDE_KEYS
        }
        self.buffer.append(dict_step)
        self.iterable_keys.extend(
            k for k in dict_step if isinstance(dict_step[k], list) and k not in self.iterable_keys
        )

    def add_trajectory(self, trajectory: Trajectory) -> None:
        """
        Add all episode steps in a trajectory to the replay buffer
        """
        for episode_step in trajectory.episode_steps:
            self.add_episode_step(episode_step)

    def add_trajectory_group(self, trajectory_group: TrajectoryGroup) -> None:
        """
        Add all episode steps, in all trajectories in a trajectory group, to the replay buffer
        """
        for trajectory in trajectory_group.trajectories:
            self.add_trajectory(trajectory)

    def build_retrievers(self) -> dict[str, Any] | None:
        """
        Build retrievers for the replay buffer -> only implemented in memory child classes
        """
        return None

    def update_buffer_ds_and_df(
        self,
        remove_duplicates: bool | None = None,
        duplicate_keys: list[str] | None = None,
    ) -> tuple[Dataset, pd.DataFrame] | tuple[None, None]:
        """
        Save the replay buffer to a DataFrame.

        Returns ``(None, None)`` (and leaves ``self.hf_ds_buffer`` /
        ``self.pd_df_buffer`` untouched) when the buffer is empty -- callers
        on the first rollout would otherwise crash on a stale assert.
        """
        if not self.buffer:
            return None, None
        self.hf_ds_buffer = Dataset.from_list(self.buffer)
        # Remove duplicates from dataset
        if remove_duplicates or self.remove_duplicates:
            self.pd_df_buffer = self.get_df_from_dataset(
                self.hf_ds_buffer,
                remove_duplicates,
                duplicate_keys,
            )
            self.hf_ds_buffer = Dataset.from_pandas(self.pd_df_buffer)
        else:
            self.pd_df_buffer = self.hf_ds_buffer.to_pandas()
        return self.hf_ds_buffer, self.pd_df_buffer

    def save_hf_dataset_to_disk(
        self,
        save_path: str,
        remove_duplicates: bool | None = None,
        duplicate_keys: list[str] | None = None,
    ) -> None:
        """
        Save all samples in ``self.buffer`` to a Hugging Face dataset on disk.

        No-op when the buffer is empty (rather than asserting), so callers
        invoked on the first rollout / before any data has been generated
        do not need to guard.
        """
        if not self.buffer:
            return
        self.hf_ds_buffer, _ = self.update_buffer_ds_and_df(remove_duplicates, duplicate_keys)
        self.hf_ds_buffer.save_to_disk(save_path)

    async def save_hf_dataset_to_disk_async(
        self,
        save_path: str,
        remove_duplicates: bool | None = None,
        duplicate_keys: list[str] | None = None,
    ) -> None:
        """
        Save all samples in self.buffer to a Hugging Face dataset asynchronously
        """
        assert len(self.buffer) > 0, "No samples to save"
        data = list(self.buffer)

        def _save():
            self.hf_ds_buffer = Dataset.from_list(data)
            if remove_duplicates or self.remove_duplicates:
                self.pd_df_buffer = self.get_df_from_dataset(
                    self.hf_ds_buffer,
                    remove_duplicates,
                    duplicate_keys,
                )
                self.hf_ds_buffer = Dataset.from_pandas(self.pd_df_buffer)
            else:
                self.pd_df_buffer = self.hf_ds_buffer.to_pandas()
            self.hf_ds_buffer.save_to_disk(save_path)

        await asyncio.to_thread(_save)

    def get_df_from_dataset(
        self,
        dataset: Dataset,
        remove_duplicates: bool,
        duplicate_keys: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Get a DataFrame from a Hugging Face dataset
        -> If remove_duplicates, remove duplicate rows by matching on `duplicate_keys`
        """
        duplicate_keys = duplicate_keys or self.duplicate_keys
        if remove_duplicates:
            return dataset.to_pandas().drop_duplicates(subset=duplicate_keys).reset_index(drop=True)
        return dataset.to_pandas()

    def load_from_hf_dataset(
        self,
        dataset: Dataset | None = None,  # datasets.arrow_dataset.Dataset
        load_path: str | None = None,
        split: str = "train",  # HF datasets will upload DatasetDict keyed by split
        verbose: bool = False,
        **load_dataset_kwargs: Any,
    ) -> None:
        """
        Load all samples from a Hugging Face dataset
        """
        # Try loading from HF dataset object if provided
        if dataset is not None:
            self.hf_ds_buffer = dataset
            logger.info("Loaded %s samples from HF dataset object!", len(self.hf_ds_buffer))
        
        else:
            assert load_path is not None, "Either dataset or load_path must be provided"
            # Try loading from Dataset Hub first
            try:
                self.hf_ds_buffer = load_dataset(load_path, split=split, **load_dataset_kwargs)
                logger.info("Loaded %s samples from Dataset Hub!", len(self.hf_ds_buffer))
            except Exception as e:  # noqa: BLE001 -- not a Hub repo, or its Arrow cast failed
                logger.info("Could not load load_path=%s as a Dataset Hub repo (%s: %s); "
                            "trying local load_from_disk instead", load_path, e.__class__.__name__, e)
                # Otherwise try loading from local disk
                self.hf_ds_buffer = Dataset.load_from_disk(load_path)
                logger.info("Loaded %s samples from local dataset!", len(self.hf_ds_buffer))

        # A split-keyed DatasetDict (passed in, or from a split-less load) -> pick the split
        if isinstance(self.hf_ds_buffer, DatasetDict):
            self.hf_ds_buffer = self.hf_ds_buffer[split]

        # Convert to pandas DataFrame for some operations
        self.pd_df_buffer = self.hf_ds_buffer.to_pandas()
        # Get iterable keys, convert df columns to lists, and initialize buffer
        self.iterable_keys.extend(
            k
            for k in self.hf_ds_buffer.features
            if isinstance(self.hf_ds_buffer[0][k], list) and k not in self.iterable_keys
        )
        self.buffer = self.hf_ds_buffer.to_list()

        if verbose:
            logger.info(f"Loaded {len(self.buffer)} samples from {load_path}")
            logger.info(f"Sample keys: {self.buffer[0].keys()}")

    def get_past_episode_steps(self, **kwargs: Any) -> list[EpisodeStep]:
        """
        Get episode steps from a prior rollout (override in child classes)
        """
        return self._get_trajectory_episode_steps(**kwargs)

    def _get_trajectory_episode_steps(
        self,
        **filter_kwargs: Any,
    ) -> list[EpisodeStep]:
        """
        Get episode steps for a prior rollout, given unique data sample id
        """
        df_rollout = self.get_df_for_trajectory(**filter_kwargs)
        step_dicts = convert_df_features_to_list(df_rollout, self.iterable_keys).to_dict(orient="records")
        return [EpisodeStep(**step_dict) for step_dict in step_dicts]

    def get_df_for_trajectory(
        self,
        split: str | None = None,
        sample_id: int | None = None,
        batch_id: int | None = None,
        try_step: int | None = None,
        timestep: int | None = None,
        generation_id: int | None = None,
        is_icl: bool = False,  # Calling this one out especially
        less_than_kwargs: dict[str, Any] | None = None,
        more_than_kwargs: dict[str, Any] | None = None,
        not_equal_kwargs: dict[str, Any] | None = None,
        leq_than_kwargs: dict[str, Any] | None = None,
        geq_than_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        Get prior trajectory steps as a DataFrame
        """
        # Could use kwargs, but better to make these arguments explicit
        filter_kwargs = {
            "split": split,
            "sample_id": sample_id,
            "batch_id": batch_id,
            "try_step": try_step,
            "generation_id": generation_id,
            "timestep": timestep,
            "is_icl": is_icl,  # if False, exclude explicit ICL samples from being retrieved
        }
        filter_kwargs.update(kwargs)
        for k in filter_kwargs:
            assert k in self.pd_df_buffer.columns, f"Invalid filter keyword argument: {k}"
        # Build selection mask for filtering
        mask = True
        for k, v in filter_kwargs.items():
            if v is not None:
                mask = mask & (self.pd_df_buffer[k] == v)
        # Optionally filter on <, >, or != certain values
        if less_than_kwargs is not None:
            for k, v in less_than_kwargs.items():
                mask = mask & (self.pd_df_buffer[k] < v)
        if more_than_kwargs is not None:
            for k, v in more_than_kwargs.items():
                mask = mask & (self.pd_df_buffer[k] > v)
        if not_equal_kwargs is not None:
            for k, v in not_equal_kwargs.items():
                mask = mask & (self.pd_df_buffer[k] != v)
        # Kinda heinous but ditto for <= and >=
        if leq_than_kwargs is not None:
            for k, v in leq_than_kwargs.items():
                mask = mask & (self.pd_df_buffer[k] <= v)
        if geq_than_kwargs is not None:
            for k, v in geq_than_kwargs.items():
                mask = mask & (self.pd_df_buffer[k] >= v)

        # # Exclude explicit ICL samples from being retrieved
        # if not is_icl:
        #     mask = mask & (~self.pd_df_buffer["is_icl"])

        return self.pd_df_buffer[mask].copy(deep=True).reset_index(drop=True)

    def get_messages_from_steps(
        self,
        steps: list[EpisodeStep],
        timestep: int = -1,  # by default, get the last-step messages,
        full_rollout: bool = True,  # i.e., the full trajectory
    ) -> list[dict[str, str]]:
        """
        Return the combined state, action, and next_obs chat for the last step of a trajectory.
        This should correspond to the "trajectory-to-go" from a given list of steps.
        """
        idx_dones = [i for i, step in enumerate(steps) if step.done]
        try:
            assert len(idx_dones) == 1, (
                f"Invalid steps for one trajectory! len({str(idx_dones)}) == {len(idx_dones)}, should be 1"
            )
        except Exception as e:
            logger.error(f"{e.__class__.__name__}: {e}")
            logger.error(f"idx_dones: {idx_dones}")
            raise e

        if len(steps) > 0:
            try:
                assert steps[-1].done, f"Last step should be done! Dones at {idx_dones}"
            except Exception as e:
                logger.error(f"{e.__class__.__name__}: {e}")
                logger.error(f"steps[-1].done: {steps[-1].done}")
                pass  # breakpoint removed
        else:
            logger.error("No steps to get messages from")
            logger.error(f"steps: {steps}")
            pass  # breakpoint removed

        # for step in steps:
        #     if step.done:
        #         return step.state + [step.action] + step.next_obs
        # raise ValueError("No done step found")
        assert (full_rollout is True and timestep == -1) or not full_rollout, (
            "If full_rollout is True, timestep must be -1"
        )  # Enforce rules for sanity-checking

        timesteps = [step.timestep for step in steps]
        _step_idx = timesteps.index(timestep) if timestep != -1 else -1
        step = steps[_step_idx]
        return step.state + [step.action] + step.next_obs

    def get_messages_from_trajectory(
        self,
        trajectory: Trajectory,
        timestep: int = -1,
        full_rollout: bool = True,
    ) -> list[dict[str, str]]:
        """
        Get the messages from a trajectory (semi-alias for get_messages_from_steps)
        """
        return self.get_messages_from_steps(trajectory.episode_steps, timestep, full_rollout)

    def get_action_and_future_trajectory_from_steps(
        self,
        steps: list[EpisodeStep],
        action_step_idx: int = 0,
    ) -> tuple[dict[str, str], list[dict[str, str]]]:
        """
        Get the action and future trajectory from a list of steps

        Returns:
        - action (dict[str, str]): Action in the form {"role": "assistant", "content": <action>}
        - trajectory_to_go (list[dict[str, str]]): The future trajectory to go, in the form
          [state_t, action_t, next_obs_t, action_t+1, next_obs_t+1, ..., action_T, next_obs_T]
        """
        #  First check that there's only one done step at end of steps
        idx_dones = [i for i, step in enumerate(steps) if step.done]
        assert len(idx_dones) == 1, (
            f"Invalid steps for one trajectory! len({str(idx_dones)}) == {len(idx_dones)}, should be 1"
        )
        assert steps[-1].done, f"Last step should be done! Dones at {idx_dones}"

        state: list[dict[str, str]] = steps[action_step_idx].state
        action: dict[str, str] = steps[action_step_idx].action
        # next_obs: list[dict[str, str]] = steps[timestep].next_obs
        trajectory_to_go = deepcopy(state)
        for _idx in range(action_step_idx, len(steps)):
            trajectory_to_go.append(steps[_idx].action)
            trajectory_to_go.extend(steps[_idx].next_obs)
        return action, trajectory_to_go

    def get_action_and_future_trajectory_from_trajectory(
        self,
        trajectory: Trajectory,
        action_step_idx: int = 0,
    ) -> tuple[dict[str, str], list[dict[str, str]]]:
        """
        Get the action and future trajectory from a trajectory
        """
        return self.get_action_and_future_trajectory_from_steps(
            trajectory.episode_steps,
            action_step_idx,
        )
