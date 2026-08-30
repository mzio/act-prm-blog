"""
Parent environment class
References: https://github.com/bentrevett/pytorch-rl/blob/master/5%20-%20Proximal%20Policy%20Optimization%20(PPO)%20%5BCartPole%5D.ipynb
"""

import logging
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from transformers import AutoTokenizer

from .types import EnvironmentState, EnvironmentStepResult

logger = logging.getLogger(__name__)

DEFAULT_TRUNCATION_TEMPLATE = "Sorry, task not solved in max turns = {max_turns} allowed. Please try again."


class Environment(ABC):
    """
    Parent class for environment
    """

    def __init__(
        self,
        max_turns: int | None = None,
        num_tries: int = 1,
        tool_role: str = "tool",
        truncation_message: str = DEFAULT_TRUNCATION_TEMPLATE,
        split: str = "train",
        seed: int = 0,
        data_seed: int | None = None,
        verbose: bool = False,
        pretrained_model_config: dict[str, Any] | None = None,
        hide_observations: bool = False,
        hidden_obs_content: str = "...",  # or "<output omitted for brevity>"
        first_obs_to_show: int = 1,  # e.g, to keep prompt
        last_obs_to_show: int = 0,  # >= 2 to keep more than the last observation
        negative_rewards: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.max_turns = max_turns
        self.num_tries = num_tries
        self.tool_role = tool_role
        self.truncation_message = truncation_message.format(max_turns=max_turns)
        self.split = split
        self.seed = seed
        self.data_seed = data_seed if data_seed is not None else seed
        self.verbose = verbose
        self.pretrained_model_config = pretrained_model_config
        self.tokenizer = self._init_tokenizer()

        # Optionally hide prior observations in current state
        self.hide_observations = hide_observations
        self.hidden_obs_content = hidden_obs_content
        self.first_obs_to_show = first_obs_to_show
        self.last_obs_to_show = last_obs_to_show

        # Set last reward to -1 if model fails
        self.negative_rewards = negative_rewards

        # Run identifiers
        self.run_url: str | None = None
        self.run_cmd: str | None = None

    def __len__(self) -> int:
        """
        Get the environment's number of sample tasks
        """
        return len(self.datasets[self.split])

    def adjust_sample_id(self, sample_id: int) -> int:
        """
        Adjust sample index to be in range of environment's number of tasks.
        -> Wrap around if out of bounds.
        """
        return sample_id % len(self.datasets[self.split])

    def _init_tokenizer(self) -> AutoTokenizer | None:
        if self.pretrained_model_config is not None:
            _pretrained_model_config = {k: v for k, v in self.pretrained_model_config.items()}
            _model_name_or_path = _pretrained_model_config["pretrained_model_name_or_path"]
            # Use the model's OWN tokenizer (a prior hack swapped Qwen3-8B for
            # Qwen2.5-3B-Instruct — wrong vocab + not cached offline). Harmless here
            # regardless since main_pytorch overwrites this with llm.tokenizer.
            _pretrained_model_config["pretrained_model_name_or_path"] = _model_name_or_path
            _chat_template_path = _pretrained_model_config.pop("chat_template_path", None)
            tokenizer = AutoTokenizer.from_pretrained(**_pretrained_model_config)
            # Override chat template if provided
            if _chat_template_path is not None:
                with open(_chat_template_path, "r", encoding="utf-8") as f:
                    tokenizer.chat_template = f.read()
                    print(f"-> Overriding chat template with {_chat_template_path}")
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            return tokenizer
        else:
            tokenizer = None
        return tokenizer

    @abstractmethod
    def reset(
        self,
        sample_id: int,
        generation_id: int,
        try_step: int = 0,
        batch_idx: int = 0,
    ) -> EnvironmentState:
        """
        Reset environment (starting new episode, or working on a new sample)
        """
        raise NotImplementedError

    @abstractmethod
    def step(self, **kwargs: Any) -> EnvironmentStepResult:
        """
        Perform one step of the environment
        """
        raise NotImplementedError

    def shuffle(self, split: str | None = None, seed: int | None = None) -> None:
        """
        Shuffle environment's samples (e.g., after going through all during training)
        """
        split = split or self.split
        # Advance the seed per call. Re-seeding with the CONSTANT self.seed made every
        # epoch's permutation IDENTICAL, so from epoch 2 on the model saw the same order
        # and the same batch composition forever -- most of the way to not shuffling at
        # all. (Found 2026-08-30 from a batch-25-35 bump that recurred at a fixed offset
        # in both insurance arms.) Still fully deterministic given self.seed.
        self._shuffle_count = getattr(self, "_shuffle_count", 0) + 1
        base_seed = self.seed if seed is None else seed
        np.random.seed((base_seed + 1_000_003 * self._shuffle_count) % (2**32 - 1))
        ds = self.datasets[split]
        indices = np.arange(len(ds))
        np.random.shuffle(indices)
        # HF Dataset: prefer .select() (returns a Dataset, not a numpy view).
        # Plain list (used by tau2bench/toolsandbox-style envs): rebuild with
        # a list comprehension since lists don't support fancy indexing.
        # Anything else (numpy array, pandas, etc.) keeps the original path.
        if hasattr(ds, "select") and callable(ds.select):
            self.datasets[split] = ds.select(indices.tolist())
        elif isinstance(ds, list):
            self.datasets[split] = [ds[i] for i in indices]
        else:
            self.datasets[split] = ds[indices]

    def maybe_hide_observations(
        self,
        messages: list[dict[str, str]],
        hide_observations: bool | None = None,
        hidden_obs_content: str | None = None,
        first_obs_to_show: int | None = None,  # e.g., to keep prompt
        last_obs_to_show: int | None = None,  # e.g., to keep last observation
    ) -> list[dict[str, str]]:
        """
        Hide observations from messages
        """
        num_messages = len(messages)
        hide_observations = hide_observations or self.hide_observations
        if num_messages == 0 or not hide_observations:
            return messages

        hidden_obs_content = hidden_obs_content or self.hidden_obs_content
        first_obs_to_show = first_obs_to_show or self.first_obs_to_show
        last_obs_to_show = last_obs_to_show or self.last_obs_to_show

        return [
            {"role": message["role"], "content": hidden_obs_content}
            if (
                message["role"] in ["user", "tool"]
                and (idx >= first_obs_to_show and idx < num_messages - last_obs_to_show)
            )
            else message
            for idx, message in enumerate(messages)
        ]


class BaseTool(ABC):
    """
    Parent tool class
    """

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs

    @abstractmethod
    def __call__(self, **kwargs: Any) -> Any:
        """
        Execute the tool
        """
        raise NotImplementedError

    @abstractmethod
    def get_tool_desc(self) -> dict[str, Any]:
        """
        Returm the tool description for function calling
        """
        raise NotImplementedError
