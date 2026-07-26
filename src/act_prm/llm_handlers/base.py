"""
Base LLM class
"""

from abc import ABC, abstractmethod
from typing import Any


class LLM(ABC):
    """
    Parent class for LLM classes
    """

    def __init__(
        self,
        model: Any | str = "",
        model_config: dict[str, Any] | None = None,
        generation_config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self.model = model
        self.model_name = model  # alias for model
        self.model_config = model_config or {}
        self.generation_config = generation_config or {}
        self.kwargs = kwargs

        self.cost = 0.0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    @abstractmethod
    def sample(
        self,
        system_prompt: str,
        messages: list[dict[str, Any]] | None = None,
        tools: list[dict[str, Any]] | None = None,
        max_new_tokens: int = 1024,
        num_return_sequences: int = 1,
        **generation_kwargs: Any,
    ) -> list[Any | None]:
        """
        Generate text from a prompt
        """
        raise NotImplementedError

    def _increment_token_count(
        self,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> None:
        """
        Increment token counts
        """
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens

    def _track_tokens(self, usage: Any | None = None) -> None | dict[str, int]:
        """
        Track token usage
        """
        if usage is None:
            return

        input_tokens = getattr(usage, "prompt_tokens", None)
        output_tokens = getattr(usage, "completion_tokens", None)

        if input_tokens is None or output_tokens is None:
            input_tokens = getattr(usage, "input_tokens", None)
            output_tokens = getattr(usage, "output_tokens", None)

        if input_tokens is None or output_tokens is None:
            return None
        self._increment_token_count(input_tokens, output_tokens)
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }
