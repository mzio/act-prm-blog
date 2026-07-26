"""
Replay buffer for storing episode steps (at minimum, (state, action, advantage) tuples)
"""

from typing import Any

from transformers import PreTrainedTokenizerBase

from .base import ReplayBuffer
from .types import MeanCenteredTrajectoryGroup, Trajectory, TrajectoryGroup


def get_replay_buffer(
    name: str,
    hf_tokenizer: PreTrainedTokenizerBase,
    **kwargs: Any,
) -> ReplayBuffer:
    """
    Get a replay buffer by name
    """
    if name == "default":
        return ReplayBuffer(**kwargs)

    elif name in ["memory", "memory_bm25"]:
        from .bm25 import MemoryReplayBufferBM25

        return MemoryReplayBufferBM25(hf_tokenizer=hf_tokenizer, **kwargs)

    elif name in ["memory_sbert"]:
        from .memory_sbert import MemoryReplayBufferSBERT

        return MemoryReplayBufferSBERT(hf_tokenizer=hf_tokenizer, **kwargs)

    elif name in ["memory_llm"]:
        # Call out as implementation TODO
        raise NotImplementedError(f"Sorry, replay buffer '{name}' is not implemented yet.")

    else:
        raise NotImplementedError(f"Sorry, replay buffer '{name}' is not implemented yet.")


__all__ = [
    "ReplayBuffer",
    "Trajectory",
    "TrajectoryGroup",
    "MeanCenteredTrajectoryGroup",
]
