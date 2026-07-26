"""
Helper functions for PyTorch-based generators
"""

import logging
from typing import Any, Callable

import torch
from transformers import PreTrainedTokenizerBase

from act_prm.replay_buffer.types import EpisodeStep, Trajectory, TrajectoryGroup

logger = logging.getLogger(__name__)


def get_response_content(msg: dict[str, Any]) -> str:
    """
    Get message content from an Environment response message
    """
    return msg["output"] if msg.get("output", None) else msg["content"]


def remove_prefix_messages(
    messages: list[dict[str, Any]],
    default_context: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """
    Remove system prompt and default context from messages
    """
    # First remove system prompt
    msgs = [msg for msg in messages if msg.get("role", "") != "system"]
    # Then remove default context:
    # -> Try removing a contiguous list of messages whenever it appears
    _default_context = default_context or []
    if len(_default_context) == 0:
        return msgs
    # -> Otherwise, remove default context messages kinda like Lomuto's
    l_idx = 0
    r_idx = 0
    w_len = len(_default_context)
    # for r_idx in range(len(msgs) - w_len + 1):
    while r_idx < len(msgs):
        if msgs[r_idx : r_idx + w_len] == _default_context:
            # Swap the default context entries
            swapped_msgs = msgs[r_idx : r_idx + w_len], msgs[l_idx : l_idx + w_len]
            msgs[l_idx : l_idx + w_len], msgs[r_idx : r_idx + w_len] = swapped_msgs
            l_idx += w_len
            r_idx += w_len
        else:
            r_idx += 1
    return msgs[l_idx:]


def get_action_logprobs_and_state_action_tokens(
    logits: torch.FloatTensor,
    input_ids: torch.LongTensor,
    attention_mask: torch.BoolTensor,  # should be, but may be LongTensor
    state_lens: list[int],
    **kwargs: Any,
) -> tuple[list[list[float]], list[list[int]]]:
    """
    Get action logprobs from logits and input_ids (i.e., labels or target token ids)

    Assumes:
    - logits are right-padded and shape (batch_size, seq_len, vocab_size)
    - input_ids is shape (batch_size, seq_len)
    - state_lens is list of length batch_size, each element is len(state_tokens)

    Returns:
    - logprobs, which is a list of length batch_size, each element is a list of length
      individual_action_len (the number of action tokens in the generation)
    """
    labels = input_ids  # convenience alias
    attention_mask = attention_mask.bool()
    # Tmask: 1111111111111111111111111111111111
    # state: we linear (state_len = 9)
    # total: we linearized the chungus among us
    # input: we linearized the chungus among u `total[:-1]`
    # label: e linearized the chungus among us `total[1:]`
    # start: --------ized the chungus among us `label[Tmask[1:]][state_len - 1:]`
    # state: every (state_len = 5)
    # Tmask: 1111111111111111111111000000000000
    # total: everything is chungus.xxxxxxxxxxxx
    # input: everything is chungus.xxxxxxxxxxx `total[:-1]`
    # label: verything is chungus.xxxxxxxxxxxx `total[1:]`
    # Lmask: 111111111111111111111000000000000 `Tmask[1:]`
    # label: verything is chungus.             `label[Tmask[1:]]`
    # start: ----thing is chungus.             `label[Tmask[1:]][state_len - 1:]`
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    _dtype = shift_logits.dtype
    # Use F.cross_entropy which fuses softmax + gather internally,
    # avoiding the (B, L, V) logsumexp materialization that causes OOM.
    # Input: (B, V, L-1), target: (B, L-1) -> output: (B, L-1)
    logprobs = -torch.nn.functional.cross_entropy(
        shift_logits.transpose(1, 2),  # (B, V, L-1)
        shift_labels,  # (B, L-1)
        reduction="none",
    ).to(dtype=_dtype)  # (B, L-1)
    del shift_logits

    # Get logprobs for action tokens only
    a_starts = [state_len - 1 for state_len in state_lens]
    try:
        logprobs = [
            logprobs[b_idx][attention_mask[b_idx, 1:]][start_idx:].tolist()  # mask matches targets
            for b_idx, start_idx in enumerate(a_starts)
        ]
    except Exception as e:
        print(f"Error with logprobs[b_idx][attention_mask[b_idx, 1:]][start_idx:].tolist(): {e}")
        print("len(state_lens)", len(state_lens))
        print("a_starts", a_starts)
        print("len(logprobs)", len(logprobs))
        raise
    # MZ 1/27/26: maybe more clear to keep separate?
    # logprobs = [logprobs[b_idx][attn_mask] for b_idx, attn_mask in enumerate(attention_mask)]
    # logprobs = [logprobs[b_idx][start_idx:] for b_idx, start_idx in enumerate(a_starts)]
    state_action_tokens = [
        # input_ids[b_idx][attention_mask[b_idx]][start_idx:].tolist()
        input_ids[b_idx][attention_mask[b_idx]].tolist()
        for b_idx, start_idx in enumerate(a_starts)
    ]  # inputs should be action_tokens[b_idx][:-1], targets action_tokens[b_idx][1:]
    return logprobs, state_action_tokens


def get_batch_model_inputs(
    input_messages: list[list[dict[str, str]]],
    tools: list[list[dict[str, str]]] | None,
    hf_tokenizer: PreTrainedTokenizerBase,
    padding_side: str = "left",
    enable_thinking: bool = False,
    add_generation_prompt: bool = True,
    continue_final_message: bool = False,
) -> tuple[dict[str, Any], list[int]]:
    """
    Get model_inputs (input_ids, attention_mask, etc.) from a batch of input messages

    Returns:
    - batch_model_inputs: Transformer model inputs (input_ids, attention_mask, etc.)
    - input_lens: list of length batch_size, each element is number of non-padded input tokens
    """
    hf_tokenizer.padding_side = padding_side
    if tools is not None:
        assert len(input_messages) == len(tools), (
            "Each input message must have corresponding tool descriptions (can be list[None] or None)"
        )

    # First, only format text to handle different available tools per state
    batch_model_texts = [
        hf_tokenizer.apply_chat_template(
            input_messages[b_idx],
            tools=tools[b_idx] if tools is not None else None,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=continue_final_message,
            enable_thinking=enable_thinking,
            padding=False,
            tokenize=False,
        )
        for b_idx in range(len(input_messages))
    ]

    # Then tokenize (get padded input_ids, attention_mask, etc.)
    try:
        batch_model_inputs = hf_tokenizer(batch_model_texts, padding=True, return_tensors="pt")
    except Exception as e:
        logger.error(
            f"Error tokenizing batch_model_texts: {e.__class__.__name__}: {e}\nbatch_model_texts: {batch_model_texts}"
        )
        raise

    input_lens = [attn_mask.sum().item() for attn_mask in batch_model_inputs["attention_mask"]]
    return batch_model_inputs, input_lens


def get_trajectory_group_from_generations(
    state_messages_in_group: list[list[dict[str, str]]],
    actions_in_group: list[dict[str, str]],
    state_len_in_group: list[int],
    state_action_tokens_in_group: list[list[int]],
    old_logprobs_in_group: list[list[float]],
    rewards_in_group: list[float],
    action_probs_in_group: list[float],
    generation_ids_in_group: list[int],
    try_step: int,
    discount_factor: float,  # self.discount_factor
    get_trajectory_group_method: Callable,
    **shared_kwargs: Any,
) -> TrajectoryGroup:
    """
    Save generations to a TrajectoryGroup
    """
    # For some cases, state will be the same for all generations
    if len(state_messages_in_group) == 1:
        state_messages_in_group = [state_messages_in_group[0]] * len(actions_in_group)
        state_len_in_group = [state_len_in_group[0]] * len(actions_in_group)

    episode_steps_in_group: list[EpisodeStep] = [
        EpisodeStep(
            state=state_messages_in_group[_idx],
            action=action,  # dict[str, str]
            state_action_tokens=state_action_tokens_in_group[_idx],
            state_len=state_len_in_group[_idx],
            old_logprobs=old_logprobs_in_group[_idx],
            reward=rewards_in_group[_idx],
            action_prob=action_probs_in_group[_idx],
            generation_id=generation_ids_in_group[_idx],
            try_step=try_step,
            **shared_kwargs,
        )
        for _idx, action in enumerate(actions_in_group)
    ]
    trajectories_in_group: list[Trajectory] = [
        Trajectory(
            episode_steps=[episode_step],
            try_step=try_step,
            discount_factor=discount_factor,
            final_reward=rewards_in_group[i],
        )
        for i, episode_step in enumerate(episode_steps_in_group)
    ]
    # self._get_trajectory_group
    return get_trajectory_group_method(
        trajectories=trajectories_in_group,
        final_rewards=rewards_in_group,
        discount_factor=discount_factor,
    )
