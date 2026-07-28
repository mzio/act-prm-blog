"""
Base class for generation / rollout sampling for Hugging Face Transformers models
"""

import concurrent.futures
import logging
import sys
from copy import copy, deepcopy
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig

# from rich import print as rich_print
from rich.console import Console
from tinker_cookbook.utils import ml_log
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase

from act_prm.environments import Environment, EnvironmentState, EnvironmentStepResult
from act_prm.llm_handlers import HuggingFaceLLM
from act_prm.llm_handlers.action_utils import get_actions
from act_prm.replay_buffer import ReplayBuffer
from act_prm.replay_buffer.types import (
    EpisodeStep,
    MeanCenteredTrajectoryGroup,
    Trajectory,
    TrajectoryGroup,
)
from act_prm.utils.display import RichTextStreamer, display_state_action_next_obs

from ..utils import (
    get_action_logprobs_and_state_action_tokens,
    get_batch_model_inputs,
    get_response_content,
    remove_prefix_messages,
)

logger = logging.getLogger(__name__)
console = Console()
ROYGBIV = ["#FF0000", "#FF7F00", "#FFFF00", "#00FF00", "#0000FF", "#4B0082", "#9400D3"]
DEBUG_COLS = ["batch_id", "split", "try_step", "generation_id", "sample_id"]


class HuggingFaceGenerator:
    """
    Compute rollouts using Hugging Face Transformers models
    """

    def __init__(
        self,
        llm: HuggingFaceLLM,
        replay_buffer: ReplayBuffer,
        hf_tokenizer: PreTrainedTokenizerBase,
        env: Environment,
        cfg: DictConfig,
        enable_thinking: bool | None = None,  # default to HF template, but often set to False
        discount_factor: float | None = None,
        mean_center: bool = False,
        ml_logger: ml_log.Logger | None = None,
        name_or_identifier: str | None = None,
        streamer: bool = False,
        verbose: bool = False,
        ground_truth_generation: bool = False,
        ground_truth_incorrects: bool = False,
        **kwargs: Any,  # swallow extra config keys (e.g. debug, last_replay_buffer_path)
    ) -> None:
        self.llm = llm
        self.replay_buffer = replay_buffer
        self.hf_tokenizer = hf_tokenizer
        self.env = env
        self.cfg = cfg

        self.enable_thinking = enable_thinking

        self.discount_factor = discount_factor or cfg.get("discount_factor", 0.9)
        self.mean_center = mean_center  # mean-center the advantages

        # Logging
        self.ml_logger = ml_logger
        self.name_or_identifier = name_or_identifier
        self.run_url, self.run_cmd = self._init_identifiers()
        # Display attributes (used by utils/display.py)
        self.last_generated_data_url: str | None = None
        self.last_replay_buffer_path: str | None = None

        # (Cost) and token tracking
        self.usage_metrics: dict[str, int | float] = self._init_usage_metrics()

        # Silly streaming
        self.streamer = (
            RichTextStreamer(
                self.hf_tokenizer,
                skip_prompt=True,
                skip_special_tokens=True,
            )
            if streamer
            else None
        )
        self.verbose = verbose

        # Generate ground-truth actions at each step (if available)
        self.ground_truth_generation = ground_truth_generation
        self.ground_truth_incorrects = ground_truth_incorrects


    def _init_identifiers(self) -> tuple[str | None, str | None]:
        """
        Initialize identifiers for the generator
        """
        run_url = self.ml_logger.get_logger_url() if self.ml_logger is not None else None
        run_cmd = self.cfg.get("run_cmd", " ".join(sys.argv))
        run_cmd = f"uv run {run_cmd}" if run_cmd else None
        return run_url, run_cmd

    def _init_usage_metrics(self) -> dict[str, int | float]:
        """
        Initialize usage metrics for the generator.
        Tracks per-batch and cumulative token counts.
        """
        return {
            "batch_input_tokens": 0,
            "batch_output_tokens": 0,
            "batch_generate_calls": 0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_generate_calls": 0,
        }

    def _reset_batch_usage_metrics(self) -> None:
        """Reset per-batch counters at the start of each group rollout."""
        self.usage_metrics["batch_input_tokens"] = 0
        self.usage_metrics["batch_output_tokens"] = 0
        self.usage_metrics["batch_generate_calls"] = 0

    def _update_usage_metrics(
        self,
        input_ids_shape: tuple[int, ...],
        output_ids_shape: tuple[int, ...],
        state_input_lens: list[int] | None = None,
    ) -> None:
        """
        Update usage metrics after a model.generate() call.

        Args:
            input_ids_shape: shape of the input_ids tensor (batch_size, seq_len)
            output_ids_shape: shape of the generate output (batch_size, seq_len + new_tokens)
            state_input_lens: actual (non-padded) input lengths per sample, if available
        """
        batch_size = input_ids_shape[0]
        padded_input_len = input_ids_shape[1]
        total_output_len = output_ids_shape[1]
        new_tokens = total_output_len - padded_input_len

        # Use actual input lengths if available (accounts for left-padding)
        if state_input_lens is not None:
            input_tokens = sum(state_input_lens)
        else:
            input_tokens = batch_size * padded_input_len

        output_tokens = batch_size * new_tokens

        self.usage_metrics["batch_input_tokens"] += input_tokens
        self.usage_metrics["batch_output_tokens"] += output_tokens
        self.usage_metrics["batch_generate_calls"] += 1
        self.usage_metrics["total_input_tokens"] += input_tokens
        self.usage_metrics["total_output_tokens"] += output_tokens
        self.usage_metrics["total_generate_calls"] += 1

    def get_usage_metrics(self) -> dict[str, int | float]:
        """Return a copy of current usage metrics."""
        return dict(self.usage_metrics)

    def _get_messages_from_state(
        self,
        state: EnvironmentState,
        split: str,
        timestep: int,
        try_step: int,
        replay_buffer: ReplayBuffer | None = None,
        default_context: list[dict[str, Any]] | None = None,
        **get_past_episode_steps_kwargs: Any,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """
        Get messages from the environment state, in the form of
        [{"role": <role>, "content": <content>}, ...]

        For multiple tries, as default behavior we build the context as:
        [system_prompt, default_context, prior_rollouts, *current_rollout]

        where *current_rollout is a list of the current:
        [prior_messages, last_model_response, new_messages]

        Returns:
        - all_messages: list of all messages in the context
        - new_messages: list of new messages (next_observations) from the environment
        """
        # Convert new messages into consistent {"role": <role>, "content": <content>} format
        # -> See `act_prm.environments` classes for environment responses
        new_messages = [
            {"role": msg["role"], "content": msg["output"]}  # (to suppport OpenAI Responses API)
            if msg.get("type", "") == "function_call_output"  #
            else msg
            for msg in state.new_messages
        ]

        # Initialize all_messages as prior observations + model's last response + environment new messages
        default_context = default_context or state.default_context or []
        prior_messages = state.prior_messages or []
        # -> Remove system prompt and default context if in prior messages to avoid duplicates
        prior_messages = remove_prefix_messages(prior_messages, default_context)
        all_messages = prior_messages + (state.model_response or []) + new_messages

        # Then get past try messages to prepend to context (default base behavior)
        past_try_messages = self._get_past_trajectory_messages(
            state=state,
            split=split,
            timestep=timestep,
            try_step=try_step,
            replay_buffer=replay_buffer,
            default_context=default_context,
            **get_past_episode_steps_kwargs,
        )

        # Return final messages list
        all_messages = [
            {"role": "system", "content": state.system_prompt},
            *deepcopy(default_context or []),
            *past_try_messages,
            *all_messages,
        ]
        return all_messages, new_messages

    def _get_past_trajectory_messages(
        self,
        state: EnvironmentState,
        split: str | None = None,
        timestep: int | None = None,
        try_step: int | None = None,
        replay_buffer: ReplayBuffer | None = None,
        default_context: list[dict[str, Any]] | None = None,
        **get_past_episode_steps_kwargs: Any,
    ) -> list[dict[str, Any]]:
        """
        Get past try messages to prepend to context (default base behavior)
        """
        past_trajectory_msgs: list[dict[str, str]] = []
        default_context = default_context or []

        # Sanity-check timestep and try_step
        self._check_timestep_and_try_step(timestep, try_step, state)
        timestep = timestep or state.timestep
        try_step = try_step or state.try_step
        # Get split from arguments or state
        split = split or getattr(state, "split", None)
        # Allow split override (state.split may not be set yet on first try)
        assert split is not None, "split must be provided"

        # Default (base) behavior: append past rollout steps
        if timestep == 0 and try_step > 0 and replay_buffer is not None:
            # Get past trajectory steps and messages from replay buffer
            # -> As we append prior rollouts to each subsequent try, we only need to get the last
            #    try's rollout to add *all* prior rollouts to context
            past_trajectory_steps = replay_buffer.get_past_episode_steps(
                split=split,
                batch_id=state.batch_id,  # specifically for standard multi-try, should match on current batch_id
                try_step=try_step - 1,
                generation_id=state.generation_id,
                sample_id=state.sample_id,
                **get_past_episode_steps_kwargs,
            )
            # TODO: add a class attribute and if-logic here where we specify only getting the latest try context
            # -> e.g., at try 2, we only see try 1. At try 3, we only see try 2. 
            #    try 2 may have try 1 context attached, so we should make sure to only get the try-2 specific context then.
            try:
                past_trajectory_msgs = replay_buffer.get_messages_from_steps(past_trajectory_steps)
            except Exception as e:
                self._debug_get_messages_from_steps(past_trajectory_steps, state, replay_buffer, error=e)
            # Hacky check, but remove system prompt and default context if in past_rollout_messages
            past_trajectory_msgs = remove_prefix_messages(past_trajectory_msgs, default_context)
        return past_trajectory_msgs

    def _get_trajectory_group(self, **kwargs: Any) -> TrajectoryGroup:
        """
        Return trajectory group class
        """
        if self.mean_center:
            # Returns trajectory group where we compute advantages by:
            # 1. Computing mean-centered final rewards: final_reward - mean(final_rewards)
            # 2. Optionally apply step-wise discounting to these values
            return MeanCenteredTrajectoryGroup(**kwargs)
        return TrajectoryGroup(**kwargs)

    def _do_group_rollout_impl(
        self,
        sample_id: int,
        batch_id: int,
        llm: HuggingFaceLLM | None = None,
        replay_buffer: ReplayBuffer | None = None,
        hf_tokenizer: PreTrainedTokenizerBase | None = None,
        env: Environment | None = None,
        cfg: DictConfig | None = None,
        split: str = "train",
        try_step: int = 0,
        discount_factor: float | None = None,
        # start_idx: int = 0,
        num_return_sequences: int | None = None,
        generation_ids: list[int] | None = None,
        max_tokens: int | None = None,
        max_input_id_len: int | None = None,
        temperature: float | None = None,
        pbar_position: int = 0,
        # ) -> dict[str, list[TrajectoryGroup]]:
    ) -> list[TrajectoryGroup]:
        """
        Run rollouts for a single batch, e.g., by generating rollouts and grading them

        Returns:
        - all_trajectory_groups: List of trajectory groups for the batch
        """
        llm = llm or self.llm
        env = env or self.env
        cfg = cfg or self.cfg

        env.split = split  # Select task split
        discount_factor = discount_factor or self.discount_factor or cfg.discount_factor

        was_training = llm.model.training
        llm.model.eval()
        device = llm.model.device

        replay_buffer = replay_buffer or self.replay_buffer
        hf_tokenizer = hf_tokenizer or self.hf_tokenizer
        # Keep track of original padding side bc we'll change it multiple times below
        og_tokenizer_padding_side = copy(hf_tokenizer.padding_side)
        hf_tokenizer.padding_side = "left"  # confirm left-padding for generation
        # Also keep track of original pad token id bc we'll change it for logprobs
        # og_pad_token_id = copy(hf_tokenizer.pad_token_id)
        # pad_token_id = -og_pad_token_id  # try this for logprobs matching

        # Generation parameters
        max_tokens = max_tokens or cfg.max_tokens
        max_input_id_len = max_input_id_len or cfg.get("max_input_id_len", None)
        temperature = temperature or cfg.temperature
        num_return_sequences = num_return_sequences or (cfg.group_size if split == "train" else cfg.eval_group_size)

        with torch.no_grad():
            # Generate rollouts
            # -> We generate until the last generation is done, keeping track of:
            #    1. All generation indices (1, ..., num_return_sequences) (fixed) (all_gen_ids)
            #    2. Batch of generation indices that are not done (gets smaller) (gen_ids_todo)
            all_episode_steps: list[list[EpisodeStep]] = [[] for _ in range(num_return_sequences)]
            all_final_rewards: list[float] = [0.0 for _ in range(num_return_sequences)]
            # Track all messages for the current try
            all_current_messages: list[list[dict[str, str]]] = [[] for _ in range(num_return_sequences)]

            # Internal indices are always 0..n-1; generation_ids maps to actual IDs for env/replay
            _generation_ids = generation_ids if generation_ids is not None else list(range(num_return_sequences))
            gen_ids_todo = list(range(num_return_sequences))
            num_todo = len(gen_ids_todo)

            batch_states: list[EnvironmentState] = [
                env.reset(
                    sample_id=sample_id, generation_id=_generation_ids[gen_id], try_step=try_step, batch_id=batch_id
                )
                for gen_id in gen_ids_todo
            ]

            desc = f"Generating {num_todo} rollouts for sample {sample_id}, batch {batch_id}"
            pbar_task = tqdm(
                total=num_return_sequences,
                desc=desc,
                colour="cyan",
                leave=False,
                position=pbar_position,
            )
            rollout_pbars = [
                tqdm(
                    total=env.max_turns,
                    desc=f"Generating rollout {gen_id}",
                    colour=ROYGBIV[gen_id % len(ROYGBIV)],
                    leave=True,
                    position=pbar_position + gen_id + 1,
                )
                for gen_id in gen_ids_todo
            ]
            while len(gen_ids_todo) > 0:
                batch_state_messages, batch_new_messages = zip(
                    *[
                        self._get_messages_from_state(
                            state=state,
                            split=split,
                            timestep=state.timestep,
                            try_step=try_step,
                            replay_buffer=replay_buffer,
                            default_context=state.default_context,
                        )
                        for state in batch_states
                    ]
                )
                batch_state_messages = list(batch_state_messages)
                batch_new_messages = list(batch_new_messages)
                for _idx, gen_id in enumerate(gen_ids_todo):
                    all_current_messages[gen_id] = all_current_messages[gen_id] + batch_new_messages[_idx]

                _batch_model_inputs = get_batch_model_inputs(
                    input_messages=batch_state_messages,
                    tools=[state.tools for state in batch_states],
                    hf_tokenizer=hf_tokenizer,
                    padding_side="left",
                    enable_thinking=self.enable_thinking,
                )
                batch_state_inputs: dict[str, Any] = _batch_model_inputs[0]
                state_input_lens: list[int] = _batch_model_inputs[1]
                state_input_len_left_padded = batch_state_inputs["input_ids"].shape[1]

                # Truncate rollouts that exceed max_input_id_len
                if max_input_id_len is not None and max(state_input_lens) > max_input_id_len:
                    logger.warning(
                        "Truncating %d rollouts: max input len %d > %d",
                        sum(1 for s in state_input_lens if s > max_input_id_len),
                        max(state_input_lens),
                        max_input_id_len,
                    )
                    for _idx in gen_ids_todo:
                        all_final_rewards[_idx] = 0.0
                    break

                # Generate ground-truth actions at each step (if available)
                if self.ground_truth_generation and split == "train":
                    # Hardcoded hack for QA-style environments
                    decoded_texts = [
                        f"Final Answer: {state.answer}"
                        if isinstance(state.answer, str)
                        else f"Final Answer: {state.answer[0]}"
                        for state in batch_states
                    ]
                    # Add wrong answers if available and try_step < env.num_tries
                    if (
                        self.ground_truth_incorrects
                        and try_step + 1 < env.num_tries - 1  # 0-indexed; last 2 tries correct
                        and getattr(batch_states[0], "wrong_answers", None) is not None
                    ):
                        for _idx, _state in enumerate(batch_states):
                            # Replace with wrong answer
                            # _ans_idx = (_idx // 2) % len(_state.wrong_answers)
                            ans_idx = np.random.randint(len(_state.wrong_answers))
                            # ans_idx = try_step % len(_state.wrong_answers)
                            # ans_idx = try_step
                            decoded_texts[_idx] = f"Final Answer: {_state.wrong_answers[ans_idx]}"

                # hack for IFBench and IFEval: short-circuit llm.generate when
                # the env has planted a ground-truth response on try_step 0
                # (e.g. IFBench model_init_reply). Used for self-distillation.
                elif try_step == 0 and batch_states[0].ground_truth_response is not None:
                    decoded_texts = [state.ground_truth_response for state in batch_states]
                else:
                    # Generate model responses
                    # (batch_size, max_input_len) -> (batch_size, max_input_len + max_new_tokens)
                    outputs = llm.model.generate(
                        **batch_state_inputs.to(device),
                        max_new_tokens=max_tokens,
                        temperature=temperature,
                        num_return_sequences=1,
                        pad_token_id=hf_tokenizer.pad_token_id,
                        # output_scores=True,  # returns logprobs, but we'll recompute for training match
                        # output_logprobs=True,  # MZ: I can't tell if above supported though, so just use logprobs
                        # Silly streaming only supports batch_size == 1
                        streamer=self.streamer if len(state_input_lens) == 1 else None,
                    )
                    self._update_usage_metrics(
                        input_ids_shape=batch_state_inputs["input_ids"].shape,
                        output_ids_shape=outputs.shape,
                        state_input_lens=state_input_lens,
                    )
                    # Decode and convert tokens to messages
                    decoded_texts = hf_tokenizer.batch_decode(
                        outputs[:, state_input_len_left_padded:],
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=True,
                    )

                # For now we only allow one tool call per response
                decoded_texts = [
                    f"{text.split(llm.tool_call_eos)[0]}{llm.tool_call_eos}" if llm.tool_call_eos in text else text
                    for text in decoded_texts
                ]
                batch_model_messages: list[list[dict[str, str]]] = [
                    [{"role": "assistant", "content": text}] for text in decoded_texts
                ]
                batch_state_action_messages = [
                    batch_state_messages[_idx] + batch_model_messages[_idx] for _idx in range(len(batch_state_messages))
                ]
                # Get logprobs for action tokens
                # -> 1. Get token_ids for all state_action generations
                batch_state_action_inputs, _ = get_batch_model_inputs(
                    input_messages=batch_state_action_messages,
                    tools=[state.tools for state in batch_states],
                    hf_tokenizer=hf_tokenizer,
                    padding_side="right",
                    enable_thinking=self.enable_thinking,
                    add_generation_prompt=False,
                    continue_final_message=False,
                )
                # -> 2. Compute model inference logprobs (matches those at training time)
                try:
                    logits = llm.model(**batch_state_action_inputs.to(device), use_cache=False).logits
                    act_logps_and_state_act_toks = get_action_logprobs_and_state_action_tokens(
                        logits=logits,
                        state_lens=state_input_lens,
                        **batch_state_action_inputs.to(device),
                    )
                except torch.OutOfMemoryError:
                    logger.warning(
                        "OOM during logprob computation (seq_len=%d), falling back to per-sample",
                        batch_state_action_inputs["input_ids"].shape[1],
                    )
                    torch.cuda.empty_cache()
                    # Per-sample fallback to reduce peak memory
                    all_act_logps = []
                    all_state_act_toks = []
                    for _s_idx in range(batch_state_action_inputs["input_ids"].shape[0]):
                        _single_inputs = {k: v[_s_idx : _s_idx + 1] for k, v in batch_state_action_inputs.items()}
                        _logits = llm.model(**_single_inputs.to(device), use_cache=False).logits
                        _result = get_action_logprobs_and_state_action_tokens(
                            logits=_logits,
                            state_lens=[state_input_lens[_s_idx]],
                            **_single_inputs.to(device),
                        )
                        all_act_logps.extend(_result[0])
                        all_state_act_toks.extend(_result[1])
                        del _logits
                        torch.cuda.empty_cache()
                    act_logps_and_state_act_toks = (all_act_logps, all_state_act_toks)
                act_logprobs: list[list[float]] = act_logps_and_state_act_toks[0]  # batch_size x action_len
                state_action_tokens: list[list[int]] = act_logps_and_state_act_toks[1]
                # ^ "shape" is batch_size x (state_len + action_len)

                if len(state_input_lens) != len(act_logprobs):
                    logger.warning("Dropped last batch item for logprobs matching...")
                    logger.warning(f"logits: {logits.shape}")
                    logger.warning(f"state_input_lens: {state_input_lens}")
                    logger.warning(f"batch_state_action_inputs: {batch_state_action_inputs}")
                    raise RuntimeError(
                        f"state_input_lens ({len(state_input_lens)}) != act_logprobs ({len(act_logprobs)})"
                    )

                # Transition to next states
                # -> Parse to consistent ActionFromLLM format
                batch_parsed_actions = [get_actions(msgs) for msgs in batch_model_messages]

                def _run_env_step(_idx: int) -> EnvironmentStepResult:
                    # NOTE: threaded-safe because each rollout has its own `state`
                    # (a separate env.reset per gen_id) which is passed in, and
                    # env.step is a user-sim network call + tau2 tool exec on CPU
                    # (no GPU/CUDA touch). This mirrors AsyncTau2BenchEnv.step_async,
                    # which is literally asyncio.to_thread(super().step).
                    return env.step(
                        parsed_actions=batch_parsed_actions[_idx],
                        model_response=batch_model_messages[_idx],
                        current_state=batch_states[_idx],
                        current_messages=batch_state_messages[_idx],
                    )

                if len(batch_states) > 1:
                    # Parallelize the independent, GPU-idle user-sim env.step calls
                    # across the group rollouts (generation is already batched).
                    batch_env_step_results: list[EnvironmentStepResult] = [None] * len(batch_states)  # type: ignore[list-item]
                    with concurrent.futures.ThreadPoolExecutor(
                        max_workers=min(len(batch_states), 16)
                    ) as _executor:
                        _future_to_idx = {
                            _executor.submit(_run_env_step, _idx): _idx for _idx in range(len(batch_states))
                        }
                        for _future in concurrent.futures.as_completed(_future_to_idx):
                            _idx = _future_to_idx[_future]
                            try:
                                batch_env_step_results[_idx] = _future.result()
                            except Exception as _exc:  # noqa: BLE001
                                logger.warning(
                                    "Concurrent env.step failed for idx %d (%s); retrying inline",
                                    _idx,
                                    _exc,
                                )
                                # Fall back to an inline (serial) retry for this one rollout.
                                batch_env_step_results[_idx] = _run_env_step(_idx)
                else:
                    batch_env_step_results = [_run_env_step(0)]
                batch_next_states = [_result.state for _result in batch_env_step_results]
                batch_rewards = [_result.reward for _result in batch_env_step_results]
                batch_next_obs = [
                    [{"role": msg["role"], "content": get_response_content(msg)} for msg in next_state.new_messages]
                    for next_state in batch_next_states
                ]
                if self.verbose:
                    max_to_display = 1  # hardcoded hack
                    for _idx in range(len(batch_next_obs))[:max_to_display]:
                        _gen_id = gen_ids_todo[_idx]
                        _header_text = (
                            f"(Method: Default) "
                            f"{split.title()} Split, Try {try_step}, Batch {batch_id},"
                            f" Sample {sample_id}, Generation {_gen_id},"
                            f" Timestep {batch_states[_idx].timestep}"
                        )
                        display_state_action_next_obs(
                            generation_id=_gen_id,
                            state_messages=batch_state_messages[_idx],
                            action_messages=batch_model_messages[_idx],
                            next_obs_messages=batch_next_obs[_idx],
                            tools=batch_states[_idx].tools,
                            hf_tokenizer=hf_tokenizer,
                            cfg=cfg,
                            generator=self,
                            header_text=_header_text,
                            group_rewards=batch_rewards,
                            state=batch_states[_idx],
                        )

                # Add new model messages to all_current_messages
                for _idx, gen_id in enumerate(gen_ids_todo):
                    all_current_messages[gen_id] = all_current_messages[gen_id] + batch_model_messages[_idx]
                    if batch_env_step_results[_idx].done:  # add last observation if done
                        all_current_messages[gen_id].extend(batch_next_obs[_idx])

                # MZ 1/27/26 NOTE: this is a bit heinous...
                batch_episode_steps = [
                    EpisodeStep(
                        state=batch_state_messages[_idx],
                        action=batch_model_messages[_idx][0],  # dict[str, str]
                        next_obs=batch_next_obs[_idx],
                        tools=batch_states[_idx].tools,
                        state_action_tokens=state_action_tokens[_idx],
                        state_len=state_input_lens[_idx],
                        old_logprobs=act_logprobs[_idx],
                        temperature=temperature,
                        reward=batch_env_step_results[_idx].reward,
                        done=batch_env_step_results[_idx].done,
                        truncated=batch_env_step_results[_idx].truncated,
                        timestep=batch_states[_idx].timestep,
                        try_step=batch_states[_idx].try_step,
                        batch_id=batch_id,
                        sample_id=sample_id,
                        generation_id=_generation_ids[gen_id],
                        split=split,
                        is_train="train" in split,
                        # Other sample metadata
                        system_prompt=batch_states[_idx].system_prompt,
                        task_prompt=batch_states[_idx].task_prompt,
                        default_context=batch_states[_idx].default_context,
                        # Prior trajectory metadata
                        current_state=all_current_messages[gen_id],
                        final_outcome=[],  # Populated upon trajectory completion
                        prior_context=getattr(batch_states[_idx], "prior_context") or [],
                        prior_rewards=getattr(batch_states[_idx], "prior_rewards") or [],
                        prior_returns=getattr(batch_states[_idx], "prior_returns") or [],
                        prior_advantages=getattr(batch_states[_idx], "prior_advantages") or [],
                    )
                    for _idx, gen_id in enumerate(gen_ids_todo)
                ]
                # Build sequence of EpisodeSteps for each generation
                for _idx, gen_id in enumerate(gen_ids_todo):
                    all_episode_steps[gen_id].append(batch_episode_steps[_idx])
                    all_final_rewards[gen_id] = batch_rewards[_idx]  # Overwrite til last reward

                # Update pbars and finished rollouts
                # Collect done indices first, then remove in reverse to avoid
                # index shifting during forward iteration
                done_indices = []
                for _idx in range(len(gen_ids_todo)):
                    rollout_pbars[_idx].update(1)
                    if batch_env_step_results[_idx].done:
                        done_indices.append(_idx)
                for _idx in reversed(done_indices):
                    rollout_pbars[_idx].close()
                    rollout_pbars.pop(_idx)
                    gen_ids_todo.pop(_idx)
                # Move to next state
                batch_states = [
                    next_state
                    for _idx, next_state in enumerate(batch_next_states)
                    if not batch_env_step_results[_idx].done
                ]
                num_done = num_todo - len(gen_ids_todo)
                num_todo = len(gen_ids_todo)
                pbar_task.update(num_done)
                pbar_task.set_description(
                    f"Generating rollout {num_done} / {num_return_sequences - 1} ({num_todo} left)"
                )
            trajectories_in_group = [
                Trajectory(
                    episode_steps=all_episode_steps[gen_id],
                    final_reward=all_final_rewards[gen_id],
                    try_step=try_step,
                    discount_factor=discount_factor,
                )
                for gen_id in range(num_return_sequences)
            ]
            # for trajectory in trajectories_in_group:
            #     trajectory.compute_returns()
            all_trajectory_groups = [
                self._get_trajectory_group(
                    trajectories=trajectories_in_group,
                    final_rewards=all_final_rewards,
                    discount_factor=discount_factor,
                )
            ]
        if was_training:  # assume single try for now
            llm.model.train()

        hf_tokenizer.padding_side = og_tokenizer_padding_side
        # return {"policy": all_trajectory_groups}
        return all_trajectory_groups

    # ------------------------------------------------------------------
    # Rollout-dataset persistence (called from do_group_rollout)
    # ------------------------------------------------------------------
    @staticmethod
    def _sanitize_for_hub(name: str, max_length: int = 96) -> str:
        """Sanitize a HF Hub repo id of the form ``<org>/<repo>`` so it
        complies with Hub restrictions:

        - alphanumeric + '-' / '_' / '.' only
        - exactly one '/' (the org/repo separator); any '/' that appears in
          the repo segment (e.g. from ``env_config="hotpotqa/4k_ne"``)
          is collapsed to '-' so we don't produce invalid two-slash ids.
        - no '--' or '..' sequences, no leading/trailing '-' or '.'
        - <= ``max_length`` chars (the Hub limit is 96)

        Mirrors :func:`strl.pytorch.trainer.rl._sanitize_hf_repo_name` but
        kept here as a static method to avoid a circular import.
        """
        import re

        def clean(s: str) -> str:
            # No '/' in either segment; replace runs of any disallowed
            # chars (including '/') with a single '-'.
            s = re.sub(r"[^a-zA-Z0-9_\-.]+", "-", s)
            while "--" in s:
                s = s.replace("--", "-")
            while ".." in s:
                s = s.replace("..", ".")
            return s.strip("-.")

        if "/" in name:
            ns, repo = name.split("/", 1)
        else:
            ns, repo = None, name
        if ns:
            ns = clean(ns)
        repo = clean(repo)
        result = f"{ns}/{repo}" if ns else repo

        if len(result) > max_length:
            if ns is not None:
                avail = max_length - len(ns) - 1  # room for '/' + repo
                if avail < 1:
                    # Namespace alone exceeds limit -- keep namespace, drop repo.
                    return ns[:max_length].rstrip("-.")
                repo = repo[:avail].rstrip("-.")
                result = f"{ns}/{repo}"
            else:
                result = result[:max_length].rstrip("-.")
        return result

    def _build_default_rollouts_hub_repo_id(self) -> str:
        """Build a unique-but-short HF Hub repo id for rollout dataset uploads.

        Format aims for:
            ``<org>/strl-rollouts-E<env>-G<gen>-S<seed>-R<rep>-h<hash8>``
        ``hash8`` is an 8-char SHA256 of the full run signature so two runs
        with similar abbreviated fields don't collide. Stays under HF's 96
        char limit via tiered truncation: when the full form overflows we
        drop fields in order [replicate -> gen -> env -> seed], always
        keeping the hash so cross-run uniqueness survives.
        """
        import hashlib
        import os

        cfg = self.cfg or {}
        org = (
            cfg.get("hub_org")
            or cfg.get("hf_user")
            or os.environ.get("HF_USER")
            or "mzio"
        )
        prefix = cfg.get("rollouts_hub_prefix", "strl-rollouts")

        env = str(cfg.get("env_config", "env"))
        gen = str(cfg.get("generator_config", "gen")).replace("strl_", "")
        rep = str(cfg.get("replicate", ""))
        seed = str(cfg.get("seed", 0))

        sig_keys = (
            "env_config", "model_config", "lora_config", "generator_config",
            "trainer_config", "replicate", "seed", "learning_rate", "max_tokens",
            "num_tries", "batch_size", "group_size",
        )
        sig = "|".join(str(cfg.get(k, "")) for k in sig_keys)
        h = hashlib.sha256(sig.encode()).hexdigest()[:8]

        # Tiered candidates: most informative first, fall back to hash-only.
        tiers: list[list[str]] = [
            # Full info (env + gen + seed + rep + hash)
            [prefix, f"E{env}", f"G{gen}", f"S{seed}"]
            + ([f"R{rep}"] if rep else [])
            + [f"h{h}"],
            # Drop replicate
            [prefix, f"E{env}", f"G{gen}", f"S{seed}", f"h{h}"],
            # Drop generator
            [prefix, f"E{env}", f"S{seed}", f"h{h}"],
            # Drop env
            [prefix, f"G{gen}", f"S{seed}", f"h{h}"],
            # Drop seed
            [prefix, f"S{seed}", f"h{h}"],
            # Hash only
            [prefix, f"h{h}"],
        ]

        for parts in tiers:
            candidate = self._sanitize_for_hub(
                f"{org}/" + "-".join(p for p in parts if p),
                max_length=96,
            )
            # Hash must survive truncation; if it didn't, this tier is too long.
            if "/" in candidate and h in candidate.split("/", 1)[1]:
                return candidate

        # All tiers truncated the hash -- final fallback, hash + namespace only.
        return self._sanitize_for_hub(f"{org}/h{h}", max_length=96)

    def _maybe_save_rollouts(self) -> None:
        """Best-effort persistence of ``self.replay_buffer.hf_ds_buffer``.

        Called from ``do_group_rollout`` after every rollout completion.
        Disk / hub failures are logged at WARNING and swallowed -- the
        rollout itself never fails because of save errors.
        """
        if self.cfg is None:
            return

        save_local_path = self.cfg.get("save_rollouts_path")
        push_to_hub = self.cfg.get("save_rollouts_to_hub", True)
        hub_repo_id = self.cfg.get("save_rollouts_hub_repo")

        ds = getattr(self.replay_buffer, "hf_ds_buffer", None)
        if ds is None:
            # Buffer empty -- nothing to save yet.
            return

        # Local disk save (only when explicitly requested).
        if save_local_path:
            try:
                self.replay_buffer.save_hf_dataset_to_disk(save_local_path)
                self.last_replay_buffer_path = save_local_path
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "save_rollouts_path=%r local save failed (%s: %s); "
                    "rollout continues.",
                    save_local_path, type(e).__name__, e,
                )

        # Hub auto-push (default ON).
        if push_to_hub:
            if not hub_repo_id:
                try:
                    hub_repo_id = self._build_default_rollouts_hub_repo_id()
                except Exception as e:  # noqa: BLE001
                    logger.warning(
                        "Could not build default rollouts hub repo id "
                        "(%s: %s); skipping hub push for this rollout.",
                        type(e).__name__, e,
                    )
                    return
            try:
                ds.push_to_hub(hub_repo_id, private=True)
                self.last_generated_data_url = (
                    f"https://huggingface.co/datasets/{hub_repo_id}"
                )
                self.last_replay_buffer_path = hub_repo_id
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "Hub push of rollouts to %r failed (%s: %s); "
                    "rollout continues without persistence.",
                    hub_repo_id, type(e).__name__, e,
                )

    def do_group_rollout(self, **kwargs: Any) -> dict[str, list[TrajectoryGroup]]:
        """
        Generate a group of trajectories in the environment,
        and return a list of the trajectory group(s).

        If cfg.num_return_sequences is set and < group_size, generates in chunks
        to reduce peak GPU memory (e.g., group_size=8, num_return_sequences=4 -> 2 rounds).

        Add completed trajectories to replay buffer too
        """
        self._reset_batch_usage_metrics()

        # Check if we need to chunk
        cfg = kwargs.get("cfg", self.cfg)
        split = kwargs.get("split", "train")
        group_size = cfg.group_size if split == "train" else cfg.eval_group_size
        chunk_size = cfg.get("num_return_sequences", None)

        if chunk_size is not None and chunk_size < group_size:
            # Generate in chunks and merge.
            # Pre-build the full gen_id list, then slice per chunk so IDs are globally unique.
            all_gen_ids = list(range(group_size))
            all_trajectories = []
            for chunk_start in range(0, group_size, chunk_size):
                chunk_gen_ids = all_gen_ids[chunk_start : chunk_start + chunk_size]
                chunk_kwargs = dict(kwargs)
                chunk_kwargs["num_return_sequences"] = len(chunk_gen_ids)
                chunk_kwargs["generation_ids"] = chunk_gen_ids
                chunk_groups = self._do_group_rollout_impl(**chunk_kwargs)
                for tg in chunk_groups:
                    all_trajectories.extend(tg.trajectories)
                torch.cuda.empty_cache()
                logger.info(
                    "Chunk gen_ids=%s: %d trajectories",
                    chunk_gen_ids,
                    sum(len(tg.trajectories) for tg in chunk_groups),
                )
            # Reassemble into one trajectory group
            all_trajectory_groups = [
                self._get_trajectory_group(
                    trajectories=all_trajectories,
                    discount_factor=self.discount_factor,
                )
            ]
        else:
            all_trajectory_groups = self._do_group_rollout_impl(**kwargs)
        for trajectory_group in all_trajectory_groups:
            trajectory_group.compute_advantages()  # could also do RAAWR here
            self.replay_buffer.add_trajectory_group(trajectory_group)
        # Update replay_buffer DataFrame and HF dataset with new trajectories.
        self.replay_buffer.update_buffer_ds_and_df()

        # Note: rollouts buffer push is now driven by the caller at
        # the end of each try (see train.py:run_rollouts and
        # run_train_rollouts). Per-rollout pushes were too chatty;
        # we want one upload per try that captures every rollout in
        # that try's batch. Toggle / overrides:
        #   cfg.save_rollouts_to_hub   (bool, default True; per-try)
        #   cfg.save_rollouts_hub_repo (str, override the auto-built repo id)
        #   cfg.save_rollouts_path     (str, additional local disk save)
        return {"policy": all_trajectory_groups}

    # ---------------
    # Utility helpers
    # ---------------
    def _check_timestep_and_try_step(
        self,
        timestep: int | None,
        try_step: int | None,
        state: EnvironmentState,
    ) -> None:
        """
        Check that timestep and try_step are consistent with the state
        """
        if timestep is not None:
            assert timestep == state.timestep, f"timestep={timestep} != state.timestep={state.timestep}"
        if try_step is not None:
            assert try_step == state.try_step, f"try_step={try_step} != state.try_step={state.try_step}"

    def _debug_get_messages_from_steps(
        self,
        past_rollout_steps: list[EpisodeStep],
        state: EnvironmentState,
        replay_buffer: ReplayBuffer,
        error: Exception,
    ) -> None:
        """
        Debug get messages from steps
        """
        logger.error("Error on:\nreplay_buffer.get_messages_from_steps(past_rollout_steps)")
        logger.error("%s: %s", error.__class__.__name__, error)
        logger.error("past_rollout_steps: %s", past_rollout_steps)
        _df = replay_buffer.pd_df_buffer
        _df_debug = _df[(_df["generation_id"] == state.generation_id) & (_df["sample_id"] == state.sample_id)][
            DEBUG_COLS
        ]
        logger.error("debug_cols: %s", DEBUG_COLS)
        logger.error("_df_debug: %s", _df_debug)
        breakpoint()
        raise error
