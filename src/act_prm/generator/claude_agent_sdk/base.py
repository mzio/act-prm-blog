"""
Base class for Claude Agent SDK Generators
"""

import asyncio
import logging
import sys
from copy import deepcopy
from typing import Any

from omegaconf import DictConfig
from pandas import DataFrame
from pandas import concat as pd_concat
from tinker_cookbook.utils import ml_log
from transformers import PreTrainedTokenizerBase

from act_prm.environments import Environment, EnvironmentState
from act_prm.llm_handlers.pricing import compute_cost_usd
from act_prm.replay_buffer import ReplayBuffer
from act_prm.replay_buffer.types import (
    MeanCenteredTrajectoryGroup,
    Trajectory,
    TrajectoryGroup,
)
from act_prm.utils.display import RichTextStreamer

from ..utils import remove_prefix_messages
from .types import ClaudeAgentResponse

logger = logging.getLogger(__name__)


class BaseClaudeAgentSDKGenerator:
    """
    Base class for generating rollouts using Claude Agent SDK
    """

    def __init__(
        self,
        hf_tokenizer: PreTrainedTokenizerBase,
        env: Environment,
        cfg: DictConfig,
        replay_buffer: ReplayBuffer,
        # Claude Agent SDK parameters
        model: str = "claude-sonnet-4-6",
        permission_mode: str = "default",
        effort: str = "low",
        max_agent_turns: int | None = None,
        # Retrieval and multi-try parameters
        use_last_try_only: bool = False,
        same_sample_tries: bool = False,
        use_same_user_ids: bool = False,  # only retrieve memory from the SAME user_id/persona
        # Optional parameters
        method_name: str = "Default",
        max_tokens: int | None = None,
        temperature: float | None = None,
        timeout: float = 120,  # how long to wait for a response from the Claude Agent SDK
        discount_factor: float | None = None,
        mean_center: bool = False,
        ml_logger: ml_log.Logger | None = None,
        name_or_identifier: str | None = None,
        streamer: bool = False,
        verbose: bool = False,
        last_generated_data_url: str | None = None,
        last_replay_buffer_path: str | None = None,
        debug: bool = False,
    ) -> None:
        self.hf_tokenizer = hf_tokenizer
        self.env = env
        self.cfg = cfg
        self.replay_buffer = replay_buffer
        self.method_name = method_name  # Method name for display purposes
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout

        # Advantage calculation
        self.discount_factor = discount_factor or cfg.get("discount_factor", 0.9)
        self.mean_center = mean_center  # mean-center the advantages

        # Retrieval and multi-try
        self.use_last_try_only = use_last_try_only
        self.same_sample_tries = same_sample_tries
        self.use_same_user_ids = use_same_user_ids

        self.ml_logger = ml_logger
        self.name_or_identifier = name_or_identifier
        self.run_url, self.run_cmd = self._init_identifiers()

        # Other identifiers that can be populated
        self.last_generated_data_url = last_generated_data_url
        self.last_replay_buffer_path = last_replay_buffer_path

        # Silly streaming (disabled for Tinker)
        self.streamer = (
            RichTextStreamer(self.hf_tokenizer, skip_prompt=True, skip_special_tokens=True)
            if streamer else None
        )
        self.verbose = verbose
        self.debug = debug

        # Initialize Claude Agent SDK options and token tracking
        self._init_claude_options(
            model=model,
            permission_mode=permission_mode,
            effort=effort,
            max_agent_turns=max_agent_turns,
        )
        self._init_usage_metrics()

    def _init_claude_options(self, **kwargs: Any) -> None:
        """
        Initialize Claude Agent SDK options

        These kwargs are forwarded verbatim to `ClaudeAgentOptions(**kwargs)`
        (see `query.py:sample_query_response`), so every key must be a real
        ClaudeAgentOptions field. Our ctor exposes `max_agent_turns` (to avoid
        clashing with the env's `max_turns`); translate it to the SDK's
        `max_turns` here.
        """
        claude_agent_option_kwargs: dict[str, Any] = {
            "allowed_tools": [],
            "tools": [],
        }
        claude_agent_option_kwargs.update(kwargs)
        # Single turn by default: we parse the <tool_call> out of ONE assistant
        # response and execute it in the env ourselves, so the CLI must not run
        # its own multi-turn tool loop. Left unset, the CLI default (>1) lets the
        # model burn turns hallucinating fake <tool_result> blocks we discard.
        # A yaml `max_agent_turns` still overrides this.
        max_agent_turns = claude_agent_option_kwargs.pop("max_agent_turns", None)
        claude_agent_option_kwargs["max_turns"] = max_agent_turns if max_agent_turns is not None else 1
        self.claude_agent_option_kwargs = claude_agent_option_kwargs

    def _init_usage_metrics(self) -> None:
        """
        Initialize usage metrics tracking

        Cumulative cost + token tracking. Token counts come from the SDK
        response.usage; dollar cost is preferred from response.cost (when
        the SDK reports it) and falls back to tokens × known rate.
        """
        self._cumulative_cost_usd: float = 0.0
        self._batch_cost_usd: float = 0.0
        self._prompt_tokens: int = 0
        self._completion_tokens: int = 0
        # Last cumulative cost reported to get_usage_metrics — used to compute
        # the batch delta when cost is derived from token counts.
        self._last_reported_cumulative_cost: float = 0.0

    def _init_identifiers(self) -> tuple[str | None, str | None]:
        """
        Initialize identifiers for the generator
        """
        run_url = self.ml_logger.get_logger_url() if self.ml_logger is not None else None
        run_cmd = self.cfg.get("run_cmd", " ".join(sys.argv))
        run_cmd = f"uv run {run_cmd}" if run_cmd else None
        return run_url, run_cmd

    def _get_trajectory_group(self, **kwargs: Any) -> TrajectoryGroup:
        """
        Return trajectory group class
        - Override in subclasses, e.g., to return MeanCenteredTrajectoryGroup
        """
        if self.mean_center:
            # Returns trajectory group where we compute advantages by:
            # 1. Computing mean-centered final rewards: final_reward - mean(final_rewards)
            # 2. Optionally apply step-wise discounting to these values
            return MeanCenteredTrajectoryGroup(**kwargs)
        return TrajectoryGroup(**kwargs)

    def _track_usage_metrics(self, response: ClaudeAgentResponse | None = None) -> None:
        """
        Track usage metrics from a ClaudeAgentResponse
        - Accumulates tokens (always) and dollar cost (when SDK provides it)
        """
        # Initialize tracking if no response is provided
        if response is None:
            return

        usage = response.usage or {}
        self._prompt_tokens += int(usage.get("input_tokens", 0) or 0)
        self._completion_tokens += int(usage.get("output_tokens", 0) or 0)
        cost = getattr(response, "cost", None)
        if cost and cost > 0:
            self._cumulative_cost_usd += cost
            self._batch_cost_usd += cost

    def get_usage_metrics(self) -> dict[str, float]:
        """
        Return cost + token metrics, using the cost path appropriate to the harness:
        - Claude Code (claude_agent_sdk): the bundled CLI reports `total_cost_usd` per
          response (accumulated into `_cumulative_cost_usd` in `_track_usage_metrics`) --
          preferred whenever present.
        - Raw Anthropic SDK backend (e.g. the MetaGen Llama passthrough): the API returns
          no dollar figure, so cost is computed from token counts via
          `llm_handlers.pricing.compute_cost_usd` (alias-aware, so `*-genai` model ids
          resolve to the right rate).
        Resets the batch counters.
        """
        model = self.claude_agent_option_kwargs.get("model", "")
        # Prefer SDK-reported cumulative cost; fall back to computed.
        if self._cumulative_cost_usd > 0:
            cumulative_cost = self._cumulative_cost_usd
        else:
            cumulative_cost = compute_cost_usd(model, self._prompt_tokens, self._completion_tokens)
        # Batch cost = delta since last reported. Works for both SDK-reported and computed paths.
        batch_cost = max(0.0, cumulative_cost - self._last_reported_cumulative_cost)
        self._last_reported_cumulative_cost = cumulative_cost
        # The SDK-direct counter is per-batch; reset for next call.
        self._batch_cost_usd = 0.0
        return {
            "cost/batch_usd": batch_cost,
            "cost/cumulative_usd": cumulative_cost,
            "cost/claude_prompt_tokens": float(self._prompt_tokens),
            "cost/claude_completion_tokens": float(self._completion_tokens),
        }

    def get_messages_from_state(
        self,
        state: EnvironmentState,
        default_context: list[dict[str, Any]] | None = None,
        include_system_prompt: bool = False,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """
        Get messages from the environment state, in the form of
        [{"role": <role>, "content": <content>}, ...]

        For multiple tries, as default behavior we build the context as:
        [system_prompt, prior_rollouts, *current_rollout]

        where *current_rollout is a list of the current:
        [prior_messages, last_model_response, new_messages]

        Returns:
        - all_messages: list of all messages in the context
        - new_messages: list of new messages (next_observations) from the environment
        """
        # Initialize current rollout messages as state's
        # `prior observations + model's last response + environment new messages`
        # 1. Hacky, but remove system prompt and default context if in prior messages
        default_context = default_context or state.default_context or []
        prior_messages = state.prior_messages or []
        prior_messages = remove_prefix_messages(prior_messages, default_context)

        # 2. Process new messages (observations) to {"role": <role>, "content": <content>} format
        #    - See `strl.environments` classes for environment responses
        new_messages = [
            {"role": msg["role"], "content": msg["output"]}
            if msg.get("type", "") == "function_call_output"  # bc we support OpenAI Responses API
            else msg
            for msg in (state.new_messages or [])
        ]

        # 3. Consolidate all messages, apply final processing
        model_response: list[dict[str, str]] = state.model_response or []
        all_messages = (state.prior_messages or []) + model_response + new_messages

        if all_messages and not include_system_prompt and all_messages[0].get("role") == "system":
            all_messages = all_messages[1:]

        return all_messages, new_messages

    def update_messages_from_past_rollouts(
        self,
        state: EnvironmentState,
        current_messages: list[dict[str, Any]],
        split: str | None = None,
        timestep: int | None = None,
        try_step: int | None = None,
        replay_buffer: ReplayBuffer | None = None,
        df_past_rollouts: DataFrame | None = None,
    ) -> list[dict[str, Any]]:
        """
        Update the current messages with the past rollouts.
        -> Default behavior is to prepend the past rollouts to the current messages,
           but we can override this behavior in subclasses.
        """
        state_messages = deepcopy(current_messages)
        
        # Base case: no retrieval available. Just return current try's messages
        if df_past_rollouts is None or len(df_past_rollouts) == 0:
            return state_messages

        # Otherwise, get past try messages and update state_messages
        # -> As default behavior, prepend the last try's messages for a new try
        if try_step > 0 and timestep == 0 and replay_buffer is not None:
            _retrieve_kwargs = {
                "split": split,
                "batch_id": state.batch_id,
                "sample_id": state.sample_id,
                "generation_id": state.generation_id,
                "try_step": try_step - 1,
            }
            # The concatenation baseline wants an EXACT-match lookup of this
            # sample's previous try. Memory buffers override
            # `get_past_episode_steps` with similarity retrieval (different,
            # required kwargs -> TypeError here); they expose the plain filter
            # as `get_past_episode_steps_base` -- prefer it when present so
            # claude_query works with any buffer class.
            _get_steps = getattr(
                replay_buffer, "get_past_episode_steps_base", None,
            ) or replay_buffer.get_past_episode_steps
            past_episode_steps = _get_steps(**_retrieve_kwargs)

            # Concatenate past trajectory's last state
            last_step = past_episode_steps[-1]
            last_try_idx = last_step.try_step
            past_try_messages = last_step.state + [last_step.action] + last_step.next_obs
            # Other task metadata
            _outcome = "Succeeded!" if last_step.reward > 0 else "Failed!"
            # `split` is the rollout split passed by the caller
            # (EnvironmentState carries no split field)
            _same_or_diff_task = (
                "Same Task"
                if last_step.sample_id == state.sample_id and last_step.split == split
                else "Different Task"
            )
            _prompt_metadata = f"Try {last_try_idx + 1}, {_same_or_diff_task}, {_outcome}"
            # Remove system prompt from past try
            if past_try_messages and past_try_messages[0].get("role") == "system":
                past_try_messages = past_try_messages[1:]
            
            # Get final messages list
            state_messages = [
                {"role": "user", "content": f"**Past Attempt ({_prompt_metadata})**"},
                *past_try_messages,
                {"role": "user", "content": f"**Current Attempt (Try {try_step + 1})**"},
                *current_messages,
            ]

        return state_messages

    def build_retrievers_and_past_rollouts(
        self,
        replay_buffer: ReplayBuffer | None = None,
        sample_id: int | None = None,
        split: str = "train",
        try_step: int = 0,
        batch_id: int = 0,
        user_id: str | None = None,
        build_retrievers: bool = False,
    ) -> DataFrame | None:
        """
        Build df_past_rollouts (and optionally retrievers) from the replay buffer.

        Call this ONCE per try before dispatching concurrent do_group_rollout calls.
        All concurrent samples within a try share the same retrievers (read-only).

        EXCEPTION: with ``use_same_user_ids`` (passing ``user_id``), the DPLM generator calls
        this PER SAMPLE inside do_single_rollout to re-scope memory to one persona. That
        rebuilds the shared ``replay_buffer.retrievers`` and is therefore NOT concurrency-safe
        -- run sequentially (no --concurrent, --tasks_per_batch 1).
        """
        replay_buffer = replay_buffer or self.replay_buffer
        if replay_buffer.pd_df_buffer is None or len(replay_buffer.pd_df_buffer) == 0:
            return None

        # Build filter kwargs for retrieving past rollouts
        filter_kwargs: dict[str, Any] = {}
        if try_step > 0 and self.use_last_try_only:
            filter_kwargs["try_step"] = try_step - 1
        if self.use_same_user_ids and user_id is not None:
            filter_kwargs["user_id"] = user_id
        if self.same_sample_tries and sample_id is not None:
            filter_kwargs["sample_id"] = sample_id
            filter_kwargs["split"] = split
        else:
            filter_kwargs["split"] = "train"  # can always retrieve from training split
        df_past_rollouts = replay_buffer.get_df_for_trajectory(**filter_kwargs)
        
        if split != "train" and sample_id is not None:
            extra_kwargs = {"batch_id": batch_id, "split": split, "sample_id": sample_id}
            df_past_rollouts = pd_concat(
                [df_past_rollouts, replay_buffer.get_df_for_trajectory(**extra_kwargs)]
            )
        if len(df_past_rollouts) == 0:
            logger.warning(
                "No replay buffer rows match filter %s; skipping memory retrieval.",
                filter_kwargs,
            )
            return None

        if build_retrievers:
            logger.info(
                "Building retrievers from %d replay buffer rows (split=%s, sample_id=%s)",
                len(df_past_rollouts),
                split,
                sample_id,
            )
            df_past_rollouts = (
                replay_buffer.text_feature_builder.build_text_features(df_past_rollouts)
            )
            replay_buffer.build_retrievers(df_past_rollouts)
        return df_past_rollouts
    
    async def _on_action_accepted(
        self, model_messages: list[dict[str, str]], sample_id: int = 0,
    ) -> None:
        """
        Narration/extension hook fired once per step with the *committed* action — i.e. after any
        LLM-as-a-judge resampling settles on the action that will actually be stepped. No-op by
        default; `TTSMixin` overrides it to speak only the accepted action (not rejected samples).
        """
        return None

    async def do_single_rollout(
        self,
        env: Environment | None = None,
        df_past_rollouts: DataFrame | None = None,
        hf_tokenizer: PreTrainedTokenizerBase | None = None,
        split: str = "train",
        batch_id: int = 0,
        sample_id: int = 0,
        generation_id: int = 0,
        try_step: int = 0,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> Trajectory:
        """
        Run one full episode and return the resulting Trajectory.
        """
        raise NotImplementedError(
            "`do_single_rollout` is not implemented for BaseClaudeAgentSDKGenerator. "
            "Please implement in a subclass."
        )

    async def do_group_rollout(
        self,
        num_return_sequences: int,
        **single_rollout_kwargs: Any,
    ) -> dict[str, list[TrajectoryGroup]]:
        """
        Generate a group of trajectories in the environment and
        return a list of the trajectory group(s).

        By default, we should just return a singleton with 1 TrajectoryGroup. However, there may
        be cases for >1 TrajectoryGroups, e.g., if we're generating multiple actions per step,
        and we want advantages over each (state, action, next_obs) tuple across generations
        """
        trajectories_in_group: list[Trajectory] = await asyncio.gather(
            *[
                self.do_single_rollout(generation_id=gen_idx, **single_rollout_kwargs)
                for gen_idx in range(num_return_sequences)
            ]
        )
        # Filter out empty trajectories (e.g. from context overflow or timeout before any step)
        trajectories_in_group = [t for t in trajectories_in_group if len(t.episode_steps) > 0]
        if not trajectories_in_group:
            logger.warning("All rollouts produced empty trajectories, skipping group")
            return {"policy": []}
        all_trajectory_groups = [
            self._get_trajectory_group(
                trajectories=trajectories_in_group,
                discount_factor=self.discount_factor,
            )
        ]
        # Update replay buffer with new trajectories
        for trajectory_group in all_trajectory_groups:
            trajectory_group.compute_advantages()
            self.replay_buffer.add_trajectory_group(trajectory_group)
        self.replay_buffer.update_buffer_ds_and_df()

        # Sometimes we may want different trajectories to evaluate vs train on
        return {"policy": all_trajectory_groups}
