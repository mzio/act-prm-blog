"""
Default Claude Agent SDK Generator using claude_agent_sdk `query()` method
"""

import asyncio
import logging
from typing import Any

from claude_agent_sdk import AssistantMessage, ClaudeAgentOptions, ResultMessage, query
from pandas import DataFrame
from rich.console import Console
from rich.panel import Panel
from transformers import PreTrainedTokenizerBase

from act_prm.environments import Environment, EnvironmentState, EnvironmentStepResult
from act_prm.llm_handlers.action_utils import get_actions
from act_prm.llm_handlers.types import ActionFromLLM
from act_prm.replay_buffer import ReplayBuffer
from act_prm.replay_buffer.types import EpisodeStep, Trajectory
from act_prm.utils.claude_auth import load_dotenv_auth
from act_prm.utils.display import display_state_action_next_obs

from .prompts import get_chat_system_prompt_template
from .anthropic_backend import anthropic_messages_response, use_anthropic_backend
from .base import BaseClaudeAgentSDKGenerator
from .types import ClaudeAgentResponse
from .utils import (
    build_model_message_content,
    convert_tools_dict_to_str,
    display_actions,
    get_actions_from_response,
    get_prompt_from_messages,
)

logger = logging.getLogger(__name__)
console = Console()

ROYGBIV = ["#FF0000", "#FF7F00", "#FFFF00", "#00FF00", "#0000FF", "#4B0082", "#9400D3"]
DEBUG_COLS = ["batch_id", "split", "try_step", "generation_id", "sample_id"]

# Anthropic's extended-thinking floor: `thinking.budget_tokens` must be >= this,
# and the budget must fit *under* the max output budget. A caller that caps
# output below this (e.g. a JSON-verdict judge with max_new_tokens=800) while
# `effort` would otherwise drive adaptive thinking has no valid budget, so the
# API 400s ("thinking.enabled.budget_tokens: Input should be >= 1024").
MIN_THINKING_BUDGET_TOKENS = 1024


async def sample_query_response(
    prompt: str,
    max_tokens: int | None = None,
    temperature: float | None = None,  # ignored
    timeout: float | None = 120,  # None disables the timeout
    verbose: bool = False,
    **claude_agent_option_kwargs: Any,
) -> ClaudeAgentResponse:
    """
    Sample a response from the Claude Agent SDK using the `query()` method.

    Raises `asyncio.TimeoutError` if the full query stream does not complete
    within `timeout` seconds (callers retry via `_get_claude_response`).

    When STRL_CLAUDE_BACKEND=anthropic_api (see anthropic_backend.py), routes to the
    raw Anthropic SDK instead -- for endpoints (e.g. Meta's Llama passthrough) whose
    model id the bundled Claude Code CLI rejects. The query() path is stateless, so it
    maps cleanly onto a single Messages call.
    """
    # Load .env auth FIRST: this sets ANTHROPIC_* + STRL_CLAUDE_BACKEND in os.environ
    # (memoized/idempotent), which both the backend check below and the raw Anthropic
    # client depend on.
    auth_env = load_dotenv_auth()
    if use_anthropic_backend():
        return await anthropic_messages_response(
            prompt,
            model=claude_agent_option_kwargs.get("model"),
            system_prompt=claude_agent_option_kwargs.get("system_prompt"),
            max_tokens=max_tokens,
            timeout=timeout,
        )

    # Initialize Claude Agent SDK options. Auth comes purely from .env: merge the
    # scrubbed auth env (see utils/claude_auth.py) beneath any caller-provided env,
    # and ignore user/project/local settings so only .env drives auth.
    claude_agent_option_kwargs = dict(claude_agent_option_kwargs)
    env = {**auth_env, **(claude_agent_option_kwargs.get("env") or {})}
    if max_tokens is not None:
        env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] = str(max_tokens)
        # Extended thinking can't fit under an output cap below its floor. When a
        # caller caps output that low without explicitly choosing a thinking
        # mode, disable thinking so `effort` doesn't trigger adaptive thinking
        # and 400 the request. Callers that set `thinking` themselves win.
        if max_tokens < MIN_THINKING_BUDGET_TOKENS and "thinking" not in claude_agent_option_kwargs:
            claude_agent_option_kwargs["thinking"] = {"type": "disabled"}
            logger.debug(
                "max_tokens=%d < %d (thinking floor); disabling extended thinking for this query.",
                max_tokens, MIN_THINKING_BUDGET_TOKENS,
            )
    if env:
        claude_agent_option_kwargs["env"] = env
    claude_agent_option_kwargs.setdefault("setting_sources", [])
    options = ClaudeAgentOptions(**claude_agent_option_kwargs)

    # Sample response using `query()`
    response = ClaudeAgentResponse()

    async def _consume_stream() -> None:
        async for msg in query(prompt=prompt, options=options):
            if isinstance(msg, AssistantMessage):
                response.assistant_messages.append(msg)
            elif isinstance(msg, ResultMessage):
                response.result = msg
                if msg.usage:
                    response.usage = {
                        "input_tokens": msg.usage.get("input_tokens", 0),
                        "output_tokens": msg.usage.get("output_tokens", 0),
                    }
                if getattr(msg, "total_cost_usd", None):
                    response.cost = msg.total_cost_usd

    try:
        if timeout is not None:
            await asyncio.wait_for(_consume_stream(), timeout=timeout)
        else:
            await _consume_stream()
    except Exception as e:
        # Newer Claude Code CLIs (claude-agent-sdk >= ~0.2.9x) intercept our
        # protocol's literal <tool_call> tags in the model's PLAIN-TEXT output,
        # try to parse them as their own tool calls, auto-inject a retry turn
        # on failure, and then exit non-zero ("Reached maximum number of turns"
        # / "The model's tool call could not be parsed"). The assistant text we
        # actually want usually arrived fine before that — salvage it and let
        # OUR parser decide, instead of discarding it and re-sending the whole
        # (often huge) prompt. Timeouts and transport errors still propagate.
        if "returned an error result" in str(e) and response.assistant_messages:
            logger.warning(
                "Claude CLI reported an error result (%s); salvaging %d assistant message(s).",
                str(e)[:120], len(response.assistant_messages),
            )
        else:
            raise

    if verbose:
        for msg_idx, msg in enumerate(response.assistant_messages):
            str_msg = str(msg)
            if "content=[TextBlock" in str_msg:
                _border_style = "dim bright_blue"
            elif "content=[ThinkingBlock" in str_msg:
                _border_style = "dim bright_magenta"
            elif "content=[ToolUseBlock" in str_msg:
                _border_style = "dim bright_red"
            elif "content=[ToolResultBlock" in str_msg:
                _border_style = "dim bright_yellow"
            else:
                _border_style = "dim"
            console.print(Panel(
                str_msg,
                title=f"** Claude Assistant Message {msg_idx + 1} (sample_query_response()) **",
                border_style=_border_style,
                style="dim",
            ))
        if getattr(response, "result", None):
            console.print(Panel(
                str(response.result),
                title="** Claude Result Message (sample_query_response()) **",
                style="dim yellow",
            ))
    # logger.info("Breakpoint at sample_query_response() in src/strl/generator/claude_agent_sdk/query.py")
    # breakpoint()
    return response


class QueryClaudeAgentSDKGenerator(BaseClaudeAgentSDKGenerator):
    """
    Base class for generating rollouts using Claude Agent SDK
    """

    def __init__(
        self,
        method_name: str = "Default (Claude Agent SDK Query)",
        prompt_name: str = "chat",
        # Claude Agent SDK parameters
        model: str = "claude-sonnet-4-6",
        permission_mode: str = "default",
        effort: str = "low",
        max_agent_turns: int | None = None,
        **kwargs: Any,
    ) -> None:
        # query() should yield a single assistant turn (optional thinking + one
        # text response carrying our <tool_call>); the CLI must not run a
        # multi-turn tool loop. base._init_claude_options defaults max_turns=1
        # when max_agent_turns is None, so we pass it through unchanged.
        super().__init__(
            method_name=method_name,
            model=model,
            permission_mode=permission_mode,
            effort=effort,
            max_agent_turns=max_agent_turns,
            **kwargs,
        )
        self.prompt_name = prompt_name

    async def _get_claude_response(
        self,
        prompt_text: str,
        sample_id: int,
        max_calls: int = 10,
        max_tokens: int | None = None,
        temperature: float | None = None,
        timeout: float | None = None,
        tool_call_bos: str = "<tool_call>",
        tool_call_eos: str = "</tool_call>",
        verbose_colors: str | None = None,
        verbose_suffix: str | None = None,
    ) -> tuple[list[dict[str, str]], ClaudeAgentResponse]:
        """
        Get a response from the Claude Agent SDK using the `query()` method.

        Returns (model_messages, claude_response).
        """
        max_tokens = max_tokens or self.max_tokens
        temperature = temperature or self.temperature
        timeout = timeout or self.timeout

        original_prompt_text = prompt_text
        num_calls = 0
        while num_calls < max_calls:
            try:
                claude_response = await sample_query_response(
                    prompt_text,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    timeout=timeout,
                    verbose=self.verbose,
                    **self.claude_agent_option_kwargs,
                )
                self._track_usage_metrics(claude_response)
                # A bit roundabout, but get list[ActionFromLLM] object from claude_response
                # to parse into our schema. Then convert from this to model_messages list[dict[str, str]]
                llm_actions = get_actions_from_response(claude_response)

                if self.verbose:
                    # console.print(f"[dim]Claude Prompt:\n{prompt_text}[/dim]")
                    prompt_style = "dim" if verbose_colors is None else f"dim {verbose_colors}".replace("dim dim", "dim")
                    action_style = f"color({(sample_id + 1) % 8 + 8})" if verbose_colors is None else verbose_colors
                    _title = "** Claude Prompt **" if verbose_suffix is None else f"** Claude Prompt ({verbose_suffix}) **"
                    console.print(Panel(prompt_text, title=_title, style=prompt_style, border_style=prompt_style))
                    display_actions(llm_actions, base_color=action_style, title_suffix=verbose_suffix)

                # logger.info("Breakpoint at QueryClaudeAgentSDKGenerator._get_claude_response() in src/strl/generator/claude_agent_sdk/query.py")
                # breakpoint()

                # Collapse parsed actions into one assistant message, concatenating
                # thinking + text up to (and including) the first complete tool call
                # and discarding any hallucinated trailing tool_result/tool_calls.
                # Shared with client_strl via utils.build_model_message_content.
                content = build_model_message_content(llm_actions, tool_call_bos, tool_call_eos)
                return [{"role": "assistant", "content": content}], claude_response

            except Exception as e:
                logger.error("Claude action parse error (%s): %s", e.__class__.__name__, e)
                # Keep the full task context on retry — `query()` is stateless,
                # so a bare error message would ask Claude to act blind.
                prompt_text = (
                    f"{original_prompt_text}\n\n"
                    f"Your previous response was not parsed correctly, please try again.\n\n{e}"
                )
                num_calls += 1

        # Exhausted retries
        raise RuntimeError(f"Failed to get valid Claude action after {max_calls} attempts")

    async def _sample_messages(
        self,
        messages: list[dict[str, str]],
        sample_id: int,
        max_calls: int = 10,
        max_tokens: int | None = None,
        temperature: float | None = None,
        timeout: float | None = None,
        tool_call_bos: str = "<tool_call>",
        tool_call_eos: str = "</tool_call>",
        verbose_colors: str | None = None,
        verbose_suffix: str | None = None,
    ) -> tuple[list[dict[str, str]], ClaudeAgentResponse]:
        """
        Model-call seam shared by query_strl's reflection / action / judge sites.

        Default: flatten the messages into a single prompt and delegate to
        `_get_claude_response` (preserves the historical query() behavior). The Anthropic
        structured generator (`query_strl_api.ClaudeApiStrlGenerator`) overrides this to
        send a native multi-turn Anthropic `messages` array instead.

        Returns (model_messages, claude_response).
        """
        prompt_text = get_prompt_from_messages(messages, system_prompt=None, delimiter="\n\n")
        return await self._get_claude_response(
            prompt_text=prompt_text,
            sample_id=sample_id,
            max_calls=max_calls,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout,
            tool_call_bos=tool_call_bos,
            tool_call_eos=tool_call_eos,
            verbose_colors=verbose_colors,
            verbose_suffix=verbose_suffix,
        )

    def get_messages_from_state(
        self,
        state: EnvironmentState,
        default_context: list[dict[str, Any]] | None = None,
        include_system_prompt: bool = False,
    ) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
        """
        Inherited from base.py class
        """
        return super().get_messages_from_state(
            state=state,
            default_context=default_context,
            include_system_prompt=include_system_prompt,
        )

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
        Inherited from base.py class
        """
        return super().update_messages_from_past_rollouts(
            state=state,
            current_messages=current_messages,
            split=split,
            timestep=timestep,
            try_step=try_step,
            replay_buffer=replay_buffer,
            df_past_rollouts=df_past_rollouts,
        )

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
        Inherited from base.py class
        """
        return super().build_retrievers_and_past_rollouts(
            replay_buffer=replay_buffer,
            sample_id=sample_id,
            split=split,
            try_step=try_step,
            batch_id=batch_id,
            user_id=user_id,
            build_retrievers=build_retrievers,
        )

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
        env = env or self.env
        hf_tokenizer = hf_tokenizer or self.hf_tokenizer

        episode_steps: list[EpisodeStep] = []
        final_reward: float = 0.0
        truncated: bool = False

        state: EnvironmentState = await env.reset_async(
            sample_id=sample_id,
            generation_id=generation_id,
            try_step=try_step,
            batch_id=batch_id,
        )
        done = False

        # Run agent loop
        try:
            while not done:
                # Get system prompt
                tools_str = convert_tools_dict_to_str(state.tools)
                system_prompt = get_chat_system_prompt_template(self.prompt_name).format(
                    system_prompt=state.system_prompt,
                    tools=tools_str,
                )
                # Update messages from latest environment state
                state_messages, new_messages = self.get_messages_from_state(
                    state=state,
                    default_context=state.default_context,
                    include_system_prompt=False,
                )
                # Update messages with past trajectories
                if df_past_rollouts is not None and len(df_past_rollouts) > 0:
                    # e.g., by default just prepend the prior trajectory
                    state_messages = self.update_messages_from_past_rollouts(
                        state=state,
                        current_messages=state_messages,
                        split=split,
                        timestep=state.timestep,
                        try_step=state.try_step,
                        replay_buffer=self.replay_buffer,
                        df_past_rollouts=df_past_rollouts,
                    )

                # "Apply chat template" to format messages for query prompt
                state_messages = [
                    {"role": "system", "content": system_prompt},
                    *state_messages,
                ]
                prompt: str = get_prompt_from_messages(state_messages, system_prompt=None, delimiter="\n\n")
                try:
                    model_messages, claude_response = await self._get_claude_response(
                        prompt_text=prompt,
                        sample_id=sample_id,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        timeout=self.timeout,
                    )
                    is_complete = True
            
                except RuntimeError as e:
                    logger.warning("Claude action failed after retries, ending episode")
                    logger.error("Error: %s", e)
                    done = True
                    truncated = True
                    is_complete = False
                    break

                if self.verbose and generation_id == 0:  #  and sample_id == 0:
                    # console.print(f"[bold magenta]Claude Action:[/bold magenta]\n{model_messages[0]['content']}")
                    console.print(Panel(
                        model_messages[0]["content"],
                        title="** Claude Action **",
                        # border_style="magenta",
                        style="dodger_blue1"
                    ))
            
                parsed_actions: list[ActionFromLLM] = get_actions(model_messages)
                # Output tokens for envs that simulate a per-token time cost
                # (e.g. Gaia2Env). Envs that don't care silently swallow it.
                _claude_usage = (claude_response.usage if claude_response else None) or {}
                last_output_tokens = int(_claude_usage.get("output_tokens", 0) or 0)

                # 4. Step through environment with Claude's action
                env_step_result: EnvironmentStepResult = await env.step_async(
                    parsed_actions=parsed_actions,
                    model_response=model_messages,
                    current_state=state,
                    current_messages=state_messages,
                    last_output_tokens=last_output_tokens,
                )
            
                next_state = env_step_result.state
                reward = env_step_result.reward
                done = env_step_result.done
                truncated = env_step_result.truncated

                # 5. Save EpisodeStep
                next_obs = [
                    {
                        "role": msg["role"],
                        "content": msg["output"] if msg.get("output", None) else msg["content"],
                    }
                    for msg in next_state.new_messages
                ]
                # For now, no training
                state_action_ids = []
                state_ids = []
                action_logprobs = []
            
                episode_steps.append(
                    EpisodeStep(
                        state=state_messages,  # list[dict[str, str]]
                        action=model_messages[0],    # dict[str, str]
                        next_obs=next_obs,     # list[dict[str, str]]
                        tools=state.tools,
                        state_action_tokens=state_action_ids,
                        state_len=len(state_ids),
                        old_logprobs=action_logprobs,
                        temperature=temperature,
                        reward=reward,
                        done=done,
                        truncated=truncated,
                        timestep=state.timestep,
                        try_step=state.try_step,
                        batch_id=batch_id,
                        sample_id=sample_id,
                        generation_id=generation_id,
                        split=split,
                        system_prompt=state.system_prompt,
                        task_prompt=state.task_prompt,
                        user_id=state.user_id,
                        default_context=state.default_context,
                        is_complete=is_complete,
                    )
                )
                if self.verbose:
                    _header_text = (
                        f"(Method: {self.method_name}) "
                        f"{split.title()} Split, Try {try_step}, Batch {batch_id},"
                        f" Sample {sample_id}, Generation {generation_id},"
                        f" Timestep {state.timestep}"
                    )
                    display_state_action_next_obs(
                        generator=self,
                        generation_id=generation_id,
                        state_messages=state_messages,
                        action_messages=model_messages,
                        next_obs_messages=next_obs,
                        tools=state.tools,
                        hf_tokenizer=hf_tokenizer,
                        header_text=_header_text,
                        group_rewards=[reward],  # final_reward
                        state=state,
                    )
                # Transition to next state
                state = next_state
                final_reward = reward
                done = done or truncated
        finally:
            await env.close_rollout(state)

        return Trajectory(
            episode_steps=episode_steps,
            try_step=try_step,
            discount_factor=self.discount_factor,
            final_reward=final_reward,
        )
