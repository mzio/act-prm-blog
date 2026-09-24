"""
Helper functions for Critic-Actor on top of the Claude Agent SDK
"""

import asyncio
import json
import logging
from typing import Any

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeSDKClient,
    ResultMessage,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from rich.console import Console
from rich.panel import Panel

from act_prm.llm_handlers.types import ActionFromLLM

from .types import (
    ClaudeAgentResponse,
    is_text_block,
    is_thinking_block,
    is_tool_result_block,
    is_tool_use_block,
)

logger = logging.getLogger(__name__)
console = Console()


def convert_tools_dict_to_str(
    tools: list[dict[str, Any]],
    include_type: bool = False,  # 'type': 'function'
) -> str:
    """
    Convert list of tool dicts to a markdown string
    """

    def get_markdown_from_dict(
        tool_dict: dict[str, Any],
        include_type: bool = False,
    ) -> str:
        """
        Convert a single tool dict to a markdown string
        """
        parts: list[str] = [f'name: "{tool_dict["name"]}"']
        keys = [k for k in tool_dict.keys() if k != "type" and k != "name"]
        if include_type:
            keys += ["type"]
        for k in keys:
            v = json.dumps(tool_dict[k]) if isinstance(tool_dict[k], (dict, list)) else tool_dict[k]
            parts.append(f"- {k}: {v}")

        return "\n".join(parts)

    return "\n\n".join([get_markdown_from_dict(t, include_type) for t in tools])


def _get_markdown_from_messages(
    messages: list[dict[str, Any]],
    max_chars: int | None = None,
    delimiter: str = "\n\n---\n\n",
) -> str:
    """
    Get a markdown string from a list of messages
    """

    def _format_role(role: str) -> str:
        """Format role for markdown string"""
        if role == "assistant":
            return "assistant (you)"  # Note that without spacing between [ ],
        if role == "tool":              # rich will interpret as invalid colors (won't display)
            return "tool_response"
        return f"{role}"

    def _format_content(content: str, max_chars: int | None = None) -> str:
        """Format content for markdown string"""
        if max_chars is not None and len(content) > max_chars:
            return content[:max_chars] + "... (truncated for brevity)"
        return content

    # Indicate index of messages, and potentially truncate messages to include
    n_msgs = len(messages)
    messages = [
        msg
        if msg["role"] == "assistant" 
        else {"role": msg["role"], "content": _format_content(msg["content"], max_chars)}
        for msg in messages
    ]

    return delimiter.join([
        f"# {_idx + 1}. {_format_role(msg["role"]).upper()} ({_idx + 1} / {n_msgs}):"
        f"\n{msg["content"].strip()}\n"
        for _idx, msg in enumerate(messages)
    ]).replace("\n\n\n", "\n\n")




def get_prompt_from_messages(
    messages: list[dict[str, Any]],
    system_prompt: str | None = None,
    # delimiter: str = "\n\n",
    delimiter: str = "\n\n---\n\n",
) -> str:
    """
    Convert a list of chat messages into a single prompt string for client.query().

    For single-turn use, we format the conversation history as a prompt.
    """
    # return _get_markdown_from_messages(messages, delimiter=delimiter)
    parts: list[str] = []
    _idx = 1
    if system_prompt:
        parts.append(f"[System]\n{system_prompt}")
        _idx += 1

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if role == "tool" or msg.get("type") == "function_call_output":
            call_id = msg.get("call_id", None)  # msg.get("call_id", "unknown")
            call_id = f" ({call_id})" if call_id else ""
            # parts.append(f"[Tool Result{call_id}]\n{content}")
            parts.append(f"<tool_result{call_id}>\n{content}\n</tool_result{call_id}>")
            # parts.append(
            #     f"# Message {_idx}"
            # )

        elif role == "assistant":
            # parts.append(f"[Assistant]\n{content}")
            parts.append(f"<assistant>\n{content}\n</assistant>")

        elif role == "user":
            # parts.append(f"[User]\n{content}")
            parts.append(f"<user>\n{content}\n</user>")

        elif role == "system":
            # parts.append(f"[System]\n{content}")
            parts.append(f"<system>\n{content}\n</system>")

        _idx += 1

    return delimiter.join(parts)


async def _sample_client_response_impl(
    client: ClaudeSDKClient,
    prompt: str,
    verbose: bool = False,
) -> ClaudeAgentResponse:
    """Inner implementation without timeout."""
    response = ClaudeAgentResponse()
    await client.query(prompt)

    async for msg in client.receive_response():
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
                title=f"** Claude Assistant Message {msg_idx + 1} (_sample_client_response_impl()) **",
                border_style=_border_style,
                style="dim",
            ))
    return response


async def sample_client_response(
    client: ClaudeSDKClient,
    prompt: str,
    timeout: float = 120,
    verbose: bool = False,
    **kwargs: Any,
) -> ClaudeAgentResponse:
    """
    Sample a response from the Claude Agent SDK client, with timeout.
    """
    try:
        return await asyncio.wait_for(
            _sample_client_response_impl(client, prompt, verbose),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        logger.warning("Claude SDK sample timed out after %.0fs, returning empty response", timeout)
        return ClaudeAgentResponse()


def get_actions_from_response(
    response: ClaudeAgentResponse,
    **kwargs: Any,
) -> list[ActionFromLLM]:
    """
    Get actions from a Claude Agent SDK response
    """
    actions: list[ActionFromLLM] = []
    for _, block in enumerate(response.output):
        # ActionFromLLM tool call default kwargs
        tool_call_kwargs: dict[str, Any] = {"call_id": None, "name": None, "arguments": None}

        if is_thinking_block(block) or isinstance(block, ThinkingBlock):
            actions.append(
                ActionFromLLM(
                    role="assistant",
                    type="reasoning",
                    text=block.thinking,
                    **tool_call_kwargs,
                )
            )

        elif (is_tool_use_block(block) or isinstance(block, ToolUseBlock)) and block.name == "StructuredOutput":
            actions.append(
                ActionFromLLM(
                    role="assistant",
                    type="function_call",
                    text=json.dumps(block.input),
                    call_id=block.id,
                    name=block.name,
                    arguments=block.input,
                )
            )

        elif is_tool_result_block(block) or isinstance(block, ToolResultBlock):
            actions.append(
                ActionFromLLM(
                    role="assistant",
                    type="function_call_output",
                    text=json.dumps(block.output),
                    call_id=block.tool_use_id,
                    name=block.name,
                    arguments=block.output,
                )
            )

        elif is_text_block(block) or isinstance(block, TextBlock):
            actions.append(
                ActionFromLLM(
                    role="assistant",
                    type="message",
                    text=block.text,
                    **tool_call_kwargs,
                )
            )
    return actions


def build_model_message_content(
    llm_actions: list[ActionFromLLM],
    tool_call_bos: str = "<tool_call>",
    tool_call_eos: str = "</tool_call>",
    require_tool_call: bool = False,
    thinking_bos: str = "",
    thinking_eos: str = "",
) -> str:
    """
    Collapse parsed Claude actions into a single assistant-message content string.

    Concatenates thinking + text up to (and INCLUDING) the first complete
    ``<tool_call>...</tool_call>`` and stops there, discarding any hallucinated
    ``<tool_result>`` / extra ``<tool_call>`` blocks the model may have appended
    within its single turn (see query.py / client_strl.py callers). Shared by the
    stateless query() and stateful client() paths so both truncate identically.

    ``thinking_bos`` / ``thinking_eos`` wrap the reasoning text. They default to EMPTY so
    the emitted content is ``<thought>\n\n<tool_call>...</tool_call>`` -- byte-compatible
    with every existing SFT corpus (``data/sft_corpus/**``, ``data/*_expert_thoughts``),
    whose assistant content is plain thought text followed by the tool call. The old
    ``**Thinking Start** / **Thinking End**`` markers would otherwise be trained into the
    policy as literal tokens it has to reproduce, and would break the ``_has_thought``
    /thought-length statistics that compare corpora. Pass them explicitly to restore the
    delimited form.

    ``require_tool_call`` makes a turn that produced text/thinking but NO
    ``<tool_call>`` a parse failure (raise -> the caller's retry loop re-prompts)
    instead of silently returning reasoning-only content, which the env then
    rejects with "No tool calls parsed" -- burning an env turn. Pass it ONLY from
    the policy-action samplers; reflection/judge calls legitimately have no tool
    call and must keep the default (``False``).

    Raises ``ValueError`` if no usable content was produced (callers retry).
    """
    content = ""
    for action in llm_actions:
        if action.type == "reasoning":
            content += (
                f"{thinking_bos}{action.text}{thinking_eos}\n\n"
            ).replace("\n\n\n", "\n\n")

        elif action.type == "function_call" and action.name == "StructuredOutput":
            # See get_actions_from_response(): StructuredOutput packs the
            # reasoning + the literal <tool_call> block into its arguments.
            reasoning = action.arguments.get("reasoning", "")
            tool_call = action.arguments.get("tool_call", "")
            if tool_call_bos not in tool_call or tool_call_eos not in tool_call:
                raise ValueError(f"Missing <tool_call> tags in: {tool_call}")
            content += f"{reasoning}\n\n{tool_call}"
            return content

        elif action.type == "message":
            content += action.text
            if tool_call_bos in content and tool_call_eos in content:
                # Cut at the first </tool_call>; drop any hallucinated trailing
                # tool_result / extra tool_calls.
                return f"{content.split(tool_call_eos)[0]}{tool_call_eos}".strip()

    # Reaching here means no complete <tool_call> (nor StructuredOutput) was
    # found -- only thinking/text. For policy turns that's a malformed action:
    # raise so the caller re-prompts, with feedback the model can act on.
    if require_tool_call:
        raise ValueError(
            "Your response contained no <tool_call> block. Respond with a brief reasoning "
            "line, then EXACTLY ONE <tool_call>...</tool_call> block describing the next action."
        )
    if content != "":
        return content
    raise ValueError(f"No parsed actions found in Claude response:\n{llm_actions}")


def display_actions(
    actions: list[ActionFromLLM],
    base_color: str,
    title_suffix: str | None = None,
) -> None:
    """
    Display actions in a readable format
    """
    for action_idx, action in enumerate(actions):
        if action.type == "reasoning":
            style = f"italic {base_color}"
        elif action.type == "message":
            style = f"{base_color}"
        elif action.type == "function_call":
            style = f"bold {base_color}"
        else:
            style = f"dim {base_color}"

        title = f"Display Claude Actions ({action_idx}. {action.type})"
        if title_suffix:
            title += f" ({title_suffix})"
        console.print(Panel(f"{action.text}", title=title, style=style))


def strip_mcp_prefix(name: str, prefix: str = "mcp__env_tools__") -> str:
    """Strip MCP server prefix from tool names (e.g. 'mcp__env_tools__get_weather' -> 'get_weather')."""
    if name.startswith(prefix):
        return name[len(prefix) :]
    return name
