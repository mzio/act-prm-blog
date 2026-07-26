"""
Claude Agent SDK LLM handlers

Two subclasses of LLM using the Claude Agent SDK:
- ClaudeQueryLLM: Uses `query()` for stateless, one-shot interactions
- ClaudeClientLLM: Uses `ClaudeSDKClient` for persistent, multi-turn sessions
"""

import asyncio
import json
import os
import queue
import threading
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
    create_sdk_mcp_server,
    query,
)
from claude_agent_sdk import (
    tool as sdk_tool,
)
from dotenv import load_dotenv

from .base import LLM
from .types import ActionFromLLM

# Default timeout (seconds) for async operations run from sync wrappers
_ASYNC_TIMEOUT = 300

load_dotenv()


def _clear_nested_session_guard() -> None:
    """Remove CLAUDECODE env var to allow launching Claude Code subprocess.

    The Claude Agent SDK spawns Claude Code as a subprocess, which refuses to
    start if CLAUDECODE is set (to prevent nested sessions). We clear it only
    when actually making SDK calls, not at import time.
    """
    os.environ.pop("CLAUDECODE", None)


@dataclass
class ClaudeAgentResponse:
    """
    Response container that mirrors the structure expected by get_actions().
    Collects AssistantMessages from the Claude Agent SDK stream.
    """

    assistant_messages: list[AssistantMessage] = field(default_factory=list)
    result: ResultMessage | None = None
    usage: dict[str, int] | None = None

    @property
    def output(self) -> list[Any]:
        """Flatten all content blocks from all assistant messages."""
        blocks = []
        for msg in self.assistant_messages:
            blocks.extend(msg.content)
        return blocks


def _convert_tools_to_mcp_server(
    tools: list[dict[str, Any]],
    server_name: str = "env_tools",
) -> tuple[dict[str, Any], list[str]]:
    """
    Convert OpenAI-style function tool definitions to an in-process MCP server.

    Expects tools in the format:
    [{"type": "function", "name": "tool_name", "description": "...",
      "parameters": {"type": "object", "properties": {...}, "required": [...]}}]
    """
    if not tools:
        return {}, []

    sdk_tools = []
    tool_names = []
    for tool_def in tools:
        name = tool_def.get("name", tool_def.get("function", {}).get("name", ""))
        description = tool_def.get("description", tool_def.get("function", {}).get("description", ""))
        parameters = tool_def.get("parameters", tool_def.get("function", {}).get("parameters", {}))

        @sdk_tool(name, description, parameters)
        async def _placeholder_handler(args: dict[str, Any], _name: str = name) -> dict[str, Any]:
            # This handler should never be called — the environment handles tool
            # execution. Return an error if somehow invoked.
            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"Error: tool '{_name}' should be handled by the environment, not the SDK.",
                    }
                ],
                "isError": True,
            }

        sdk_tools.append(_placeholder_handler)
        tool_names.append(f"mcp__{server_name}__{name}")

    server = create_sdk_mcp_server(
        name=server_name,
        version="1.0.0",
        tools=sdk_tools,
    )
    return {server_name: server}, tool_names


def _messages_to_prompt(
    system_prompt: str | None,
    messages: list[dict[str, Any]],
) -> str:
    """
    Convert a list of chat messages into a single prompt string for query().

    For single-turn use, we format the conversation history as a prompt.
    """
    parts = []
    if system_prompt:
        parts.append(f"[System]\n{system_prompt}")

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if msg.get("type") == "function_call_output":
            call_id = msg.get("call_id", "unknown")
            parts.append(f"[Tool Result ({call_id})]\n{content}")
        elif role == "assistant":
            parts.append(f"[Assistant]\n{content}")
        elif role == "user":
            parts.append(f"[User]\n{content}")
        elif role == "system":
            parts.append(f"[System]\n{content}")

    return "\n\n".join(parts)


def _update_messages_impl(
    messages: list[dict[str, Any]],
    model_response: ClaudeAgentResponse | None,
    prior_messages: list[dict[str, Any]],
    interleave: bool = False,
) -> list[dict[str, Any]]:
    """
    Shared update_messages logic for both ClaudeQueryLLM and ClaudeClientLLM.

    Converts ClaudeAgentResponse content blocks back into message dicts
    that can be passed in the next call's messages list.
    """
    if interleave or model_response is None:
        new_messages: list[dict[str, Any]] = []
    else:
        new_messages = _response_to_message_dicts(model_response)

    for message in messages:
        if message.get("type") == "function_call_output":
            msg = {k: v for k, v in message.items() if k != "role"}
            try:
                image_message = msg.pop("image_output")
                new_messages.extend([msg, image_message])
            except KeyError:
                new_messages.append(msg)
        else:
            new_messages.append(message)

    return prior_messages + new_messages


def _track_usage(llm: LLM, usage: dict[str, int]) -> None:
    """Track token usage from a ClaudeAgentResponse on an LLM instance."""
    llm.last_usage = llm._track_tokens(SimpleNamespace(**usage))
    if llm.last_usage is not None:
        print("Claude Agent SDK token usage:")
        for k, v in llm.last_usage.items():
            print(f"-> {k}: {v}")


class ClaudeQueryLLM(LLM):
    """
    Claude Agent SDK LLM using `query()` — stateless, one-shot interactions.

    Each call to `sample()` creates a fresh session with no memory of previous
    interactions. Best for independent rollout steps where conversation context
    is managed externally via message lists.

    Example usage:
    ```
    llm = ClaudeQueryLLM(model="claude-sonnet-4-6")
    responses = llm.sample(
        system_prompt="You are a helpful agent.",
        messages=[{"role": "user", "content": "Hello"}],
        tools=[{"type": "function", "name": "search", ...}],
    )
    actions = llm.get_actions(responses[0])
    ```
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        generation_config: Any | None = None,
        max_turns: int | None = 1,
        permission_mode: str = "default",  #  "bypassPermissions",
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, generation_config=generation_config, **kwargs)
        self.max_turns = max_turns
        self.permission_mode = permission_mode
        self.last_usage = None

    async def _query_single(
        self,
        system_prompt: str | None,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        max_new_tokens: int | None = None,
        **generation_kwargs: Any,
    ) -> ClaudeAgentResponse | None:
        """Run a single query() call and collect the response."""
        _clear_nested_session_guard()
        prompt = _messages_to_prompt(system_prompt, messages)

        mcp_servers, allowed_tools = _convert_tools_to_mcp_server(tools or [])

        options_kwargs: dict[str, Any] = {
            "model": self.model,
            "system_prompt": system_prompt,
            "max_turns": self.max_turns,
            "permission_mode": self.permission_mode,
            "mcp_servers": mcp_servers,
            "allowed_tools": allowed_tools,
        }
        # When env tools are provided, restrict Claude to only those tools
        # (otherwise Claude Code's built-in tools like ToolSearch are also available)
        if allowed_tools:
            options_kwargs["tools"] = allowed_tools
        options = ClaudeAgentOptions(**options_kwargs)

        response = ClaudeAgentResponse()

        try:
            async for message in query(prompt=prompt, options=options):
                if isinstance(message, AssistantMessage):
                    response.assistant_messages.append(message)
                elif isinstance(message, ResultMessage):
                    response.result = message
                    if message.usage:
                        response.usage = {
                            "input_tokens": message.usage.get("input_tokens", 0),
                            "output_tokens": message.usage.get("output_tokens", 0),
                        }
        except Exception as e:
            print(f"ClaudeQueryLLM query error: {type(e).__name__}: {e}")
            return None

        if response.usage:
            _track_usage(self, response.usage)

        return response

    def _sample_single(
        self,
        system_prompt: str | None,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        max_new_tokens: int | None = None,
        **generation_kwargs: Any,
    ) -> ClaudeAgentResponse | None:
        """Synchronous wrapper around async _query_single."""
        return asyncio.run(
            self._query_single(
                system_prompt=system_prompt,
                messages=messages,
                tools=tools,
                max_new_tokens=max_new_tokens,
                **generation_kwargs,
            )
        )

    def sample(
        self,
        system_prompt: str | None,
        messages: list[dict[str, Any]] | None = None,
        tools: list[dict[str, Any]] | None = None,
        max_new_tokens: int | None = None,
        num_return_sequences: int = 1,
        **generation_kwargs: Any,
    ) -> list[ClaudeAgentResponse | None]:
        """Generate response(s) from Claude via query()."""
        return [
            self._sample_single(
                system_prompt=system_prompt,
                messages=messages or [],
                tools=tools or [],
                max_new_tokens=max_new_tokens,
                **generation_kwargs,
            )
            for _ in range(num_return_sequences)
        ]

    async def sample_async(
        self,
        system_prompt: str | None,
        messages: list[dict[str, Any]] | None = None,
        tools: list[dict[str, Any]] | None = None,
        max_new_tokens: int | None = None,
        num_return_sequences: int = 1,
        **generation_kwargs: Any,
    ) -> list[ClaudeAgentResponse | None]:
        """Async wrapper around `sample` (offloaded to a worker thread).

        The Agent SDK `query()` path is sync, so async callers (e.g. an env's
        `_step_impl_async` invoking an LLM-judge grader's `sample_async`) run it
        in a thread to avoid blocking the event loop.
        """
        return await asyncio.to_thread(
            self.sample,
            system_prompt,
            messages,
            tools,
            max_new_tokens,
            num_return_sequences,
            **generation_kwargs,
        )

    def update_messages(
        self,
        messages: list[dict[str, Any]],
        model_response: ClaudeAgentResponse | None,
        prior_messages: list[dict[str, Any]],
        interleave: bool = False,
    ) -> list[dict[str, Any]]:
        """Return updated messages for the model."""
        return _update_messages_impl(messages, model_response, prior_messages, interleave)

    def get_actions(self, response: ClaudeAgentResponse | None) -> list[ActionFromLLM]:
        """Extract actions from a ClaudeAgentResponse."""
        return _extract_actions(response)


class ClaudeClientLLM(LLM):
    """
    Claude Agent SDK LLM using `ClaudeSDKClient` — persistent, multi-turn sessions.

    Maintains a conversation session across multiple `sample()` calls. The client
    remembers previous context, making it suitable for multi-step agent interactions
    where you want Claude to maintain state.

    Uses a single-task worker pattern on a background event loop thread so that all
    async SDK operations (connect, query, disconnect) run in the same asyncio Task.
    This is required because `ClaudeSDKClient` uses anyio cancel scopes internally,
    which are task-local and cannot cross task boundaries.

    Example usage:
    ```
    llm = ClaudeClientLLM(model="claude-sonnet-4-6")
    llm.connect()  # Start session

    responses = llm.sample(
        system_prompt="You are a helpful agent.",
        messages=[{"role": "user", "content": "Hello"}],
        tools=[{"type": "function", "name": "search", ...}],
    )
    actions = llm.get_actions(responses[0])

    # Later calls reuse the same session
    responses = llm.sample(
        system_prompt="You are a helpful agent.",
        messages=[{"role": "user", "content": "Follow up"}],
    )

    llm.disconnect()  # End session
    ```
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        generation_config: Any | None = None,
        max_turns: int | None = 1,
        permission_mode: str = "bypassPermissions",
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, generation_config=generation_config, **kwargs)
        self.max_turns = max_turns
        self.permission_mode = permission_mode
        self.last_usage = None
        self._client: ClaudeSDKClient | None = None
        self._connected = False
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_thread: threading.Thread | None = None
        self._cmd_queue: asyncio.Queue | None = None
        self._result_queue: queue.Queue = queue.Queue()

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        """Start a background event loop with a single worker task.

        All SDK operations are dispatched to the worker task via an asyncio.Queue,
        ensuring they all run in the same task context (required by anyio).
        """
        if self._loop is not None and not self._loop.is_closed():
            return self._loop

        self._loop = asyncio.new_event_loop()
        ready = threading.Event()

        def _run() -> None:
            asyncio.set_event_loop(self._loop)
            self._cmd_queue = asyncio.Queue()
            self._loop.create_task(self._worker())
            ready.set()
            self._loop.run_forever()

        self._loop_thread = threading.Thread(target=_run, daemon=True)
        self._loop_thread.start()
        ready.wait(timeout=10)
        return self._loop

    async def _worker(self) -> None:
        """Single persistent task that processes all SDK commands sequentially."""
        while True:
            cmd_name, args, kwargs = await self._cmd_queue.get()
            if cmd_name == "__stop__":
                self._result_queue.put((True, None))
                break
            try:
                method = getattr(self, f"_do_{cmd_name}")
                result = await method(*args, **kwargs)
                self._result_queue.put((True, result))
            except Exception as e:
                self._result_queue.put((False, e))

    def _send_cmd(
        self,
        cmd_name: str,
        *args: Any,
        timeout: float = _ASYNC_TIMEOUT,
        **kwargs: Any,
    ) -> Any:
        """Send a command to the worker task and wait for the result."""
        loop = self._ensure_loop()
        loop.call_soon_threadsafe(self._cmd_queue.put_nowait, (cmd_name, args, kwargs))
        ok, result = self._result_queue.get(timeout=timeout)
        if not ok:
            raise result
        return result

    def _get_options(
        self,
        system_prompt: str | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> ClaudeAgentOptions:
        """Build ClaudeAgentOptions from current config."""
        mcp_servers, allowed_tools = _convert_tools_to_mcp_server(tools or [])
        options_kwargs: dict[str, Any] = {
            "model": self.model,
            "system_prompt": system_prompt,
            "max_turns": self.max_turns,
            "permission_mode": self.permission_mode,
            "mcp_servers": mcp_servers,
            "allowed_tools": allowed_tools,
        }
        if allowed_tools:
            options_kwargs["tools"] = allowed_tools
        return ClaudeAgentOptions(**options_kwargs)

    # -- Worker command handlers (run inside the single worker task) --

    async def _do_connect(
        self,
        system_prompt: str | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> None:
        _clear_nested_session_guard()
        options = self._get_options(system_prompt, tools)
        self._client = ClaudeSDKClient(options=options)
        await self._client.connect()
        self._connected = True

    async def _do_disconnect(self) -> None:
        if self._client is not None:
            await self._client.disconnect()
            self._client = None
            self._connected = False

    async def _do_query(self, prompt: str) -> ClaudeAgentResponse | None:
        if self._client is None:
            raise RuntimeError("Client not connected. Call connect() first.")

        await self._client.query(prompt)

        response = ClaudeAgentResponse()
        try:
            async for message in self._client.receive_response():
                if isinstance(message, AssistantMessage):
                    response.assistant_messages.append(message)
                elif isinstance(message, ResultMessage):
                    response.result = message
                    if message.usage:
                        response.usage = {
                            "input_tokens": message.usage.get("input_tokens", 0),
                            "output_tokens": message.usage.get("output_tokens", 0),
                        }
        except Exception as e:
            print(f"ClaudeClientLLM query error: {type(e).__name__}: {e}")
            return None

        if response.usage:
            _track_usage(self, response.usage)

        return response

    # -- Public API --

    def connect(
        self,
        system_prompt: str | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> None:
        """Connect the client and start a session."""
        self._send_cmd("connect", system_prompt, tools)

    def disconnect(self) -> None:
        """Disconnect the client and end the session."""
        self._send_cmd("disconnect")
        # Stop the worker and shut down the event loop
        self._send_cmd("__stop__")
        if self._loop is not None and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
            if self._loop_thread is not None:
                self._loop_thread.join(timeout=5)
            self._loop = None
            self._loop_thread = None

    def _sample_single(
        self,
        system_prompt: str | None,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        max_new_tokens: int | None = None,
        **generation_kwargs: Any,
    ) -> ClaudeAgentResponse | None:
        """Synchronous wrapper around async query."""
        if not self._connected:
            self.connect(system_prompt=system_prompt, tools=tools)

        prompt = _messages_to_prompt(None, messages)  # system_prompt set at connect
        return self._send_cmd("query", prompt)

    def sample(
        self,
        system_prompt: str | None,
        messages: list[dict[str, Any]] | None = None,
        tools: list[dict[str, Any]] | None = None,
        max_new_tokens: int | None = None,
        num_return_sequences: int = 1,
        **generation_kwargs: Any,
    ) -> list[ClaudeAgentResponse | None]:
        """Generate response(s) from Claude via ClaudeSDKClient."""
        return [
            self._sample_single(
                system_prompt=system_prompt,
                messages=messages or [],
                tools=tools or [],
                max_new_tokens=max_new_tokens,
                **generation_kwargs,
            )
            for _ in range(num_return_sequences)
        ]

    async def sample_async(
        self,
        system_prompt: str | None,
        messages: list[dict[str, Any]] | None = None,
        tools: list[dict[str, Any]] | None = None,
        max_new_tokens: int | None = None,
        num_return_sequences: int = 1,
        **generation_kwargs: Any,
    ) -> list[ClaudeAgentResponse | None]:
        """Async wrapper around `sample` (offloaded to a worker thread).

        See ClaudeQueryLLM.sample_async — the underlying ClaudeSDKClient path is
        sync, so async callers run it in a thread to avoid blocking the loop.
        """
        return await asyncio.to_thread(
            self.sample,
            system_prompt,
            messages,
            tools,
            max_new_tokens,
            num_return_sequences,
            **generation_kwargs,
        )

    def update_messages(
        self,
        messages: list[dict[str, Any]],
        model_response: ClaudeAgentResponse | None,
        prior_messages: list[dict[str, Any]],
        interleave: bool = False,
    ) -> list[dict[str, Any]]:
        """Return updated messages for the model."""
        return _update_messages_impl(messages, model_response, prior_messages, interleave)

    def get_actions(self, response: ClaudeAgentResponse | None) -> list[ActionFromLLM]:
        """Extract actions from a ClaudeAgentResponse."""
        return _extract_actions(response)

    def __del__(self) -> None:
        """Cleanup on garbage collection."""
        if self._connected:
            try:
                self.disconnect()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _is_tool_use_block(block: Any) -> bool:
    """Check if block is a ToolUseBlock (by structure, not isinstance)."""
    return hasattr(block, "name") and hasattr(block, "input") and hasattr(block, "id")


def _is_text_block(block: Any) -> bool:
    """Check if block is a TextBlock (by structure, not isinstance)."""
    return hasattr(block, "text") and not hasattr(block, "thinking")


def _is_thinking_block(block: Any) -> bool:
    """Check if block is a ThinkingBlock (by structure, not isinstance)."""
    return hasattr(block, "thinking")


def _is_tool_result_block(block: Any) -> bool:
    """Check if block is a ToolResultBlock (by structure, not isinstance)."""
    return hasattr(block, "tool_use_id") and not hasattr(block, "name")


def _strip_mcp_prefix(name: str, prefix: str = "mcp__env_tools__") -> str:
    """Strip MCP server prefix from tool names (e.g. 'mcp__env_tools__get_weather' -> 'get_weather')."""
    if name.startswith(prefix):
        return name[len(prefix) :]
    return name


def _extract_actions(response: ClaudeAgentResponse | None) -> list[ActionFromLLM]:
    """
    Extract ActionFromLLM items from a ClaudeAgentResponse.

    Maps Claude Agent SDK content blocks to ActionFromLLM:
    - ToolUseBlock -> type="function_call"
    - TextBlock -> type="message"
    - ThinkingBlock -> type="reasoning"
    """
    actions = []
    if response is None:
        return actions

    for block in response.output:
        if _is_tool_use_block(block):
            tool_name = _strip_mcp_prefix(block.name)
            text_repr = json.dumps({"name": tool_name, "arguments": block.input})
            text_repr = f"<tool_call>\n{text_repr}\n</tool_call>"
            actions.append(
                ActionFromLLM(
                    role="assistant",
                    type="function_call",
                    text=text_repr,
                    call_id=block.id,
                    name=tool_name,
                    arguments=block.input,
                )
            )
        elif _is_thinking_block(block):
            actions.append(
                ActionFromLLM(
                    role="assistant",
                    type="reasoning",
                    text=block.thinking,
                    call_id=None,
                    name=None,
                    arguments=None,
                )
            )
        elif _is_text_block(block):
            actions.append(
                ActionFromLLM(
                    role="assistant",
                    type="message",
                    text=block.text,
                    call_id=None,
                    name=None,
                    arguments=None,
                )
            )
        elif _is_tool_result_block(block):
            # Tool results are environment responses, not model actions.
            pass
        else:
            print(f"Warning: unknown content block type: {type(block)}")

    return actions


def _response_to_message_dicts(
    response: ClaudeAgentResponse,
) -> list[dict[str, Any]]:
    """
    Convert a ClaudeAgentResponse's content blocks into message dicts
    suitable for inclusion in a messages list.
    """
    message_dicts = []
    for block in response.output:
        if _is_tool_use_block(block):
            message_dicts.append(
                {
                    "role": "assistant",
                    "type": "function_call",
                    "call_id": block.id,
                    "name": _strip_mcp_prefix(block.name),
                    "arguments": json.dumps(block.input),
                }
            )
        elif _is_thinking_block(block):
            message_dicts.append(
                {
                    "role": "assistant",
                    "type": "reasoning",
                    "content": block.thinking,
                }
            )
        elif _is_text_block(block):
            message_dicts.append(
                {
                    "role": "assistant",
                    "content": block.text,
                }
            )
    return message_dicts
