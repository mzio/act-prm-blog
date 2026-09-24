"""
Types for Claude Agent SDK Generators
"""

from dataclasses import dataclass, field
from typing import Any

from claude_agent_sdk import AssistantMessage, ResultMessage


@dataclass
class ClaudeAgentResponse:
    """
    Response container that mirrors the structure expected by get_actions().
    Collects AssistantMessages from the Claude Agent SDK stream.
    """

    assistant_messages: list[AssistantMessage] = field(default_factory=list)
    result: ResultMessage | None = None
    usage: dict[str, int] | None = None
    cost: float | None = 0.0

    @property
    def output(self) -> list[Any]:
        """Flatten all content blocks from all assistant messages."""
        blocks = []
        for msg in self.assistant_messages:
            blocks.extend(msg.content)
        return blocks


# ---------------------------------------------------------------------------
# Helpers for determining block types
# ---------------------------------------------------------------------------


def is_tool_use_block(block: Any) -> bool:
    """Check if block is a ToolUseBlock (by structure, not isinstance)."""
    return hasattr(block, "name") and hasattr(block, "input") and hasattr(block, "id")


def is_text_block(block: Any) -> bool:
    """Check if block is a TextBlock (by structure, not isinstance)."""
    return hasattr(block, "text") and not hasattr(block, "thinking")


def is_thinking_block(block: Any) -> bool:
    """Check if block is a ThinkingBlock (by structure, not isinstance)."""
    return hasattr(block, "thinking")


def is_tool_result_block(block: Any) -> bool:
    """Check if block is a ToolResultBlock (by structure, not isinstance)."""
    return hasattr(block, "tool_use_id") and not hasattr(block, "name")
