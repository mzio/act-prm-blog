"""
Claude Agent SDK generators — drive env rollouts with a Claude teacher.

Used to collect FULL thought+action expert trajectories ourselves, as an
alternative to the logged GPT-5-mini corpora (whose thoughts are present on only
50-64% of turns). The Claude Agent SDK emits ThinkingBlocks alongside ToolUseBlocks,
so both halves of each step are captured at the source rather than inferred.
"""

from .base import BaseClaudeAgentSDKGenerator
from .query import QueryClaudeAgentSDKGenerator
from .teacher import ClaudeTeacherGenerator

__all__ = [
    "BaseClaudeAgentSDKGenerator",
    "QueryClaudeAgentSDKGenerator",
    "ClaudeTeacherGenerator",
]
