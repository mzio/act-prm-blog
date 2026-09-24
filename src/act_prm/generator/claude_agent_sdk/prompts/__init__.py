"""
Claude Agent SDK Generator Prompts

The canonical prompt templates for this repo's Claude Agent SDK generators.
Upstream `strl` keeps its copy under `show_dont_tell/prompts`; this one is
intentionally generic — a `<tool_name>` placeholder rather than a hardcoded
browser example, and no `[ref=eN]` snapshot guidance — since the Act-PRM
generators are not browser-specific.

Only the chat system prompt is ported. The judge / reflection / summary
templates stay upstream with the show-don't-tell generators that use them.
"""

from .default import get_system_prompt_template as get_chat_system_prompt_template

__all__ = [
    "get_chat_system_prompt_template",
]
