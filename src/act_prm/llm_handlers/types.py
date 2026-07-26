"""
Types for LLM handlers
"""

from typing import Any

from pydantic import BaseModel


class ActionFromLLM(BaseModel):
    """
    LLM action
    """

    role: str
    type: str
    text: str | None = None
    call_id: str | None = None
    name: str | None = None
    arguments: dict[str, Any] | None = None
