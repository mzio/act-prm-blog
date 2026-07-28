"""
Tools for Snorkel Agent Insurance Underwriting environment.

7 tools: get_underwriting_guidelines, get_table_descriptions,
get_table_data_dictionary, list_tables, get_table_schema, read_query,
respond_user

Tool implementations match the reference MCP server at
https://github.com/snorkel-ai/multi-turn-insurance-underwriting-benchmark-generation/blob/main/src/mcp_server.py
"""

from __future__ import annotations

import logging
from typing import Any

from ..base import BaseTool

logger = logging.getLogger(__name__)


class GetUnderwritingGuidelinesTool(BaseTool):
    """Return the plain-text underwriting guidelines."""

    def __call__(self, backend: Any, **kwargs: Any) -> str:
        return backend.get_underwriting_guidelines()

    def get_tool_desc(self) -> dict[str, Any]:
        return {
            "type": "function",
            "name": "get_underwriting_guidelines",
            "description": (
                "Get the underwriting guidelines for All National Insurance. "
                "Returns the complete set of rules for appetite determination, "
                "policy limits, deductibles, and special conditions."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        }


class GetTableDescriptionsTool(BaseTool):
    """Return descriptions of all database tables."""

    def __call__(self, backend: Any, **kwargs: Any) -> str:
        return backend.get_table_descriptions()

    def get_tool_desc(self) -> dict[str, Any]:
        return {
            "type": "function",
            "name": "get_table_descriptions",
            "description": (
                "Get descriptions of all tables available in the insurance "
                "database. Use this to understand what data is available "
                "before querying."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        }


class GetTableDataDictionaryTool(BaseTool):
    """Return column-level metadata for a specific table."""

    def __call__(self, table: str, backend: Any, **kwargs: Any) -> str:
        return backend.get_table_data_dictionary(table)

    def get_tool_desc(self) -> dict[str, Any]:
        return {
            "type": "function",
            "name": "get_table_data_dictionary",
            "description": (
                "Get detailed column-level metadata for a specific table, "
                "including column names, types, and descriptions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "table": {
                        "type": "string",
                        "description": "Name of the table to get the data dictionary for",
                    },
                },
                "required": ["table"],
            },
        }


class ListTablesTool(BaseTool):
    """List all tables in the database."""

    def __call__(self, backend: Any, **kwargs: Any) -> str:
        return backend.list_tables()

    def get_tool_desc(self) -> dict[str, Any]:
        return {
            "type": "function",
            "name": "list_tables",
            "description": ("List all tables available in the insurance database."),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        }


class GetTableSchemaTool(BaseTool):
    """Return the schema (column info) for a table."""

    def __call__(self, table_name: str, backend: Any, **kwargs: Any) -> str:
        return backend.get_table_schema(table_name)

    def get_tool_desc(self) -> dict[str, Any]:
        return {
            "type": "function",
            "name": "get_table_schema",
            "description": ("Get the schema for a specific table, including column names, types, and constraints."),
            "parameters": {
                "type": "object",
                "properties": {
                    "table_name": {
                        "type": "string",
                        "description": "Name of the table to get the schema for",
                    },
                },
                "required": ["table_name"],
            },
        }


class ReadQueryTool(BaseTool):
    """Execute a read-only SQL query against the database."""

    def __call__(self, query: str, backend: Any, **kwargs: Any) -> str:
        return backend.read_query(query)

    def get_tool_desc(self) -> dict[str, Any]:
        return {
            "type": "function",
            "name": "read_query",
            "description": (
                "Execute a read-only SQL SELECT query against the insurance "
                "database. Only SELECT statements are allowed. Results are "
                "returned as JSON. Maximum 500 rows returned."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": ("SQL SELECT query to execute against the database."),
                    },
                },
                "required": ["query"],
            },
        }


class RespondUserTool(BaseTool):
    """Provide final answer to the user."""

    def __call__(self, text: str, **kwargs: Any) -> str:
        # This is handled by the environment's step logic (triggers grading)
        return text

    def get_tool_desc(self) -> dict[str, Any]:
        return {
            "type": "function",
            "name": "respond_user",
            "description": (
                "Provide your final answer to the underwriter's question. "
                "Include a clear rationale based on the data and guidelines "
                "you found using the tools."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Your final answer text with rationale",
                    },
                },
                "required": ["text"],
            },
        }


TOOL_CLASSES = {
    "get_underwriting_guidelines": GetUnderwritingGuidelinesTool,
    "get_table_descriptions": GetTableDescriptionsTool,
    "get_table_data_dictionary": GetTableDataDictionaryTool,
    "list_tables": ListTablesTool,
    "get_table_schema": GetTableSchemaTool,
    "read_query": ReadQueryTool,
    "respond_user": RespondUserTool,
}
