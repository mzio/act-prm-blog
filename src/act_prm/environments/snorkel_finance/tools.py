"""
Tools for Snorkel Agent Finance Reasoning environment.

5 tools: get_descriptions, get_table_info, sql_query, calculator, respond_user

Tool implementations match the reference at
https://github.com/snorkel-ai/FinQABenchmark/blob/main/src/tools.py
"""

from __future__ import annotations

import json
import logging
from typing import Any

from ..base import BaseTool

logger = logging.getLogger(__name__)


class GetDescriptionsTool(BaseTool):
    """Get list of available financial tables for a company."""

    def __call__(self, company_name: str, backend: Any, **kwargs: Any) -> str:
        return backend.get_descriptions(company_name)

    def get_tool_desc(self) -> dict[str, Any]:
        return {
            "type": "function",
            "name": "get_descriptions",
            "description": (
                "Get a list of possible texts to look up for. Each text "
                "contains a singular table that is described by its complete name."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "company_name": {
                        "type": "string",
                        "description": ("The name of the company to get the table names for"),
                    },
                },
                "required": ["company_name"],
            },
        }


class GetTableInfoTool(BaseTool):
    """Get metadata about a specific financial table."""

    def __call__(
        self,
        company_name: str,
        table_name: str,
        backend: Any,
        **kwargs: Any,
    ) -> str:
        return backend.get_table_info(company_name, table_name)

    def get_tool_desc(self) -> dict[str, Any]:
        return {
            "type": "function",
            "name": "get_table_info",
            "description": (
                "Get table associated with table_name. Returns metadata "
                "including column names, data types, and unique values per "
                "column for query columns."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "company_name": {
                        "type": "string",
                        "description": "The name of the company to query",
                    },
                    "table_name": {
                        "type": "string",
                        "description": "The name of the table to retrieve",
                    },
                },
                "required": ["company_name", "table_name"],
            },
        }


class SqlQueryTool(BaseTool):
    """Execute SQL queries on financial tables."""

    def __call__(
        self,
        company_name: str,
        table_name: str,
        query: str,
        backend: Any,
        **kwargs: Any,
    ) -> str:
        return backend.sql_query(company_name, table_name, query)

    def get_tool_desc(self) -> dict[str, Any]:
        return {
            "type": "function",
            "name": "sql_query",
            "description": (
                "Given a table name and a SQL query, use SQLite to process "
                "the query over the table and return the result. "
                "Provide queries in SQLite compatible format. "
                "If the query is for the whole table/whole columns without "
                "filters, the query is too inefficient and you will get an error."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "company_name": {
                        "type": "string",
                        "description": "Name of the company provided by the user",
                    },
                    "table_name": {
                        "type": "string",
                        "description": "Name of the table to query",
                    },
                    "query": {
                        "type": "string",
                        "description": ("SQL query to execute on the table. No SELECT * allowed."),
                    },
                },
                "required": ["company_name", "table_name", "query"],
            },
        }


class CalculatorTool(BaseTool):
    """Execute Python math expressions."""

    def __call__(self, expression: str, **kwargs: Any) -> str:
        try:
            # Safe eval: only allow math operations
            allowed_names = {"__builtins__": {}}
            import math

            allowed_names.update({k: v for k, v in vars(math).items() if not k.startswith("_")})
            result = eval(expression, allowed_names)  # noqa: S307
            return json.dumps([result, f"Calculated: {expression} = {result}"])
        except Exception as e:
            return f"Calculator error: {type(e).__name__}: {e}"

    def get_tool_desc(self) -> dict[str, Any]:
        return {
            "type": "function",
            "name": "calculator",
            "description": (
                "Given a equation/mathematical expression, evaluate it "
                "and return the result. Supports standard math operations "
                "(+, -, *, /, **, %) and math module functions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": ("A Python math expression (e.g., '(310 / 6759) * 100')"),
                    },
                },
                "required": ["expression"],
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
                "Provide your final answer to the user's question. "
                "Generate a concise paragraph summarizing your findings "
                "with all relevant figures."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Your final answer text",
                    },
                },
                "required": ["text"],
            },
        }


TOOL_CLASSES = {
    "get_descriptions": GetDescriptionsTool,
    "get_table_info": GetTableInfoTool,
    "sql_query": SqlQueryTool,
    "calculator": CalculatorTool,
    "respond_user": RespondUserTool,
}
