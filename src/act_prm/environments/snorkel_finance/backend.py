"""
Data backend for Snorkel Agent Finance Reasoning.

Loads raw financial table data (JSON) and supports real SQL queries
via in-memory SQLite, matching the reference implementation at
https://github.com/snorkel-ai/FinQABenchmark/blob/main/src/tools.py
"""

from __future__ import annotations

import glob
import json
import logging
import os
import re
import sqlite3
from io import StringIO
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


# SQL filters that indicate a valid filtered query (from reference tools.py)
SQL_FILTERS = [
    "WHERE",
    "HAVING",
    "IN",
    "NOT IN",
    "EXISTS",
    "NOT EXISTS",
    "ANY",
    "SOME",
    "ALL",
    "LIKE",
    "NOT LIKE",
    "BETWEEN",
    "NOT BETWEEN",
    "IS NULL",
    "IS NOT NULL",
    "CASE",
    "FILTER",
]


class DataBackend:
    """
    Backend that serves real financial data from local JSON files
    and executes SQL queries via in-memory SQLite.

    Expects the data directory layout from FinQABenchmark:
      data_path/
        tables_cleaned_all_companies.json
        <company_name>/
          <table_name>.json
          ...
    """

    def __init__(self, data_path: str) -> None:
        self.data_path = data_path
        if not os.path.isdir(data_path):
            raise FileNotFoundError(
                f"Data directory not found: {data_path}. "
                f"Clone https://github.com/snorkel-ai/FinQABenchmark "
                f"and point data_path to its data/raw/ directory."
            )

        # Load the cleaned tables metadata (used by get_table_info)
        tables_json_path = os.path.join(data_path, "tables_cleaned_all_companies.json")
        if os.path.isfile(tables_json_path):
            with open(tables_json_path) as f:
                self.tables_cleaned: dict[str, Any] = json.load(f)
        else:
            self.tables_cleaned = {}
            logger.warning("tables_cleaned_all_companies.json not found at %s", tables_json_path)

        self._companies = sorted(
            d
            for d in os.listdir(data_path)
            if os.path.isdir(os.path.join(data_path, d)) and not d.startswith(".") and d != "raw"
        )
        logger.info(
            "DataBackend: %d companies, %d cleaned table entries at %s",
            len(self._companies),
            len(self.tables_cleaned),
            data_path,
        )

    def get_descriptions(self, company_name: str) -> str:
        """List available tables for a company (JSON files in its directory)."""
        company_dir = os.path.join(self.data_path, company_name)
        if not os.path.isdir(company_dir):
            return f"Company '{company_name}' not found. Available companies: {self._companies}"
        table_paths = glob.glob(os.path.join(company_dir, "*.json"))
        table_names = [os.path.basename(f).replace(".json", "") for f in table_paths]
        return json.dumps(table_names)

    def get_table_info(self, company_name: str, table_name: str) -> str:
        """Get metadata about a table: columns, dtypes, unique values."""
        cleaned_table_name = table_name.replace(".json", "").replace(".txt", "")
        key = f"{company_name}/{cleaned_table_name}"

        if key not in self.tables_cleaned:
            return (
                f"No table info found for '{table_name}' in '{company_name}'. "
                f"Try calling get_descriptions first to see available tables."
            )

        cleaned_table_info = dict(self.tables_cleaned[key])
        table_df = pd.read_json(StringIO(cleaned_table_info["table"]))

        # Identify numeric columns to drop (only show structure columns)
        cols_to_drop = []
        for col in table_df.columns.tolist()[1:]:  # skip first column
            vals = table_df[col].tolist()[1:]
            cleaned_vals = ["".join(c for c in str(x) if c.isalnum()).strip() for x in vals]
            all_numeric = all(v.isnumeric() or len(v) == 0 for v in cleaned_vals)
            if all_numeric:
                cols_to_drop.append(col)

        cleaned_table_info["column_dtypes"] = {col: str(table_df[col].dtype) for col in table_df.columns.tolist()}
        table_df = table_df.drop(cols_to_drop, axis=1)
        del cleaned_table_info["table"]
        cleaned_table_info["unique_vals_per_col"] = {
            col: list(table_df[col].unique()) for col in table_df.columns.tolist()
        }
        return json.dumps(cleaned_table_info, indent=0).replace("\n", "")

    def sql_query(self, company_name: str, table_name: str, query: str) -> str:
        """Execute a SQL query on a financial table via in-memory SQLite."""
        # Block SELECT *
        if "select *" in query.lower():
            return "Error: SELECT * is not allowed, highly inefficient!"

        # Check for required filter clauses
        query_cleaned = re.sub(r"(\\r|\\n|\\t|[\r\n\t])+", " ", query).upper()
        pattern = r"(?<!\w|\[)(" + "|".join(re.escape(f) for f in SQL_FILTERS) + r")(?!\w|\])"

        has_filter = len(re.findall(pattern, query_cleaned)) > 0

        if not has_filter:
            return "Error: You are trying to query without any kind of filters, which is not allowed!"

        # Only allow SELECT statements
        stmt = query.strip().upper()
        if not stmt.startswith("SELECT"):
            return "Error: Only SELECT queries are allowed."

        cleaned_table_name = table_name.replace(".txt", "").replace(".json", "")
        table_path = os.path.join(self.data_path, company_name, f"{cleaned_table_name}.json")
        if not os.path.isfile(table_path):
            return f"Error: Table '{table_name}' not found for company '{company_name}'."

        conn = sqlite3.connect(":memory:")
        try:
            df = pd.read_json(table_path)
            df.to_sql(cleaned_table_name, conn, index=False, if_exists="replace")
            result = pd.read_sql_query(query, conn)
            return result.to_json(orient="records")
        except Exception as e:
            return f"SQL error: {type(e).__name__}: {e}"
        finally:
            conn.close()

    def list_companies(self) -> list[str]:
        """List all available companies."""
        return list(self._companies)
