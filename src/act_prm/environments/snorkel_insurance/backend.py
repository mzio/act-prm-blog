"""
Data backend for Snorkel Agent Insurance Underwriting.

Loads resource data (parquet files) into an in-memory SQLite database
and serves underwriting guidelines, matching the reference implementation at
https://github.com/snorkel-ai/multi-turn-insurance-underwriting-benchmark-generation
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


class DataBackend:
    """
    Backend that serves insurance underwriting data from a SQLite database
    built from parquet resource files, plus plain-text underwriting guidelines.

    Expects the data directory layout:
      data_path/
        resources/
          naics_table_2022.parquet
          naics_2digit_table_2022.parquet
          naics_2022_2017_conversion.parquet
          naics_2012_2017_conversion.parquet
          sba_size_standards_by_2012_naics.parquet
          small_business_insurance_appetite.parquet
          small_business_lobs.parquet
          states.parquet
        tool_data/
          underwriting_rules.txt
          table_dictionary.json
          table_data_dictionaries.json
          building_construction_types.json
    """

    def __init__(self, data_path: str) -> None:
        self.data_path = data_path
        if not os.path.isdir(data_path):
            raise FileNotFoundError(
                f"Data directory not found: {data_path}. "
                f"Clone https://github.com/snorkel-ai/multi-turn-insurance-underwriting-benchmark-generation "
                f"and point data_path to its root directory."
            )

        # Load underwriting guidelines
        rules_path = os.path.join(data_path, "tool_data", "underwriting_rules.txt")
        if os.path.isfile(rules_path):
            with open(rules_path) as f:
                self.underwriting_guidelines: str = f.read()
        else:
            self.underwriting_guidelines = ""
            logger.warning("underwriting_rules.txt not found at %s", rules_path)

        # Load table dictionary (high-level table descriptions)
        table_dict_path = os.path.join(data_path, "tool_data", "table_dictionary.json")
        if os.path.isfile(table_dict_path):
            with open(table_dict_path) as f:
                self.table_dictionary: dict[str, str] = json.load(f)
        else:
            self.table_dictionary = {}
            logger.warning("table_dictionary.json not found at %s", table_dict_path)

        # Load per-table data dictionaries (column-level metadata)
        data_dict_path = os.path.join(data_path, "tool_data", "table_data_dictionaries.json")
        if os.path.isfile(data_dict_path):
            with open(data_dict_path) as f:
                self.table_data_dictionaries: dict[str, Any] = json.load(f)
        else:
            self.table_data_dictionaries = {}
            logger.warning("table_data_dictionaries.json not found at %s", data_dict_path)

        # Build SQLite database from parquet files
        self.conn = self._build_database()

        logger.info(
            "DataBackend: loaded %d tables from %s",
            len(self.table_dictionary),
            data_path,
        )

    def _build_database(self) -> sqlite3.Connection:
        """Load all parquet files from resources/ into an in-memory SQLite DB."""
        conn = sqlite3.connect(":memory:")
        resources_dir = os.path.join(self.data_path, "resources")

        if not os.path.isdir(resources_dir):
            logger.warning("resources/ directory not found at %s", resources_dir)
            return conn

        for filename in sorted(os.listdir(resources_dir)):
            if not filename.endswith(".parquet"):
                continue
            table_name = filename.replace(".parquet", "")
            filepath = os.path.join(resources_dir, filename)
            try:
                df = pd.read_parquet(filepath)
                df.to_sql(table_name, conn, index=False, if_exists="replace")
                logger.debug("Loaded table '%s' (%d rows)", table_name, len(df))
            except Exception as e:
                logger.warning("Failed to load %s: %s: %s", filename, type(e).__name__, e)

        return conn

    def get_underwriting_guidelines(self) -> str:
        """Return the plain-text underwriting guidelines."""
        return self.underwriting_guidelines

    def get_table_descriptions(self) -> str:
        """Return JSON descriptions of all database tables."""
        return json.dumps(self.table_dictionary, indent=2)

    def get_table_data_dictionary(self, table: str) -> str:
        """Return column-level metadata for a specific table."""
        if table in self.table_data_dictionaries:
            return json.dumps(self.table_data_dictionaries[table], indent=2)
        return (
            f"No data dictionary found for table '{table}'. "
            f"Available tables: {list(self.table_data_dictionaries.keys())}"
        )

    def list_tables(self) -> str:
        """List all tables in the SQLite database."""
        cursor = self.conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        tables = [row[0] for row in cursor.fetchall()]
        return json.dumps(tables)

    def get_table_schema(self, table_name: str) -> str:
        """Return PRAGMA table_info for a table."""
        # Validate table name against known tables to prevent SQL injection
        import json as _json

        known_tables = _json.loads(self.list_tables())
        if table_name not in known_tables:
            return f"Table '{table_name}' not found. Available tables: {known_tables}"
        try:
            cursor = self.conn.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()
            if not columns:
                return f"Table '{table_name}' not found or has no columns."
            schema = [
                {
                    "cid": col[0],
                    "name": col[1],
                    "type": col[2],
                    "notnull": col[3],
                    "default_value": col[4],
                    "pk": col[5],
                }
                for col in columns
            ]
            return json.dumps(schema, indent=2)
        except Exception as e:
            return f"Error getting schema for '{table_name}': {type(e).__name__}: {e}"

    def read_query(self, query: str) -> str:
        """Execute a read-only SQL SELECT query against the database."""
        # Safety checks
        query_stripped = query.strip()
        query_upper = query_stripped.upper()

        if not query_upper.startswith("SELECT"):
            return "Error: Only SELECT queries are allowed."

        # Block write operations
        write_keywords = ["INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE"]
        for kw in write_keywords:
            if kw in query_upper:
                return f"Error: {kw} operations are not allowed. Read-only queries only."

        # Block multiple statements
        if ";" in query_stripped[:-1]:  # allow trailing semicolon
            return "Error: Multiple SQL statements are not allowed."

        try:
            result = pd.read_sql_query(query_stripped, self.conn)
            # Limit to 500 rows
            if len(result) > 500:
                result = result.head(500)
                return result.to_json(orient="records") + "\n\n[WARNING: Results truncated to 500 rows]"
            return result.to_json(orient="records")
        except Exception as e:
            return f"SQL error: {type(e).__name__}: {e}"

    def close(self) -> None:
        """Close the SQLite connection."""
        self.conn.close()
