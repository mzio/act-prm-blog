"""
Data loading + action extraction for Act-PRM logged trajectories.

Ported from ``scripts/act_prm_length_penalty.py`` (the Tinker training script) so
the PyTorch path consumes the exact same action-only traces.
"""

import json
import re
from pathlib import Path
from typing import Any

DATASET = "mzio/aprm-snorkelai_agent_finance_reasoning"


def extract_action(content: str) -> str | None:
    """Cut the narration out of an assistant message, keeping only the explicit
    action: the ``<tool_call>...</tool_call>`` block (or a ``Final Answer:``
    suffix). Returns None if neither is present."""
    m = re.search(r"<tool_call>.*?</tool_call>", content, flags=re.DOTALL)
    if m:
        return m.group(0).strip()
    if "Final Answer:" in content:
        return ("Final Answer:" + content.split("Final Answer:", 1)[1]).strip()
    return None


def _traj_from_row(row: dict[str, Any]) -> dict[str, Any] | None:
    """Convert one dataset row (state + action, with narration) into an
    action-only trajectory dict, or None if any action is unparseable / the
    trajectory is too short (< 2 actions)."""
    messages = list(row["state"]) + [row["action"]]
    traj: list[dict[str, str]] = []
    for msg in messages:
        if msg["role"] == "assistant":
            action = extract_action(msg["content"])
            if action is None:  # unparseable action -> drop trajectory
                return None
            traj.append({"role": "assistant", "content": action})
        else:
            traj.append({"role": msg["role"], "content": msg["content"]})
    if sum(m["role"] == "assistant" for m in traj) < 2:
        return None
    return {
        "messages": traj,
        "system_prompt": row.get("system_prompt") or "You are a helpful assistant.",
        "uid": row.get("unique_data_sample_id"),
    }


def load_trajectories(n: int, max_timestep: int, dataset: str = DATASET) -> list[dict[str, Any]]:
    """Stream the dataset; keep the first ``n`` successful (done, return_>0)
    rollouts that finished within ``max_timestep`` steps. Assistant turns are
    stripped to action-only content."""
    from datasets import load_dataset

    ds = load_dataset(dataset, split="train", streaming=True)
    out: list[dict[str, Any]] = []
    seen: set[Any] = set()
    for row in ds:
        if not (row["done"] and row["return_"] > 0):
            continue
        if row["max_timestep"] > max_timestep:
            continue
        uid = (row["unique_data_sample_id"], row["generation_id"])
        if uid in seen:
            continue
        seen.add(uid)
        traj = _traj_from_row(row)
        if traj is not None:
            out.append(traj)
        if len(out) >= n:
            break
    if len(out) < n:
        raise RuntimeError(f"only found {len(out)}/{n} usable trajectories in {dataset}")
    return out


def load_trajectories_split(
    split_file: str, dataset: str = DATASET
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load ALL successful trajectories, partitioned by a canonical task split
    (a JSON with ``train_uids`` / ``eval_uids``). Returns (train_pool, eval_pool),
    each sorted by task uid for reproducibility."""
    from datasets import load_dataset

    split = json.loads(Path(split_file).read_text())
    train_uids, eval_uids = set(split["train_uids"]), set(split["eval_uids"])
    want = train_uids | eval_uids
    ds = load_dataset(split.get("dataset", dataset), split="train", streaming=True)
    train_pool: list[dict[str, Any]] = []
    eval_pool: list[dict[str, Any]] = []
    seen: set[Any] = set()
    for row in ds:
        if not (row["done"] and row["return_"] > 0):
            continue
        uid = row["unique_data_sample_id"]
        if uid not in want or uid in seen:
            continue
        seen.add(uid)
        traj = _traj_from_row(row)
        if traj is None:
            continue
        (train_pool if uid in train_uids else eval_pool).append(traj)
    train_pool.sort(key=lambda t: str(t["uid"]))
    eval_pool.sort(key=lambda t: str(t["uid"]))
    return train_pool, eval_pool


def compact_observations(
    messages: list[dict[str, str]],
    obs_max_chars: int,
    first_to_show: int = 2,
    last_to_show: int = 1,
) -> list[dict[str, str]]:
    """Cap observation lengths and hide middle observations (the codebase's
    hide_observations trick) so long tool outputs don't blow up the context."""
    out = [dict(m) for m in messages]
    obs_idx = [i for i, m in enumerate(out) if m["role"] in ("user", "tool")]
    for j, i in enumerate(obs_idx):
        if j >= first_to_show and j < len(obs_idx) - last_to_show:
            out[i]["content"] = "..."
        elif len(out[i]["content"]) > obs_max_chars:
            out[i]["content"] = out[i]["content"][:obs_max_chars] + " ...[truncated]"
    return out


# ---------------------------------------------------------------------------
# Synthetic fallback — a couple of hand-built action-only traces so the training
# path can be smoke-tested offline (no HF Hub download / auth needed).
# ---------------------------------------------------------------------------
_SYN_SYSTEM = "You are a helpful assistant."

SYNTHETIC_TRAJECTORIES: list[dict[str, Any]] = [
    {
        "uid": "synthetic-0",
        "system_prompt": _SYN_SYSTEM,
        "messages": [
            {
                "role": "user",
                "content": (
                    "Here is the question: What was the company's total revenue growth in 2024? "
                    "The company to query in the database: acme"
                ),
            },
            {
                "role": "assistant",
                "content": '<tool_call>\n{"name": "get_descriptions", "arguments": {"company_name": "acme"}}\n</tool_call>',
            },
            {
                "role": "tool",
                "content": (
                    "Available tables for acme: acme_IncomeStatement, acme_BalanceSheet, "
                    "acme_RevenueBySegment."
                ),
            },
            {
                "role": "assistant",
                "content": '<tool_call>\n{"name": "query_table", "arguments": {"table": "acme_IncomeStatement", "field": "total_revenue"}}\n</tool_call>',
            },
            {
                "role": "tool",
                "content": "acme_IncomeStatement.total_revenue: 2023=100.0, 2024=118.0 (millions USD).",
            },
            {
                "role": "assistant",
                "content": "Final Answer: Total revenue grew 18% in 2024 (from $100.0M to $118.0M).",
            },
        ],
    },
    {
        "uid": "synthetic-1",
        "system_prompt": _SYN_SYSTEM,
        "messages": [
            {
                "role": "user",
                "content": (
                    "Here is the question: How much did operating expenses change year over year? "
                    "The company to query in the database: globex"
                ),
            },
            {
                "role": "assistant",
                "content": '<tool_call>\n{"name": "get_descriptions", "arguments": {"company_name": "globex"}}\n</tool_call>',
            },
            {
                "role": "tool",
                "content": "Available tables for globex: globex_IncomeStatement, globex_CashFlow.",
            },
            {
                "role": "assistant",
                "content": '<tool_call>\n{"name": "query_table", "arguments": {"table": "globex_IncomeStatement", "field": "operating_expenses"}}\n</tool_call>',
            },
            {
                "role": "tool",
                "content": "globex_IncomeStatement.operating_expenses: 2023=40.0, 2024=44.0 (millions USD).",
            },
            {
                "role": "assistant",
                "content": "Final Answer: Operating expenses rose 10% year over year (from $40.0M to $44.0M).",
            },
        ],
    },
]


def load_synthetic() -> list[dict[str, Any]]:
    """Return a deep-ish copy of the built-in synthetic action-only trajectories."""
    return [
        {
            "uid": t["uid"],
            "system_prompt": t["system_prompt"],
            "messages": [dict(m) for m in t["messages"]],
        }
        for t in SYNTHETIC_TRAJECTORIES
    ]
