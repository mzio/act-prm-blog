"""
Helper functions for STRL trainers
"""

from typing import Any

from rich.console import Console
from rich.table import Table

console = Console()


def is_better(x: float, y: float, metric: str) -> bool:
    """
    Determine if x is better than y for a given metric
    """
    return x <= y if metric in ["loss"] else x >= y


def display_metrics(
    metrics: dict[str, Any],
    title: str | None = None,
    style: str = "bright_yellow",
) -> None:
    """
    Display metrics in a table
    """
    table = Table(title=title, style=style)
    table.add_column("Metric", justify="left", style=style)
    table.add_column("Value", justify="left", style=f"bold {style}")
    for k, v in metrics.items():
        table.add_row(k, f"{v:.4f}" if isinstance(v, float) else str(v))
    console.print(table)


def action_start_token(
    tokenizer: Any, ids: list[int], state_len: int, target_content: str | None
) -> int:
    """First token index (into ``ids``) at which the explicit action begins within
    the target span ``ids[state_len:]``.

    Mirrors ``scripts/eval_action_subspan.py::action_start_token`` but operates
    directly on the *already-tokenized* ``state_action_tokens`` (no re-render / no
    second forward). The action (``<tool_call>...</tool_call>`` block or a
    ``Final Answer:`` suffix) is always a **suffix** of the assistant content, so we
    find the largest token index ``k >= state_len`` such that the decoded tail
    ``decode(ids[k:])`` still fully contains the action string — that token is the
    action's first token. Returns ``state_len`` (whole target == action) when there
    is no separable reasoning prefix, matching the subspan script's fallback.

    Shared by the SFT eval metrics (``eval_actiononly_*``) and, when
    ``train_action_only`` is set, by ``prepare_minibatch``'s label mask — so the
    tokens trained and the tokens scored are the same span by construction.
    """
    from act_prm.environments.act_prm_traces.data import extract_action

    action_str = extract_action(target_content or "")
    if not action_str:
        return state_len  # no separable action -> whole target span is the action

    def _tail_has_action(k: int, needle: str) -> bool:
        return needle in tokenizer.decode(ids[k:])

    # Sanity: the action must appear somewhere in the target tail. If the exact
    # extracted string can't be located (chat-template / whitespace artifacts),
    # fall back to a looser marker, else to the whole-target span.
    if not _tail_has_action(state_len, action_str):
        for marker in ("<tool_call>", "Final Answer:"):
            if _tail_has_action(state_len, marker):
                action_str = marker
                break
        else:
            return state_len

    a_start = state_len
    for k in range(state_len, len(ids)):
        if _tail_has_action(k, action_str):
            a_start = k
        else:
            break
    return a_start
