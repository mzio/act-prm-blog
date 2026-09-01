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


def rotate_stale_run_artifacts(cfg: Any, checkpoint_path: str | None) -> None:
    """Move a PREVIOUS run's metrics + step snapshots aside before this run writes here.

    Run dirs are named from a hash of the DECLARED config, so a re-run with identical
    config resolves to the same directory. Two artifacts had no protection:
      * metrics.jsonl APPENDS -> one file holding two runs with the batch counter
        resetting mid-file (scripts/truncate_restarted_metrics.py exists to clean this up
        after the fact; this prevents it instead).
      * step_NNNN/ overwrite only as far as the NEW run gets. On 2026-09-01 an
        expert_thoughts_all re-run early-stopped at b120, so b130/b140 survived as the
        PREVIOUS (invalid) run's weights inside the live dir -- a glob over step_* then
        mixes two models into one curve.
    generations.jsonl already rotates in generator/act_prm/base.py; this closes the rest.
    """
    import os
    import shutil

    log_path = cfg.get("log_path") if cfg is not None else None
    if log_path:
        m = os.path.join(log_path, "metrics.jsonl")
        if os.path.exists(m) and os.path.getsize(m) > 0:
            n = 0
            while os.path.exists(f"{m}.prev{n}"):
                n += 1
            os.rename(m, f"{m}.prev{n}")
            print(f"NOTE: rotated a previous run's metrics.jsonl -> metrics.jsonl.prev{n}")
    if checkpoint_path and os.path.isdir(checkpoint_path):
        stale = sorted(
            d for d in os.listdir(checkpoint_path)
            if d.startswith("step_") and os.path.isdir(os.path.join(checkpoint_path, d))
        )
        if stale:
            k = 0
            while os.path.exists(os.path.join(checkpoint_path, f"prev_run_{k}")):
                k += 1
            dst = os.path.join(checkpoint_path, f"prev_run_{k}")
            os.makedirs(dst, exist_ok=True)
            for d in stale:
                shutil.move(os.path.join(checkpoint_path, d), os.path.join(dst, d))
            print(f"NOTE: moved {len(stale)} previous-run snapshot(s) -> prev_run_{k}/")
