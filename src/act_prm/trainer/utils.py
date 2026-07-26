"""
Helper functions for STRL trainers
"""

from typing import Any

from rich.console import Console
from rich.table import Table

console = Console()


def is_better(x: float, y: float, metric: str) -> bool:
    """
    Determine if x is better than y for a given metric.

    Lower is better for loss / perplexity / NLL-style metrics (matched by
    substring, so ``eval/ppl``, ``train/loss`` etc. are covered); higher is
    better otherwise (reward/accuracy).
    """
    lower_is_better = any(t in metric.lower() for t in ("loss", "ppl", "perplex", "nll"))
    return x <= y if lower_is_better else x >= y


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
