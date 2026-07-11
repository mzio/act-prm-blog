#!/usr/bin/env python3
"""Overlay the λ-sweep runs: held-out likelihood and thought length vs iteration.

Usage:
  uv run --with matplotlib python scripts/plot_lambda_sweep.py \
      0.15=runs/length_penalty_qwen3_8b_100.json \
      0.4=runs/length_penalty_qwen3_8b_lam04.json \
      1.0=runs/length_penalty_qwen3_8b_lam10.json
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

runs = []
for arg in (sys.argv[1:] or [
    "0.15=runs/length_penalty_qwen3_8b_100.json",
    "0.4=runs/length_penalty_qwen3_8b_lam04.json",
    "1.0=runs/length_penalty_qwen3_8b_lam10.json",
]):
    label, path = arg.split("=", 1)
    runs.append((label, json.loads(Path(path).read_text())))

OUT = Path("assets/img")
INK, MUTED, GRID = "#52514e", "#898781", "#e1e0d9"
COLORS = ["#2a78d6", "#1baf7a", "#eda100", "#4a3aa7"]   # λ ascending + lift variant

plt.rcParams.update({
    "font.family": "Helvetica Neue, Helvetica, Arial, sans-serif",
    "font.size": 11, "text.color": INK,
    "axes.edgecolor": GRID, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "figure.facecolor": "white", "axes.facecolor": "white",
})


def style(ax, title, ylabel):
    ax.set_title(title, fontsize=12, color="#1a1a19", pad=12, loc="left")
    ax.set_xlabel("EM iteration")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=10)


def series(log, key, sub):
    xs = [it["iteration"] for it in log["iterations"]]
    ys = [it[sub][key] for it in log["iterations"]]
    return xs, ys


fig, ax = plt.subplots(figsize=(7, 4.2), dpi=160)
for (label, log), c in zip(runs, COLORS):
    xs, ys = series(log, "mean_likelihood", "eval")
    ax.plot(xs, ys, color=c, lw=2, label=f"λ = {label}" if label[0].isdigit() else label)
style(ax, "Held-out action likelihood p(x | s, z) across length penalties",
      "mean action-token likelihood")
fig.tight_layout()
fig.savefig(OUT / "lenpen-sweep-likelihood.png", bbox_inches="tight")

fig, ax = plt.subplots(figsize=(7, 4.2), dpi=160)
for (label, log), c in zip(runs, COLORS):
    xs, ys = series(log, "mean_thought_tokens", "eval")
    ax.plot(xs, ys, color=c, lw=2, label=f"λ = {label}" if label[0].isdigit() else label)
ax.axhline(200, color=MUTED, lw=1, ls=":")
ax.annotate("thought-token budget (L_max = 200)", xy=(1, 200), xytext=(2, 206),
            fontsize=9, color=MUTED)
style(ax, "Held-out thought length |z| across length penalties", "thought tokens")
fig.tight_layout()
fig.savefig(OUT / "lenpen-sweep-length.png", bbox_inches="tight")

print("wrote", OUT / "lenpen-sweep-likelihood.png", "and", OUT / "lenpen-sweep-length.png")
