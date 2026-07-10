#!/usr/bin/env python3
"""Plot reward + thought-length curves from a length-penalty run log.

Usage:
  uv run --with matplotlib python scripts/plot_length_penalty_run.py \
      [runs/length_penalty_qwen3_8b_full.json] [assets/img]
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOG = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/length_penalty_qwen3_8b_full.json")
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "assets/img")
OUT.mkdir(parents=True, exist_ok=True)

log = json.loads(LOG.read_text())
iters = [it["iteration"] for it in log["iterations"]]

INK, MUTED, GRID = "#52514e", "#898781", "#e1e0d9"
BLUE, AQUA = "#2a78d6", "#1baf7a"      # eval, train (entity-stable across both charts)

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


def series(key, sub):
    return [it[sub][key] for it in log["iterations"]]


# --- chart 1: action-likelihood reward ---
fig, ax = plt.subplots(figsize=(7, 4.2), dpi=160)
ax.plot(iters, series("mean_likelihood", "train"), color=AQUA, lw=2, label="train (mean of G)")
ax.plot(iters, series("mean_best_likelihood", "train"), color=AQUA, lw=2, ls="--", label="train (best of G)")
ax.plot(iters, series("mean_likelihood", "eval"), color=BLUE, lw=2, label="held-out (mean of G)")
ax.plot(iters, series("mean_best_likelihood", "eval"), color=BLUE, lw=2, ls="--", label="held-out (best of G)")
style(ax, "Action likelihood p(x | s, z) during length-penalized Act-PRM training",
      "mean action-token likelihood")
fig.tight_layout()
fig.savefig(OUT / "lenpen-likelihood.png", bbox_inches="tight")

# --- chart 2: thought length ---
fig, ax = plt.subplots(figsize=(7, 4.2), dpi=160)
ax.plot(iters, series("mean_thought_tokens", "train"), color=AQUA, lw=2, label="train (mean of G)")
ax.plot(iters, series("mean_selected_thought_tokens", "train"), color=AQUA, lw=2, ls="--", label="train (selected ẑ)")
ax.plot(iters, series("mean_thought_tokens", "eval"), color=BLUE, lw=2, label="held-out (mean of G)")
ax.plot(iters, series("mean_selected_thought_tokens", "eval"), color=BLUE, lw=2, ls="--", label="held-out (selected ẑ)")
style(ax, "Thought length |z| during length-penalized Act-PRM training", "thought tokens")
fig.tight_layout()
fig.savefig(OUT / "lenpen-thought-length.png", bbox_inches="tight")

print("wrote", OUT / "lenpen-likelihood.png", "and", OUT / "lenpen-thought-length.png")

# --- checkpoint generations, printed as a markdown snippet for the README ---
for ck in log.get("checkpoints", []):
    p = ck["probe"]
    b = p["best"]
    print(f"\n### checkpoint @ iteration {ck['iteration']}")
    print(f"sampler: `{ck['sampler_path']}`")
    print(f"best-of-G: p(x|s,z)={p['likelihoods'][b]:.4f}, |z|={p['thought_tokens'][b]} tokens")
    print(f"> {p['thoughts'][b]}")
