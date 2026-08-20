#!/usr/bin/env python3
"""Compare Stage-2 SFT learning rates on one arm: loss + PPL + accuracy curves.

2x2 panels — train loss, train action-only PPL, held-out action-only PPL, held-out
action-only accuracy — with one line per learning rate. Existence of the train-side
action-only keys depends on when the run was launched (they were added mid-sweep), so
a panel with no data is annotated rather than left blank.

The point of the figure: at 4e-5 and 1e-4 the held-out curves are flat noise; at 1e-3
they descend monotonically. That is the difference between a LoRA that moved and one
that did not (max|(alpha/r)B@A| 3.7e-5 / 9.2e-5 / ~9e-4 respectively, against base
weights of order 1e-2).

Usage:
  uv run --with matplotlib python scripts/plot_lr_comparison.py
      [--env act_prm_tau2_retail] [--dom retail] [--variant actions_only]
      [--out notebooks/figs_sft]
"""
import argparse
import glob
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, MUTED, GRID = "#52514e", "#898781", "#e1e0d9"
# Categorical slots 1-4, fixed order (validated all-pairs for the first three).
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]

plt.rcParams.update({
    "font.family": "Helvetica Neue, Helvetica, Arial, sans-serif",
    "font.size": 10, "text.color": INK,
    "axes.edgecolor": GRID, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "figure.facecolor": "white", "axes.facecolor": "white",
})


def load(run_dir):
    m = os.path.join(run_dir, "metrics.jsonl")
    if not os.path.exists(m):
        return []
    out = []
    for line in open(m):
        try:
            out.append(json.loads(line))
        except Exception:
            pass
    return out


def series(rows, key):
    pts = {}
    for r in rows:
        v = r.get(key)
        if v is not None and r.get("progress/batch") is not None:
            pts[r["progress/batch"]] = v
    xs = sorted(pts)
    return xs, [pts[x] for x in xs]


def style(ax, ylabel, title):
    ax.set_title(title, fontsize=11, color="#1a1a19", pad=8, loc="left")
    ax.set_xlabel("batch")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="act_prm_tau2_retail")
    ap.add_argument("--dom", default="retail")
    ap.add_argument("--variant", default="actions_only")
    ap.add_argument("--model", default="hf_qwen3_4b_instruct")
    ap.add_argument("--out", default="notebooks/figs_sft")
    args = ap.parse_args()

    root = f"logs/{args.env}/{args.model}"
    base = f"{args.dom}_s2_{args.variant}"
    # (label, glob) — the baseline has no _lr tag; the sweep arms carry _lr<rate>[_nb<n>]
    candidates = [
        ("lr 4e-5 (shipped)", f"{root}/{base}_heldout-*/"),
        ("lr 1e-4",           f"{root}/{base}_lr1e_4_heldout-*/"),
        ("lr 1e-3",           f"{root}/{base}_lr1e_3_heldout-*/"),
        ("lr 1e-3, 150 batches", f"{root}/{base}_lr1e_3_nb150_heldout-*/"),
    ]
    runs = []
    for i, (label, pat) in enumerate(candidates):
        ds = sorted(glob.glob(pat))
        if not ds:
            continue
        rows = load(ds[0])
        if rows:
            runs.append((label, rows, COLORS[i % len(COLORS)]))
    if not runs:
        print(f"no runs found for {base}"); return

    PANELS = [
        ("train loss", ["train/loss"], "loss", True),
        ("train action-only PPL", ["train/actiononly_ppl"], "PPL (lower better)", True),
        ("held-out action-only PPL", ["eval/eval_actiononly_ppl"], "PPL (lower better)", True),
        ("held-out action-only accuracy", ["eval/eval_actiononly_accuracy"], "accuracy", False),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.4))
    for i, (title, keys, ylab, lower) in enumerate(PANELS):
        ax = axes[i // 2][i % 2]
        drew = False
        for label, rows, color in runs:
            xs, ys = [], []
            for k in keys:
                xs, ys = series(rows, k)
                if xs:
                    break
            if not xs:
                continue
            drew = True
            ax.plot(xs, ys, color=color, linewidth=2.0, label=label, zorder=3)
            pick = min if lower else max
            bi = pick(range(len(ys)), key=lambda j: ys[j])
            ax.plot(xs[bi], ys[bi], "o", color=color, markersize=5,
                    markeredgecolor="white", markeredgewidth=1.4, zorder=4)
        if not drew:
            ax.text(0.5, 0.5, f"not logged\n({keys[0]})", ha="center", va="center",
                    transform=ax.transAxes, color=MUTED, fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
        style(ax, ylab, title)

    h, l = [], []
    for ax in axes.flat:
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            if ll not in l:
                h.append(hh); l.append(ll)
    fig.legend(h, l, loc="lower center", ncol=min(4, len(l)), frameon=False,
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(
        f"Stage-2 SFT learning-rate comparison — {args.dom} / {args.variant} / hide-observations "
        f"({args.model})\nloss trains the full thought+action span; PPL/accuracy score the "
        f"action tokens only",
        fontsize=12.5, color="#1a1a19", x=0.005, ha="left", y=1.0,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    p = outdir / f"lr_comparison_{args.dom}_{args.variant}.png"
    fig.savefig(p, dpi=160, bbox_inches="tight")
    print(f"wrote {p}  ({len(runs)} runs: {', '.join(l for _, l in zip(runs, l))})")


if __name__ == "__main__":
    main()
