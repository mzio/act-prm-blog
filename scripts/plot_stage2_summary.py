#!/usr/bin/env python
"""Figures for the Stage-2 results write-up (notes/cc-14.0-stage2-summary.md).

Four figures, each answering one question:
  1. stage2-curves-ppl.png   -- do the arms differ on the SFT objective? (training curves)
  2. stage2-inverted-u.png   -- the headline: light fit beats base, PPL-optimal fit is worse
  3. stage2-fit-vs-behaviour.png -- completion as a function of fit depth (retail ladder)
  4. stage2-noise-floors.png -- why only insurance can resolve arm differences

Conventions follow the rest of the blog: Okabe-Ito palette with fixed roles, one hue per
arm held constant across every panel, no dual axes, direct labels where they fit, recessive
grid. Metric is ALWAYS eval_actiononly_ppl -- the action sub-span, the only span comparable
across arms (actions_only has no reasoning prefix).
"""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "img")
os.makedirs(OUT, exist_ok=True)

# Okabe-Ito. Colour follows the ARM, never its rank, and is identical in every panel.
C = {
    "actions_only":        "#D55E00",
    "thoughts_policy":     "#0072B2",
    "expert_thoughts_all": "#009E73",
    "thoughts_base":       "#CC79A7",
    "base":                "#666666",
}
GRID = dict(color="#DDDDDD", lw=0.7)

DOMAINS = [
    ("retail",    "act_prm_tau2_retail",           "retail"),
    ("airline",   "act_prm_tau2_airline",          "airline"),
    ("finance",   "act_prm_snorkel_finance_split", "snorkel_finance_split"),
    ("insurance", "act_prm_snorkel_insurance",     "snorkel_insurance"),
]
ARMS = [
    ("actions_only",             "actions_only",        C["actions_only"]),
    ("thoughts_policy_adamw30",  "thoughts_policy",     C["thoughts_policy"]),
    ("expert_thoughts_all",      "expert_thoughts_all", C["expert_thoughts_all"]),
    ("thoughts_base_adamw30",    "thoughts_base",       C["thoughts_base"]),
]
PAT = "lr1e_3_nb200_flat32sgd_heldout"


def curve(envdir, prefix, arm):
    g = sorted(glob.glob(f"logs/{envdir}/hf_qwen3_4b_instruct/{prefix}_s2_{arm}_{PAT}-*/metrics.jsonl"))
    if not g:
        return [], []
    rows = [json.loads(l) for l in open(g[-1]) if l.strip()]
    pts = [(r["progress/batch"], r["eval/eval_actiononly_ppl"])
           for r in rows if "eval/eval_actiononly_ppl" in r]
    return [p[0] for p in pts], [p[1] for p in pts]


def fig_curves():
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.9))
    for ax, (dom, envdir, prefix) in zip(axes, DOMAINS):
        for arm, label, col in ARMS:
            x, y = curve(envdir, prefix, arm)
            if not x:
                continue
            ax.plot(x, y, color=col, lw=2, label=label)
            bi = min(range(len(y)), key=lambda i: y[i])
            ax.plot(x[bi], y[bi], marker="*", ms=13, color=col, mec="white", mew=1.2, zorder=5)
        ax.axvline(20, color="#999999", ls=":", lw=1.4)
        ax.text(22, ax.get_ylim()[1], "step_0020", fontsize=7.5, color="#666666", va="top")
        ax.set_title(dom, fontsize=11, weight="bold")
        ax.set_xlabel("batch")
        ax.grid(True, **GRID)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("eval action-token PPL")
    axes[0].legend(frameon=False, fontsize=8.5)
    fig.suptitle("Stage-2 SFT training curves  (SGD 1e-3, sft_flat, nb200)  —  ★ = step_best",
                 fontsize=11.5, y=1.04)
    fig.tight_layout()
    fig.savefig(f"{OUT}/stage2-curves-ppl.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_inverted_u():
    """Insurance, every arm, step_0020 vs step_best against the measured base."""
    arms = [("actions_only", 50.0, 10.0), ("thoughts_policy", 52.5, 15.0),
            ("expert_thoughts_all", 47.5, 10.0), ("thoughts_base", 47.5, 20.0),
            ("top-1", 37.5, 10.0), ("top-2", 50.0, 12.5),
            ("top-4", 45.0, 10.0), ("top-8", 52.5, 15.0)]
    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    xs = range(len(arms))
    w = 0.38
    ax.bar([x - w / 2 for x in xs], [a[1] for a in arms], w,
           color="#0072B2", label="step_0020  (light fit)")
    ax.bar([x + w / 2 for x in xs], [a[2] for a in arms], w,
           color="#D55E00", label="step_best  (PPL-optimal)")
    ax.axhline(30.0, color=C["base"], ls="--", lw=1.8)
    # Label sits in clear space ABOVE the line at the left edge; putting it on the right
    # collided with the top-4 bar and the line itself.
    # White backing box: the bars are taller than the line everywhere, so any in-axes
    # placement overlaps one of them.
    ax.text(-0.45, 31.6, "base, no Stage-2  30.0%", fontsize=9,
            color=C["base"], ha="left", va="bottom",
            bbox=dict(facecolor="white", edgecolor="none", pad=1.8, alpha=0.9))
    for x, a in zip(xs, arms):
        ax.text(x - w / 2, a[1] + 1.0, f"{a[1]:.0f}", ha="center", fontsize=8, color="#0072B2")
        ax.text(x + w / 2, a[2] + 1.0, f"{a[2]:.0f}", ha="center", fontsize=8, color="#D55E00")
    ax.set_xticks(list(xs))
    ax.set_xticklabels([a[0] for a in arms], rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("task completion (%)")
    ax.set_ylim(0, 70)
    # upper-right: at upper-left the legend collided with the thoughts_policy value label
    ax.legend(frameon=False, fontsize=9, loc="upper right", ncol=2)
    ax.grid(True, axis="y", **GRID)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title("Insurance (40 held-out tasks): every corpus behaves the same;\n"
                 "only WHEN you stop matters", fontsize=11.5, weight="bold")
    fig.tight_layout()
    fig.savefig(f"{OUT}/stage2-inverted-u.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_fit_vs_behaviour():
    """Retail ladder: completion as a function of how far Stage-2 fit."""
    pts = [("base", 3.8428, 14.3), ("SGD 1e-3\nstep_0020", 3.0847, 31.0),
           ("SGD 1e-3\nstep_best", 2.4625, 23.8), ("SGD 3e-3\nstep_best", 2.2822, 9.5),
           ("AdamW 1e-3\nstep_0020", 1.7828, 11.9), ("AdamW 1e-3\nstep_best", 1.7150, 0.0)]
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    xs = [p[1] for p in pts]
    ys = [p[2] for p in pts]
    ax.plot(xs, ys, color="#999999", lw=1.4, zorder=1)
    ax.scatter(xs, ys, s=110, color="#0072B2", zorder=3, edgecolor="white", lw=1.5)
    ax.axhline(14.3, color=C["base"], ls="--", lw=1.6)
    # Line label goes BELOW the line at the left edge; on the right it collided with the
    # AdamW step_0020 annotation.
    ax.text(3.90, 12.6, "base 14.3%", fontsize=9, color=C["base"], ha="left", va="top")
    # Per-point offsets: the two AdamW points sit near the base line and near y=0, so a
    # uniform +12pt offset overlapped the line, each other, and the connecting segment.
    offs = [(0, 13), (0, 13), (0, 13), (0, -26), (14, 13), (-2, -26)]
    for (lbl, x, y), off in zip(pts, offs):
        ax.annotate(lbl, (x, y), textcoords="offset points", xytext=off,
                    ha="center", fontsize=8, color="#333333")
    ax.invert_xaxis()  # left-to-right = deeper fit
    ax.set_xlabel("eval action-token PPL   (→ deeper fit)")
    ax.set_ylabel("task completion (%)")
    ax.set_ylim(-9, 40)
    ax.grid(True, **GRID)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title("Retail: the SFT objective and agentic competence are anti-correlated",
                 fontsize=11.5, weight="bold")
    fig.tight_layout()
    fig.savefig(f"{OUT}/stage2-fit-vs-behaviour.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_noise():
    """Repeat rollouts of UNCHANGED checkpoints, per harness."""
    data = [("insurance\n(snorkel, n=40)", [52.5, 55.0, 47.5, 55.0], "#009E73"),
            ("retail\n(tau2, n=42)",       [31.0, 16.7, 33.3],       "#D55E00"),
            ("airline\n(tau2, n=18)",      [61.1, 83.3],             "#CC79A7")]
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    for i, (lbl, vals, col) in enumerate(data):
        ax.scatter([i] * len(vals), vals, s=95, color=col, zorder=3,
                   edgecolor="white", lw=1.4)
        ax.plot([i, i], [min(vals), max(vals)], color=col, lw=2.5, alpha=0.35, zorder=2)
        ax.text(i + 0.13, (min(vals) + max(vals)) / 2,
                f"spread\n{max(vals)-min(vals):.1f} pt", fontsize=8.5, color=col, va="center")
    ax.set_xticks(range(len(data)))
    ax.set_xticklabels([d[0] for d in data], fontsize=9.5)
    ax.set_ylabel("task completion (%)")
    ax.grid(True, axis="y", **GRID)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title("Same checkpoint, repeated rollouts:\ntau2 swings 17–22 pts because its user simulator is unseeded",
                 fontsize=11.5, weight="bold")
    fig.tight_layout()
    fig.savefig(f"{OUT}/stage2-noise-floors.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_curves(); print("  wrote stage2-curves-ppl.png")
    fig_inverted_u(); print("  wrote stage2-inverted-u.png")
    fig_fit_vs_behaviour(); print("  wrote stage2-fit-vs-behaviour.png")
    fig_noise(); print("  wrote stage2-noise-floors.png")
