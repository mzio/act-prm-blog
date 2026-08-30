#!/usr/bin/env python
"""Stage-2 SFT TRAIN curves: action-token PPL and accuracy per domain, three arms overlaid.

Companion to plot_stage2_curves.py (which plots the EVAL curves). Same 2x4 layout so the
two figures can be read side by side: top row PPL, bottom row accuracy, one column per
domain, three arms per panel.

Train metrics are logged EVERY batch over a batch of 4 trajectories, so they are far
noisier than the every-10-batch eval points: raw series are drawn translucent and a
centred running mean carries the trend.

Both metrics are the ACTION SUB-SPAN (train/actiononly_*), matching the eval figure --
NOT the span the loss uses. The loss is over the full label_mask (thought + action);
action_mask is metrics-only and never enters it (sft.py compute_loss). So for the thought
arms these curves score ~59% of the tokens the model is actually trained on
(insurance thoughts_policy: only 48%).
"""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARMS = [
    ("actions_only",            "actions_only",            "#D55E00"),
    ("thoughts_policy_adamw30", "thoughts_policy (ActPRM)", "#0072B2"),
    ("expert_thoughts",         "expert_thoughts",         "#009E73"),
]
DOMAINS = [
    ("retail",    "act_prm_tau2_retail",           "retail"),
    ("airline",   "act_prm_tau2_airline",          "airline"),
    ("finance",   "act_prm_snorkel_finance_split", "snorkel_finance_split"),
    ("insurance", "act_prm_snorkel_insurance",     "snorkel_insurance"),
]
OUT, WIN = "/tmp/aprm_plots", 9


def series(envdir, prefix, arm):
    """[(batch, ppl, acc)] of TRAIN points for one arm."""
    pats = [f"logs/{envdir}/hf_qwen3_4b_instruct/{prefix}_s2_{arm}_lr1e_3_adamw_nb100_heldout-*/",
            f"logs/{envdir}/hf_qwen3_4b_instruct/{prefix}_s2_{arm}_lr1e_3_adamw_nb150_heldout-*/"]
    ds = [d for p in pats for d in glob.glob(p) if os.path.exists(d + "metrics.jsonl")]

    def read(d):
        out = {}
        for line in open(d + "metrics.jsonl"):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("train/actiononly_ppl") is not None and r.get("progress/batch") is not None:
                out[r["progress/batch"]] = (r["train/actiononly_ppl"],
                                            r.get("train/actiononly_accuracy", 0.0))
        return out

    # most points wins, not newest: killed attempts leave newer but near-empty run dirs
    cands = sorted(((len(read(d)), d) for d in ds), reverse=True)
    if not cands or not cands[0][0]:
        return []
    pts = read(cands[0][1])
    return [(b, *pts[b]) for b in sorted(pts)]


def running_mean(v, win):
    n, h = len(v), win // 2
    return [sum(v[max(0, i - h):min(n, i + h + 1)]) / len(v[max(0, i - h):min(n, i + h + 1)])
            for i in range(n)]


data = {dom: {arm: series(envd, pref, arm) for arm, _, _ in ARMS} for dom, envd, pref in DOMAINS}

fig, axes = plt.subplots(2, 4, figsize=(19, 8.4))
for col, (dom, _, _) in enumerate(DOMAINS):
    for row, (mi, ylab) in enumerate([(1, "TRAIN action-token PPL"),
                                      (2, "TRAIN action-token accuracy")]):
        ax = axes[row][col]
        got = False
        for arm, label, colour in ARMS:
            s = data[dom].get(arm) or []
            if len(s) < 3:
                continue
            got = True
            xs = [p[0] for p in s]
            ys = [p[mi] for p in s]
            ax.plot(xs, ys, color=colour, lw=1.0, alpha=.22, zorder=2)          # raw, noisy
            ax.plot(xs, running_mean(ys, WIN), color=colour, lw=2.2, zorder=4,
                    label=label)                                                 # trend
        if not got:
            ax.text(.5, .5, "no runs yet", transform=ax.transAxes, ha="center",
                    color="#999", style="italic")
        if row == 0:
            ax.set_title(dom, fontsize=13, loc="left", pad=8, weight="bold")
        if col == 0:
            ax.set_ylabel(ylab)
        if row == 1:
            ax.set_xlabel("SFT batch")
        ax.grid(alpha=.25, lw=.7)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
axes[0][0].legend(frameon=False, fontsize=9, loc="upper right")
fig.suptitle("Act-PRM Stage-2 SFT — TRAIN action-token metrics by domain  ·  AdamW, lr 1e-3, "
             f"hide-observations, r8/a16  ·  faint = per-batch, bold = running mean (window {WIN})",
             fontsize=13, y=.98)
fig.tight_layout(rect=(0, 0, 1, .95))
os.makedirs(OUT, exist_ok=True)
p = f"{OUT}/stage2_train_curves.png"
fig.savefig(p, dpi=150, bbox_inches="tight", facecolor="white")
print("wrote", p)
for dom, _, _ in DOMAINS:
    for arm, label, _ in ARMS:
        s = data[dom].get(arm) or []
        if len(s) < 3:
            print(f"  {dom:10} {label:26} -- none"); continue
        sm_p = running_mean([x[1] for x in s], WIN)
        sm_a = running_mean([x[2] for x in s], WIN)
        print(f"  {dom:10} {label:26} ppl {sm_p[0]:.3f}->{sm_p[-1]:.3f}  "
              f"acc {sm_a[0]:.3f}->{sm_a[-1]:.3f}  ({len(s)} batches)")
