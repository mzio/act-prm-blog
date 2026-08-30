#!/usr/bin/env python
"""Stage-2 SFT: action-token PPL and accuracy per domain, three arms overlaid.

Layout: 2 rows x 4 cols. Top row = eval_actiononly_ppl, bottom = eval_actiononly_accuracy,
one column per domain, three arms overlaid per panel. ★ marks each arm's best step (the
one step_best is saved from).

Both metrics are the ACTION SUB-SPAN only -- the thought tokens are excluded via
action_start_token. That is the only span comparable across arms: actions_only has no
reasoning prefix, so its whole target span IS its action span, while the thought arms'
whole-span numbers include tokens the baseline never has to model.

Arms share a y-axis WITHIN a domain but not across domains: the domains sit at very
different absolute difficulty (airline ~2.0, finance ~1.6) and a shared scale would flatten
the within-domain arm gaps this chart exists to show.
"""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Okabe-Ito, fixed roles: baseline / method-under-test / reference upper bound.
ARMS = [
    ("actions_only",            "actions_only",           "#D55E00"),
    ("thoughts_policy_adamw30", "thoughts_policy (ActPRM)", "#0072B2"),
    ("expert_thoughts",         "expert_thoughts",        "#009E73"),
]
DOMAINS = [
    ("retail",    "act_prm_tau2_retail",          "retail"),
    ("airline",   "act_prm_tau2_airline",         "airline"),
    ("finance",   "act_prm_snorkel_finance_split", "snorkel_finance_split"),
    ("insurance", "act_prm_snorkel_insurance",    "snorkel_insurance"),
]
OUT = "/tmp/aprm_plots"


def curve(envdir, prefix, arm):
    """(batch -> (ppl, acc)) for one arm; falls back to the nb150 dir for retail
    actions_only, which was trained before the batch cap was cut to 100."""
    pats = [f"logs/{envdir}/hf_qwen3_4b_instruct/{prefix}_s2_{arm}_lr1e_3_adamw_nb100_heldout-*/",
            f"logs/{envdir}/hf_qwen3_4b_instruct/{prefix}_s2_{arm}_lr1e_3_adamw_nb150_heldout-*/"]
    ds = [d for p in pats for d in glob.glob(p) if os.path.exists(d + "metrics.jsonl")]
    if not ds:
        return {}

    def read(d):
        pts = {}
        for line in open(d + "metrics.jsonl"):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if "eval/eval_actiononly_ppl" in r and r.get("progress/batch") is not None:
                pts[r["progress/batch"]] = (r["eval/eval_actiononly_ppl"],
                                            r.get("eval/eval_actiononly_accuracy", 0.0))
        return pts

    # Pick the run with the MOST eval points, not the newest. Killed/restarted attempts
    # leave behind newer run dirs that hold a metrics.jsonl with no eval rows -- taking
    # the newest silently reported "no runs" for retail actions_only, whose real data
    # lives in the older nb150 dir.
    cands = sorted(((len(read(d)), d) for d in ds), reverse=True)
    return dict(sorted(read(cands[0][1]).items())) if cands[0][0] else {}


data = {dom: {arm: curve(envd, pref, arm) for arm, _, _ in ARMS}
        for dom, envd, pref in DOMAINS}

fig, axes = plt.subplots(2, 4, figsize=(19, 8.4))
for col, (dom, _, _) in enumerate(DOMAINS):
    for row, (mi, ylab) in enumerate([(0, "eval action-token PPL"),
                                      (1, "eval action-token accuracy")]):
        ax = axes[row][col]
        any_data = False
        for arm, label, colour in ARMS:
            pts = data[dom].get(arm) or {}
            if not pts:
                continue
            any_data = True
            xs = list(pts)
            ys = [v[mi] for v in pts.values()]
            ax.plot(xs, ys, color=colour, lw=2.0, marker="o", ms=3.6, label=label, zorder=3)
            # ★ = step_best, always chosen on MIN ppl (even on the accuracy panel)
            bb = min(pts, key=lambda b: pts[b][0])
            ax.plot(bb, pts[bb][mi], marker="*", ms=16, color=colour,
                    mec="white", mew=1.2, zorder=6, linestyle="none")
        if not any_data:
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
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
axes[0][0].legend(frameon=False, fontsize=9, loc="upper right")
fig.suptitle("Act-PRM Stage-2 SFT — action-token metrics by domain  ·  AdamW, lr 1e-3, "
             "hide-observations, r8/a16  ·  ★ = step_best (min PPL)", fontsize=13, y=.98)
fig.tight_layout(rect=(0, 0, 1, .95))
p = f"{OUT}/stage2_curves.png"
os.makedirs(OUT, exist_ok=True)
fig.savefig(p, dpi=150, bbox_inches="tight", facecolor="white")
print("wrote", p)
for dom, _, _ in DOMAINS:
    for arm, label, _ in ARMS:
        pts = data[dom].get(arm) or {}
        if not pts:
            print(f"  {dom:10} {label:26} -- none"); continue
        bb = min(pts, key=lambda b: pts[b][0])
        print(f"  {dom:10} {label:26} best ppl={pts[bb][0]:.4f} acc={pts[bb][1]:.4f} "
              f"@b{bb:<4} ({len(pts)} evals, to b{max(pts)})")
