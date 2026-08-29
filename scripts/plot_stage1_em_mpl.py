#!/usr/bin/env python
"""Stage-1 EM (AdamW / lr 4e-5 / r32-a32 / lp=0 / 30 batches) — ONE figure, per-domain panels.

Layout: a 2x2 grid of per-domain reward panels, plus a short full-width action_prob strip.

Each domain panel shows the per-batch reward as translucent points (the actual, noisy
measurements -- one batch is 4 trajectories x 4 sampled thoughts) with an opaque centred
running average carrying the trend. All four panels SHARE a y-axis: per-panel autoscaling
would make a plateau at 0.72 and one at 0.81 look identical.

``final_reward`` IS the EM objective p(x|s,z): length_penalty is 0 here, so
reward == likelihood exactly (verified against generations.jsonl). ``action_prob`` is NOT
that -- it is pinned at 1.0 in every logged row of every domain (validity/parse rate,
trivially 1 because the action is teacher-forced), hence one shared strip, not four copies.

Eval ran only at the final batch (--no_initial_eval, --eval_every = num_batches), so eval
is a single marker per domain, never line-joined.
"""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = [
    ("retail",    "#0072B2", "logs/act_prm_tau2_retail/hf_qwen3_4b_instruct/retail_s1em_policy_adamw30-*/"),
    ("airline",   "#E69F00", "logs/act_prm_tau2_airline/hf_qwen3_4b_instruct/airline_s1em_policy_adamw30-*/"),
    ("finance",   "#009E73", "logs/act_prm_snorkel_finance_split/hf_qwen3_4b_instruct/finance_s1em_policy_adamw30-*/"),
    ("insurance", "#D55E00", "logs/act_prm_snorkel_insurance/hf_qwen3_4b_instruct/insurance_s1em_policy_adamw30-*/"),
]
OUT, WIN = "/tmp/aprm_plots", 5


def load(pat):
    d = max(glob.glob(pat), key=os.path.getmtime)
    rows = [json.loads(l) for l in open(d + "metrics.jsonl") if l.strip()]
    out = {}
    for split in ("train", "eval"):
        pts = {}
        for r in rows:
            b = r.get("progress/batch")
            for m in ("final_reward", "action_prob"):
                v = r.get(f"{split}/try_0/{m}")
                if v is not None and b is not None:
                    pts.setdefault(m, {})[b] = v      # dedupe repeated eval rows
        out[split] = {m: sorted(d_.items()) for m, d_ in pts.items()}
    return out


def running_mean(vals, win):
    """Centred moving average; windows shrink at the edges rather than dropping points."""
    n, h = len(vals), win // 2
    return [sum(vals[max(0, i - h):min(n, i + h + 1)]) / len(vals[max(0, i - h):min(n, i + h + 1)])
            for i in range(n)]


data = {name: load(pat) for name, _, pat in RUNS}
os.makedirs(OUT, exist_ok=True)

allv = [v for n, _, _ in RUNS for _, v in data[n]["train"]["final_reward"]]
allv += [v for n, _, _ in RUNS for _, v in data[n]["eval"]["final_reward"]]
pad = (max(allv) - min(allv)) * .09
ylim = (min(allv) - pad, max(allv) + pad)

fig = plt.figure(figsize=(13.4, 9.6))
gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, .52], hspace=.34, wspace=.16)

for i, (name, colour, _) in enumerate(RUNS):
    ax = fig.add_subplot(gs[i // 2, i % 2])
    tr = data[name]["train"]["final_reward"]
    xs, ys = [b for b, _ in tr], [v for _, v in tr]
    sm = running_mean(ys, WIN)
    ax.scatter(xs, ys, s=40, color=colour, alpha=.28, edgecolors="none", zorder=3,
               label="per-batch reward")
    ax.plot(xs, sm, color=colour, lw=2.4, zorder=4, label=f"running mean (window {WIN})")
    for b, v in data[name]["eval"]["final_reward"]:
        ax.plot(b, v, marker="*", ms=18, color=colour, mec="white", mew=1.4, zorder=6,
                linestyle="none", label=f"eval @ batch {b}")
        ax.annotate(f"{v:.3f}", (b, v), textcoords="offset points", xytext=(11, -4),
                    fontsize=9.5, color=colour, weight="bold")
    ax.set_title(f"{name}   {ys[0]:.3f} → {sm[-1]:.3f}", fontsize=12.5, loc="left", pad=7)
    ax.set_ylim(*ylim)
    ax.set_xlim(-1, 34)
    ax.grid(alpha=.25, lw=.7)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if i % 2 == 0:
        ax.set_ylabel("reward  =  p(x | s, z)")
    if i // 2 == 1:
        ax.set_xlabel("EM training batch")
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")

axp = fig.add_subplot(gs[2, :])
for i, (name, colour, _) in enumerate(RUNS):
    tr = data[name]["train"]["action_prob"]
    axp.plot([b for b, _ in tr], [v for _, v in tr], color=colour, label=name,
             lw=3.4 - .7 * i, alpha=.85, ls=["-", "--", "-.", ":"][i], zorder=3 + i)
axp.set_ylim(.90, 1.02)
axp.set_xlim(-1, 34)
axp.set_xlabel("EM training batch")
axp.set_ylabel("action_prob")
axp.set_title("action_prob — pinned at 1.0 in all four domains (validity rate, not a likelihood)",
              fontsize=11, loc="left", pad=7)
axp.text(.5, .38, "constant 1.0000 in every logged row, train and eval",
         transform=axp.transAxes, ha="center", fontsize=10, color="#555", style="italic")
axp.grid(alpha=.25, lw=.7)
axp.set_axisbelow(True)
for s in ("top", "right"):
    axp.spines[s].set_visible(False)
axp.legend(frameon=False, fontsize=9, ncol=4, loc="lower right")

fig.suptitle("Act-PRM Stage-1 EM  ·  AdamW, lr 4e-5, r32/a32, length_penalty 0, 30 batches"
             "   —   shared y-axis across domains",
             fontsize=13, y=.975)
p = f"{OUT}/stage1_em_panels.png"
fig.savefig(p, dpi=165, bbox_inches="tight", facecolor="white")
print("wrote", p)
for n, _, _ in RUNS:
    tr = data[n]["train"]["final_reward"]; ys = [v for _, v in tr]
    print(f"  {n:10} {ys[0]:.3f} -> {running_mean(ys, WIN)[-1]:.3f} (raw last {ys[-1]:.3f})"
          f"  eval {data[n]['eval']['final_reward'][-1][1]:.4f}")
