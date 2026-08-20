#!/usr/bin/env python3
"""Plot Stage-2 SFT (behavioral-cloning) train + eval curves across every dataset.

One figure per context regime, laid out 4 x 3 -- train group first, then eval:
    row 1 = TRAIN token PPL              row 3 = EVAL token PPL
    row 2 = TRAIN action-token accuracy  row 4 = EVAL action-token accuracy
    columns = tau2 retail | tau2 airline | snorkel finance
Every method's curve is overlaid within a panel, one fixed hue per method across
all panels and figures.

"Action-token accuracy" is teacher-forced next-token accuracy over the action span
(`act_correct / act_tokens`, trainers/sft.py:184) -- the fraction of `<tool_call>`
tokens whose argmax matches gold. It is NOT generated-sequence exact match, so it
does not collapse to zero.

The eval metric family is selectable:
  --span subspan  (default) eval_actiononly_{ppl,accuracy} -- ONLY the action tokens.
  --span whole    eval_action_{ppl,accuracy} -- the whole (thought + action) span,
                  which is inflated by verbose thoughts and so penalizes the thought
                  arms for wordiness rather than for mispredicting the action.
`actions_only` emits no thought tokens, so its two families coincide.

Train-side coverage is uneven and is drawn only where it exists; a panel with no
series is annotated rather than left blank:
  train/ppl                 all three datasets
  train/action_accuracy     airline + finance only (retail's runs predate it)
  train/actiononly_{ppl,accuracy}   finance only (preferred when --span subspan)

Usage:
  uv run --with matplotlib python scripts/plot_sft_curves.py [--span subspan|whole]
      [--model hf_qwen3_4b_instruct] [--out notebooks/figs_sft]
"""
import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (env dir, run-tag domain prefix, display name)
DOMAINS = [
    ("act_prm_tau2_retail", "retail", "tau2 retail"),
    ("act_prm_tau2_airline", "airline", "tau2 airline"),
    ("act_prm_snorkel_finance_split", "snorkel_finance_split", "snorkel finance"),
]
# Fixed categorical order -- assigned by slot, never cycled, so a method keeps its
# hue across every panel and figure. Validated (light surface, adjacent pairlist):
# worst CVD dE 9.1, worst normal-vision dE 19.6.
VARIANTS = [
    ("actions_only", "actions_only (baseline)", "#2a78d6"),
    ("expert_thoughts", "expert_thoughts (oracle)", "#eb6834"),
    ("thoughts_policy", "thoughts_policy", "#1baf7a"),
    ("thoughts_base", "thoughts_base", "#eda100"),
    ("thoughts_policy_last", "thoughts_policy_last", "#e87ba4"),
    ("thoughts_base_last", "thoughts_base_last", "#008300"),
]
INK, MUTED, GRID = "#52514e", "#898781", "#e1e0d9"

plt.rcParams.update({
    "font.family": "Helvetica Neue, Helvetica, Arial, sans-serif",
    "font.size": 10, "text.color": INK,
    "axes.edgecolor": GRID, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "figure.facecolor": "white", "axes.facecolor": "white",
})


def load_runs(env_dir: str, dom: str, model: str):
    """{(variant, regime): [rows]} for one dataset's Stage-2 SFT runs."""
    root = Path("logs") / env_dir / model
    out = {}
    for d in sorted(root.glob(f"{dom}_s2_*")):
        m = d / "metrics.jsonl"
        if not (d.is_dir() and m.exists()):
            continue
        tag = d.name.split("-act-prm")[0]
        regime = "full" if tag.endswith("_fullctx") else "hide"
        variant = re.sub(rf"^{dom}_s2_", "", tag).replace("_heldout_fullctx", "").replace("_heldout", "")
        rows = []
        for line in m.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
        if rows:
            out[(variant, regime)] = rows
    return out


def series(rows, keys):
    """(x, y, key) for the first key in `keys` that has data. Deduped on batch."""
    for key in keys:
        pts = {}
        for r in rows:
            if r.get(key) is None or r.get("progress/batch") is None:
                continue
            pts[r["progress/batch"]] = r[key]
        if pts:
            xs = sorted(pts)
            return xs, [pts[x] for x in xs], key
    return [], [], None


def style(ax, ylabel, title):
    ax.set_title(title, fontsize=10.5, color="#1a1a19", pad=8, loc="left")
    ax.set_xlabel("batch")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--span", choices=["subspan", "whole"], default="subspan")
    ap.add_argument("--model", default="hf_qwen3_4b_instruct")
    ap.add_argument("--out", default="notebooks/figs_sft")
    args = ap.parse_args()

    sub = args.span == "subspan"
    span_label = "action-subspan" if sub else "whole-span"
    # Each row: (label, [candidate keys in preference order], ylabel, lower_is_better)
    ROWS = [
        ("TRAIN token PPL",
         ["train/actiononly_ppl", "train/ppl"] if sub else ["train/ppl"],
         "PPL (lower better)", True),
        ("TRAIN action-token accuracy",
         ["train/actiononly_accuracy", "train/action_accuracy"] if sub else ["train/action_accuracy"],
         "accuracy", False),
        (f"EVAL token PPL ({span_label})",
         ["eval/eval_actiononly_ppl"] if sub else ["eval/eval_action_ppl"],
         "PPL (lower better)", True),
        (f"EVAL action-token accuracy ({span_label})",
         ["eval/eval_actiononly_accuracy"] if sub else ["eval/eval_action_accuracy"],
         "accuracy", False),
    ]

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    data = {dom: load_runs(env, dom, args.model) for env, dom, _ in DOMAINS}

    for regime in ("hide", "full"):
        fig, axes = plt.subplots(len(ROWS), len(DOMAINS),
                                 figsize=(5.2 * len(DOMAINS), 3.7 * len(ROWS)))
        drew_any = False
        for col, (_, dom, disp) in enumerate(DOMAINS):
            for r, (rowtitle, keys, ylab, lower_better) in enumerate(ROWS):
                ax = axes[r][col]
                used_key = None
                for variant, label, color in VARIANTS:
                    rows = data[dom].get((variant, regime))
                    if not rows:
                        continue
                    xs, ys, key = series(rows, keys)
                    if not xs:
                        continue
                    drew_any = True
                    used_key = key
                    ax.plot(xs, ys, color=color, linewidth=2.0, label=label, zorder=3)
                    # Selective direct mark at the best point only -- also the relief
                    # the light-surface contrast WARN requires.
                    pick = min if lower_better else max
                    bi = pick(range(len(ys)), key=lambda i: ys[i])
                    ax.plot(xs[bi], ys[bi], "o", color=color, markersize=5,
                            markeredgecolor="white", markeredgewidth=1.4, zorder=4)
                title = f"{disp} — {rowtitle}"
                # Say so when the row fell back to a different key, or has nothing.
                if used_key is None:
                    ax.text(0.5, 0.5, f"not logged\n({' / '.join(keys)})", ha="center",
                            va="center", transform=ax.transAxes, color=MUTED, fontsize=10)
                    ax.set_xticks([]); ax.set_yticks([])
                elif used_key != keys[0]:
                    title += f"  [{used_key.split('/')[-1]}]"
                style(ax, ylab, title)

        if not drew_any:
            plt.close(fig)
            print(f"[{regime}] no runs found — skipped")
            continue

        # Method identity via one figure-level legend (>4 series, so legend-borne);
        # text stays in ink, the colored line sample carries the hue.
        handles, labels = [], []
        for ax in axes.flat:
            for h, l in zip(*ax.get_legend_handles_labels()):
                if l not in labels:
                    handles.append(h); labels.append(l)
        fig.legend(handles, labels, loc="lower center", ncol=min(3, len(labels)),
                   frameon=False, bbox_to_anchor=(0.5, -0.005))
        fig.suptitle(
            f"Stage-2 SFT (behavioral cloning) — {regime}-observations regime "
            f"({args.model})\naction-token accuracy = teacher-forced argmax==gold over "
            f"action tokens; eval = {span_label}",
            fontsize=13, color="#1a1a19", x=0.005, ha="left", y=1.0,
        )
        fig.tight_layout(rect=(0, 0.045, 1, 0.965))
        path = outdir / f"sft_curves_{args.span}_{regime}.png"
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
