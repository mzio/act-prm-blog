"""cc-finance analysis helpers — load Act-PRM (snorkel_finance) training logs and
plot the offline trajectory metrics. Shared by the cc-finance-1.x notebooks and by
scripts so figures are reproducible headless (no nbconvert needed).

Metrics recorded per run in ``logs/<env>/<model>/<run_tag>-.../metrics.jsonl``:
  train (every step): train/loss, train/ppl, train/action_accuracy,
                      train/actiononly_ppl, train/actiononly_accuracy
  eval  (every eval_every): eval/eval_action_{ppl,accuracy,loss}  (whole thought+action span)
                            eval/eval_actiononly_{ppl,accuracy,loss}  (tool_call/Final-Answer sub-span)
x-axis = progress/batch. Whole-span is inflated by verbose thoughts; the ACTION
SUB-SPAN (`*_actiononly_*`) is the trustworthy cross-variant metric (isolated via
action_start_token — only the <tool_call>…/Final Answer: tokens).
"""
from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
ENV = "act_prm_snorkel_finance_split"
MODEL = "hf_qwen3_4b_instruct"

# variant tag (between "s2_" and "-act-prm") -> display label; order = plot order.
VARIANTS = {
    "actions_only": "actions_only (expert action)",
    "expert_thoughts": "expert_thoughts (oracle thought+action)",
    "thoughts_policy": "aprm thought+action (policy, best)",
    "thoughts_policy_last": "aprm thought+action (policy, last)",
    "thoughts_base": "aprm thought+action (base, best)",
    "thoughts_base_last": "aprm thought+action (base, last)",
}
COLORS = {
    "actions_only": "#4C78A8",
    "expert_thoughts": "#F58518",
    "thoughts_policy": "#54A24B",
    "thoughts_policy_last": "#88C580",
    "thoughts_base": "#B279A2",
    "thoughts_base_last": "#D0A9C6",
}


def _parse_tag(run_dir_name: str) -> tuple[str, str] | None:
    """(variant, regime) from a run-dir name, or None if not a Stage-2 SFT run."""
    head = run_dir_name.split("-act-prm")[0]
    key = "_s2_"
    if key not in head:
        return None
    tag = head.split(key, 1)[1]              # e.g. thoughts_policy_last_heldout_fullctx
    regime = "full" if tag.endswith("_fullctx") else "hide"
    tag = tag[: -len("_fullctx")] if regime == "full" else tag
    tag = tag[: -len("_heldout")] if tag.endswith("_heldout") else tag
    if tag not in VARIANTS:
        return None
    return tag, regime


def load_sft_runs(env: str = ENV, model: str = MODEL) -> dict:
    """{(variant, regime): {"train": rows, "eval": rows_deduped}} for every SFT run
    found on disk. Robust to partial completion (returns whatever exists)."""
    out: dict = {}
    base = REPO / "logs" / env / model
    for md in sorted(glob.glob(str(base / "*" / "metrics.jsonl"))):
        parsed = _parse_tag(Path(md).parent.name)
        if parsed is None:
            continue
        rows = [json.loads(l) for l in open(md) if l.strip()]
        train = [r for r in rows if any(k == "train/loss" for k in r)]
        # eval rows: dedup by progress/batch, keep last (early-flush writes a first row,
        # the step-end writes a second with train/* — keep the later, complete one).
        seen, ev = {}, []
        for r in rows:
            if any(k.endswith("eval_action_ppl") and not k.endswith("_best") for k in r):
                seen[r.get("progress/batch")] = r
        ev = [seen[k] for k in sorted(seen, key=lambda x: (x is None, x))]
        out[parsed] = {"train": train, "eval": ev}
    return out


def _series(rows: list[dict], key_suffix: str, x_key: str = "progress/batch"):
    xs, ys = [], []
    for i, r in enumerate(rows):
        hit = [r[k] for k in r if k.endswith(key_suffix) and not k.endswith("_best") and not k.endswith("_best_step")]
        if not hit:
            continue
        xs.append(r.get(x_key, i))
        ys.append(hit[0])
    return xs, ys


def plot_sft_grid(runs: dict, split: str = "eval", out_path: str | None = None):
    """2x2 grid: {ppl, accuracy} x {whole-span, action-sub-span} vs batch, one line
    per variant, solid=hide / dashed=full. ``split`` in {"eval","train"}."""
    if split == "eval":
        panels = [
            ("eval_action_ppl", "whole-span ppl (thought+action)"),
            ("eval_actiononly_ppl", "ACTION-subspan ppl (tool_call only)"),
            ("eval_action_accuracy", "whole-span token accuracy"),
            ("eval_actiononly_accuracy", "ACTION-subspan token accuracy"),
        ]
    else:
        panels = [
            ("train/ppl", "train whole-span ppl"),
            ("train/actiononly_ppl", "train ACTION-subspan ppl"),
            ("train/action_accuracy", "train whole-span accuracy"),
            ("train/actiononly_accuracy", "train ACTION-subspan accuracy"),
        ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, (suf, title) in zip(axes.ravel(), panels):
        for (variant, regime), d in sorted(runs.items(), key=lambda kv: list(VARIANTS).index(kv[0][0])):
            xs, ys = _series(d[split], suf)
            if not xs:
                continue
            ax.plot(xs, ys, label=f"{variant} [{regime}]", color=COLORS.get(variant, "gray"),
                    ls="-" if regime == "hide" else "--", marker="o", ms=3, lw=1.6, alpha=.9)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("training batch")
        ax.grid(alpha=.25)
    axes.ravel()[0].legend(fontsize=7, loc="upper right")
    fig.suptitle(f"snorkel_finance {MODEL} — SFT {split} curves (whole-span vs action-subspan)", fontsize=13)
    fig.tight_layout()
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        print("wrote", out_path)
    return fig


def best_table(runs: dict) -> list[dict]:
    """Per run: best (min) eval action-subspan ppl + accuracy at that point."""
    tbl = []
    for (variant, regime), d in sorted(runs.items(), key=lambda kv: list(VARIANTS).index(kv[0][0])):
        xs, ppl = _series(d["eval"], "eval_actiononly_ppl")
        _, acc = _series(d["eval"], "eval_actiononly_accuracy")
        _, wppl = _series(d["eval"], "eval_action_ppl")
        if not ppl:
            continue
        i = min(range(len(ppl)), key=lambda k: ppl[k])
        tbl.append({"variant": variant, "regime": regime, "n_eval": len(ppl),
                    "best_actiononly_ppl": round(ppl[i], 4),
                    "actiononly_acc_at_best": round(acc[i], 4) if acc else None,
                    "wholespan_ppl_at_best": round(wppl[i], 4) if wppl else None,
                    "at_batch": xs[i]})
    return tbl


def load_em_runs(env: str = ENV, model: str = MODEL) -> dict:
    """{scorer: rows} for the Stage-1 EM runs (scorer in {policy, base})."""
    out = {}
    base = REPO / "logs" / env / model
    for scorer in ("policy", "base"):
        hits = sorted(glob.glob(str(base / f"snorkel_finance_split_s1_{scorer}-*" / "metrics.jsonl")))
        if hits:
            out[scorer] = [json.loads(l) for l in open(hits[0]) if l.strip()]
    return out


def plot_em(em: dict, out_path: str | None = None):
    """Stage-1 EM: held-out eval reward (mean p(x|s,z)) vs batch, per scorer, with the
    argmax `best_step` marked. Shows whether the thought policy keeps improving past the
    first epoch (batch 29) or plateaus early (as on airline)."""
    fig, ax = plt.subplots(figsize=(9, 5))
    col = {"policy": "#54A24B", "base": "#B279A2"}
    for scorer, rows in em.items():
        xs, ys = _series(rows, "final_reward")  # per-batch rollout reward proxy
        if xs:
            ax.plot(xs, ys, marker="o", ms=3, lw=1.5, color=col.get(scorer, "gray"),
                    label=f"{scorer}-scored", alpha=.85)
        # mark the eval best_step if recorded
        bsteps = [r[k] for r in rows for k in r if k.endswith("final_reward_best_step")]
        if bsteps:
            ax.axvline(bsteps[-1], ls=":", color=col.get(scorer, "gray"), alpha=.6)
    ax.axvline(29, ls="--", color="gray", alpha=.4)
    ax.text(29, ax.get_ylim()[1], " 1 epoch", va="top", fontsize=8, color="gray")
    ax.set(xlabel="EM batch", ylabel="reward (mean p(x|s,z) − len penalty)",
           title=f"snorkel_finance {MODEL} — Stage-1 EM reward (dotted = eval best_step)")
    ax.grid(alpha=.25); ax.legend()
    fig.tight_layout()
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path, dpi=120, bbox_inches="tight"); print("wrote", out_path)
    return fig
