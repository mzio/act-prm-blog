#!/usr/bin/env python
"""Collect Stage-2 SFT results for one env into a synced analysis note + CSV.

Reads every run's ``logs/<env>/<model>/<run_tag>-.../metrics.jsonl``, extracts the
best held-out action-PPL (early-stop point) + accuracy there + final train loss, and
writes a markdown table to ``notes/cc-3.0-<env>_sft_results[_<model>].md`` and a CSV under
the model's log dir (``logs/<env>/<model>/<dom>_sft_summary.csv``).

The model is chosen by the ``MODEL_CFG`` env var (default ``hf_qwen3_4b_instruct``); the
default model keeps the original note filename, others get a per-model suffix.

Usage:  MODEL_CFG=hf_qwen3_8b uv run --no-sync python scripts/analyze_sft.py act_prm/tau2_retail
"""
import csv
import json
import os
import re
import sys
from pathlib import Path

ENVCFG = sys.argv[1] if len(sys.argv) > 1 else "act_prm/tau2_retail"
ENVDIR = ENVCFG.replace("/", "_")               # act_prm_tau2_retail
# Model is parametrized via MODEL_CFG (default 4B) so multi-model results coexist.
DEFAULT_MODEL = "hf_qwen3_4b_instruct"
MODEL = os.environ.get("MODEL_CFG", DEFAULT_MODEL)
LOGROOT = Path("logs") / ENVDIR / MODEL
DOM = ENVCFG.split("/")[-1].replace("tau2_", "")  # retail / airline


def load_rows(mfile: Path):
    rows = []
    for line in mfile.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except Exception:
            pass
    return rows


def summarize(run_dir: Path):
    m = run_dir / "metrics.jsonl"
    if not m.exists():
        return None
    rows = load_rows(m)
    if not rows:
        return None
    tag = run_dir.name.split("-act-prm")[0]           # retail_s2_thoughts_policy_heldout[_fullctx]
    regime = "full" if tag.endswith("_fullctx") else "hide"
    variant = re.sub(r"^%s_s2_" % DOM, "", tag).replace("_heldout_fullctx", "").replace("_heldout", "")
    import math
    # Two held-out metrics per eval point:
    #   eval_action_ppl      — the WHOLE (thought + action) span. Inflated by verbose
    #                          thoughts, so it penalizes the thought arms for being
    #                          wordy rather than for predicting the action badly.
    #   eval_actiononly_ppl  — the ACTION SUBSPAN (<tool_call> tokens only). This is
    #                          the metric that actually answers "does the thought help
    #                          predict the next action", and it flips the ranking.
    # actions_only has no thought tokens, so the two coincide for that arm.
    evals = [r for r in rows if "eval/eval_action_ppl" in r]
    subevals = [r for r in rows if r.get("eval/eval_actiononly_ppl") is not None]

    best_ppl = best_acc = best_step = best_eval_loss = None
    sub_at_best_ppl = sub_at_best_acc = None
    if evals:
        b = min(evals, key=lambda r: r["eval/eval_action_ppl"])
        best_ppl = b["eval/eval_action_ppl"]
        best_acc = b.get("eval/eval_action_accuracy")
        best_step = b.get("progress/batch")
        # eval loss is derivable: eval_action_ppl = exp(mean CE) -> eval_loss = ln(ppl)
        best_eval_loss = math.log(best_ppl) if best_ppl and best_ppl > 0 else None
        # What the SAVED step_best checkpoint scores on the subspan. The early-stop
        # criterion is whole-span, so this — not the subspan optimum — is what the
        # downstream RL arm actually warm-starts from.
        sub_at_best_ppl = b.get("eval/eval_actiononly_ppl")
        sub_at_best_acc = b.get("eval/eval_actiononly_accuracy")

    sub_ppl = sub_acc = sub_step = None
    if subevals:
        s = min(subevals, key=lambda r: r["eval/eval_actiononly_ppl"])
        sub_ppl = s["eval/eval_actiononly_ppl"]
        sub_acc = s.get("eval/eval_actiononly_accuracy")
        sub_step = s.get("progress/batch")

    losses = [r["train/loss"] for r in rows if "train/loss" in r]
    return {
        "variant": variant, "regime": regime,
        # action-subspan (primary)
        "best_actiononly_ppl": sub_ppl, "actiononly_acc_at_best": sub_acc,
        "actiononly_best_step": sub_step,
        "actiononly_ppl_at_wholespan_best": sub_at_best_ppl,
        "actiononly_acc_at_wholespan_best": sub_at_best_acc,
        # whole-span (early-stop criterion; kept for continuity)
        "best_eval_action_ppl": best_ppl, "eval_loss_at_best": best_eval_loss,
        "acc_at_best": best_acc, "best_step": best_step,
        "final_train_loss": (losses[-1] if losses else None),
        "n_eval_points": len(evals), "n_subspan_points": len(subevals),
        "run_dir": str(run_dir),
    }


def fmt(x, nd=4):
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) else "—"


def main():
    if not LOGROOT.exists():
        print(f"no logs at {LOGROOT}"); return
    runs = sorted(d for d in LOGROOT.glob(f"{DOM}_s2_*") if d.is_dir())
    results = [s for d in runs if (s := summarize(d))]
    if not results:
        print(f"no SFT runs with metrics under {LOGROOT}"); return
    # order: variant, then hide before full
    order = {"actions_only": 0, "expert_thoughts": 1, "thoughts_policy": 2, "thoughts_base": 3}
    results.sort(key=lambda r: (order.get(r["variant"], 9), r["regime"] != "hide"))

    # Model-aware CSV (under the <MODEL> dir) so per-model summaries coexist.
    csv_path = LOGROOT / f"{DOM}_sft_summary.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader(); w.writerows(results)

    has_sub = any(r["best_actiononly_ppl"] is not None for r in results)
    lines = [
        f"# cc-3.0 — {ENVCFG} Stage-2 SFT results ({MODEL})",
        "",
        "Auto-generated by `scripts/analyze_sft.py`. Lower PPL / higher accuracy = better",
        "next-action fit on the held-out `act_prm_eval` split.",
        "",
        "**Read the action-subspan columns.** `eval_actiononly_ppl` scores ONLY the",
        "`<tool_call>` action tokens; `eval_action_ppl` scores the whole (thought + action)",
        "span and is therefore inflated by verbose thoughts — it penalizes the thought arms",
        "for wordiness rather than for mispredicting the action, which inverts the ranking.",
        "`actions_only` emits no thoughts, so its two columns coincide (a useful sanity check).",
        "",
        "| variant | regime | **action-subspan PPL** | **subspan acc** | subspan best step | whole-span PPL | whole-span acc | whole-span best step | final train loss |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r['variant']} | {r['regime']} | {fmt(r['best_actiononly_ppl'])} | "
            f"{fmt(r['actiononly_acc_at_best'])} | "
            f"{r['actiononly_best_step'] if r['actiononly_best_step'] is not None else '—'} | "
            f"{fmt(r['best_eval_action_ppl'])} | {fmt(r['acc_at_best'])} | "
            f"{r['best_step'] if r['best_step'] is not None else '—'} | "
            f"{fmt(r['final_train_loss'])} |"
        )
    if has_sub:
        lines += [
            "",
            "### What the saved `step_best` checkpoint actually scores",
            "",
            "Early stopping used the **whole-span** metric, so `step_best` — the checkpoint the",
            "Stage-3 RL arms warm-start from — is not the subspan optimum. Subspan scores at",
            "that saved step:",
            "",
            "| variant | regime | subspan PPL @ saved step | subspan acc @ saved step |",
            "|---|---|---|---|",
        ]
        for r in results:
            lines.append(
                f"| {r['variant']} | {r['regime']} | "
                f"{fmt(r['actiononly_ppl_at_wholespan_best'])} | "
                f"{fmt(r['actiononly_acc_at_wholespan_best'])} |"
            )
    lines += [
        "",
        f"CSV: `{csv_path}` (all columns).",
        "",
        "Reading: `actions_only` is the no-thoughts baseline; `expert_thoughts` the oracle",
        "upper-bound; `thoughts_{policy,base}` the Act-PRM inferred-thought arms. The question:",
        "do inferred thoughts recover the expert-thought lift over actions-only, and does it",
        "hold under both context regimes?",
    ]
    # Default model keeps the original note filename (back-compat); other models get a
    # per-model suffix so multi-model notes coexist instead of clobbering each other.
    suffix = "" if MODEL == DEFAULT_MODEL else f"_{MODEL}"
    note = Path("notes") / f"cc-3.0-{DOM}_sft_results{suffix}.md"
    note.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {note} and {csv_path}")


if __name__ == "__main__":
    main()
