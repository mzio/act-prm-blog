#!/usr/bin/env python
"""Compact status + results table for the Stage-2 SFT LR sweep.

One row per arm: its held-out action-subspan curve (first -> best -> last), the total
improvement, whether it early-stopped, and the train-side action-only metrics. Reads
metrics.jsonl directly, so it works on in-flight runs.

The action-subspan metrics (eval_actiononly_*, train/actiononly_*) are the ones to read:
they score ONLY the <tool_call> action tokens, while the loss trains the full
thought+action span. Whole-span PPL is inflated by verbose thoughts and ranks the
thought arms artificially low.

Usage: uv run --no-project python scripts/report_sft_sweep.py [--match _lr1e_3]
"""
import argparse
import glob
import json
import os
import re

DOMAINS = [
    ("act_prm_tau2_retail", "retail"),
    ("act_prm_tau2_airline", "airline"),
    ("act_prm_snorkel_finance_split", "snorkel_finance_split"),
]


def curve(rows, key):
    pts = {}
    for r in rows:
        v = r.get(key)
        if v is not None and r.get("progress/batch") is not None:
            pts[r["progress/batch"]] = v
    return [(b, pts[b]) for b in sorted(pts)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--match", default="_lr1e_3", help="substring the run tag must contain")
    ap.add_argument("--model", default="hf_qwen3_4b_instruct")
    args = ap.parse_args()

    print(f"{'arm':52} {'reg':4} {'b':>4} {'eval ppl first->last':>22} {'Δ%':>7} {'acc':>7} {'train ao ppl':>13}")
    print("-" * 118)
    n_done = n_run = 0
    for env, dom in DOMAINS:
        for d in sorted(glob.glob(f"logs/{env}/{args.model}/{dom}_s2_*{args.match}*/")):
            m = os.path.join(d, "metrics.jsonl")
            if not os.path.exists(m):
                continue
            tag = os.path.basename(d.rstrip("/")).split("-act-prm")[0]
            rows = [json.loads(l) for l in open(m) if l.strip()]
            if not rows:
                continue
            regime = "full" if tag.endswith("_fullctx") else "hide"
            variant = re.sub(rf"^{dom}_s2_", "", tag).replace("_heldout_fullctx", "").replace("_heldout", "")
            last_b = max(r.get("progress/batch", -1) for r in rows)
            nb = int((re.search(r"-nb=(\d+)", d) or [0, 0])[1])
            ec = curve(rows, "eval/eval_actiononly_ppl")
            ac = curve(rows, "eval/eval_actiononly_accuracy")
            tc = curve(rows, "train/actiononly_ppl")
            if not ec:
                print(f"{dom+'/'+variant:52} {regime:4} {last_b:4} {'(no evals yet)':>22}")
                continue
            first, last = ec[0][1], ec[-1][1]
            delta = 100 * (first - last) / first
            done = nb and last_b >= nb - 1
            n_done += bool(done); n_run += 1
            flag = "" if done else "  <running>"
            print(f"{dom+'/'+variant:52} {regime:4} {last_b:4} "
                  f"{first:10.4f} ->{last:9.4f} {delta:6.2f}% {ac[-1][1] if ac else float('nan'):7.4f} "
                  f"{tc[-1][1] if tc else float('nan'):13.4f}{flag}")
    print(f"\n{n_done}/{n_run} arms complete   (Δ% = held-out action-subspan PPL improvement, higher is better)")


if __name__ == "__main__":
    main()
