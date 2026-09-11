#!/usr/bin/env python3
"""Export rollout conversation text to compact gzipped JSONL, one file per rollout run.

Writes results/trajectories/<domain>__<arm>__<recipe>__<snapshot>__seed<N>.jsonl.gz

Each line is one EVAL episode:
    {task_id, reward, solved, n_steps, n_assistant, n_user, n_tool, messages:[{role,content}]}

Why gzip: the raw trajectories.jsonl files total ~393 MB, which does not belong in git.
They compress ~20x (measured 5%), and the headline set (SGD 1e-3 + seed sweeps + top-K)
is ~44 MB raw -> a couple of MB packed.

Only eval episodes are kept. rollouts_per_task.jsonl / trajectories.jsonl also contain
TRAIN rows -- rl.py generates train rollouts even under --no_train -- and including them
silently inflated a completion denominator once already (44 rows for 42 eval tasks).
"""
import argparse
import glob
import gzip
import json
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
OUT = os.path.join(REPO, "results", "trajectories")

SOURCES = [
    ("logs/tau2bench_retail_rlvr/hf_qwen3_4b_instruct/retail_rollout_*/", "retail"),
    ("logs/tau2bench_airline_rlvr/hf_qwen3_4b_instruct/airline_rollout_*/", "airline"),
    ("logs/act_prm_snorkel_insurance_gym/hf_qwen3_4b_instruct/insurance_rollout_*/", "insurance"),
    ("logs/act_prm_snorkel_finance_gym/hf_qwen3_4b_instruct/finance_rollout_*/", "finance"),
]
# default: the runs the write-ups actually cite
DEFAULT_KEEP = re.compile(r"sgdlr1e_3|g8top|_seed\d+|_s\d{3,}_|BASE")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true",
                    help="export every rollout, not just the cited SGD 1e-3 / seed / top-K set")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    n_files = n_eps = 0
    total = 0
    for pat, domain in SOURCES:
        for d in sorted(glob.glob(os.path.join(REPO, pat))):
            tj = os.path.join(d, "trajectories.jsonl")
            if not os.path.exists(tj):
                continue
            tag = os.path.basename(d.rstrip("/")).split("-act-prm")[0]
            if not args.all and not DEFAULT_KEEP.search(tag):
                continue
            eps = []
            for line in open(tj):
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if r.get("split") != "eval":
                    continue
                msgs = [{"role": m.get("role"), "content": m.get("content")}
                        for m in (r.get("messages") or []) if isinstance(m, dict)]
                roles = [m["role"] for m in msgs]
                eps.append({
                    "task_id": r.get("task_id"),
                    "reward": r.get("final_reward"),
                    "solved": bool((r.get("final_reward") or 0) > 0),
                    "n_steps": r.get("n_steps"),
                    "n_assistant": roles.count("assistant"),
                    "n_user": roles.count("user"),
                    "n_tool": roles.count("tool"),
                    "messages": msgs,
                })
            if not eps:
                continue
            fp = os.path.join(OUT, f"{domain}__{tag}.jsonl.gz")
            with gzip.open(fp, "wt", encoding="utf-8") as f:
                for e in eps:
                    f.write(json.dumps(e, ensure_ascii=False) + "\n")
            total += os.path.getsize(fp)
            n_files += 1
            n_eps += len(eps)
    print(f"  wrote {n_files} files, {n_eps} eval episodes -> {OUT}")
    print(f"  packed size: {total/1048576:.1f} MB")


if __name__ == "__main__":
    main()
