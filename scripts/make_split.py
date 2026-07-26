#!/usr/bin/env python3
"""Build a task-level split for an Act-PRM traces dataset.

Streams the dataset, reconstructs one action-only trajectory per *task* (the
highest-return successful generation), then partitions the usable task ids into:

  - act_prm_train : tasks we infer thoughts on + train on
  - act_prm_eval  : held-out tasks for eval during act-prm generation
  - rl_eval       : held-out tasks reserved for eventual RL evaluation
                    (NEVER seen by act-prm)

Writes data/splits/<name>.json = {dataset, seed, fractions, act_prm_train,
act_prm_eval, rl_eval, counts}.

Egress: needs the HF Hub. Set the forward proxy + token (as elsewhere):
  export https_proxy=http://fwdproxy:8080 http_proxy=http://fwdproxy:8080
  export HF_TOKEN=$(cat /home/mzio/models/token)

Usage:
  uv run python scripts/make_split.py \
    --dataset mzio/aprm-tau2-retail-gpt5m_med-gs8-s0-train --name tau2_retail
  uv run python scripts/make_split.py \
    --dataset mzio/aprm-tau2-airline-gpt5m_med-gs4-s0-train --name tau2_airline
  uv run python scripts/make_split.py \
    --dataset mzio/aprm-snorkelai_agent_finance_reasoning --name snorkel_finance
"""
import argparse
import json
import logging
import random
from pathlib import Path

from act_prm.environments.act_prm_traces.data import load_grouped


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="HF dataset id")
    ap.add_argument("--name", required=True, help="split name -> data/splits/<name>.json")
    ap.add_argument("--out_dir", default="data/splits")
    ap.add_argument("--train_frac", type=float, default=0.70)
    ap.add_argument("--eval_frac", type=float, default=0.15)
    # rl_eval_frac is the remainder (1 - train - eval)
    ap.add_argument("--min_actions", type=int, default=2, help="min actions for a usable trajectory")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    for n in ("httpx", "huggingface_hub", "datasets", "urllib3", "filelock", "fsspec"):
        logging.getLogger(n).setLevel(logging.ERROR)

    print(f"streaming + reconstructing usable tasks from {args.dataset} ...", flush=True)
    by_uid = load_grouped(args.dataset, min_actions=args.min_actions)
    tasks = sorted(by_uid.keys(), key=str)
    n = len(tasks)
    if n == 0:
        raise SystemExit("no usable tasks found (need done+reward>0 rollouts with >= min_actions)")

    rng = random.Random(args.seed)
    rng.shuffle(tasks)
    n_train = int(args.train_frac * n)
    n_eval = int(args.eval_frac * n)
    act_prm_train = tasks[:n_train]
    act_prm_eval = tasks[n_train : n_train + n_eval]
    rl_eval = tasks[n_train + n_eval :]

    out = Path(args.out_dir) / f"{args.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": args.dataset,
        "name": args.name,
        "seed": args.seed,
        "fractions": {
            "act_prm_train": args.train_frac,
            "act_prm_eval": args.eval_frac,
            "rl_eval": round(1 - args.train_frac - args.eval_frac, 4),
        },
        "counts": {
            "total_usable_tasks": n,
            "act_prm_train": len(act_prm_train),
            "act_prm_eval": len(act_prm_eval),
            "rl_eval": len(rl_eval),
        },
        "act_prm_train": act_prm_train,
        "act_prm_eval": act_prm_eval,
        "rl_eval": rl_eval,
    }
    out.write_text(json.dumps(payload, indent=1))
    print(
        f"wrote {out}\n  usable tasks: {n} | act_prm_train {len(act_prm_train)} | "
        f"act_prm_eval {len(act_prm_eval)} | rl_eval {len(rl_eval)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
