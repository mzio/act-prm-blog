#!/usr/bin/env python
"""Build the on-disk (train, eval) trajectory pools for an Act-PRM traces env
*offline*, from the HF-hub-cached parquet — no Hub streaming.

Needed on boxes whose agent egress can't reach HF's CDN/Xet (so the env's normal
first-run `load_dataset(repo, streaming=True)` fails), but where the dataset repo
has already been `hf download`ed into HF_HOME. It runs the env's EXACT grouping /
splitting logic (`load_split`), only redirecting the single streaming call to read
the local parquet, then writes train.json/eval.json/meta.json via `save_pools` —
so a subsequent run hits the env's `pools_exist` fast path with identical pools.

Usage:
  HF_HOME=/data/users/$USER/models/hf_cache HF_HUB_OFFLINE=1 \
    uv run python scripts/prebuild_pools_offline.py \
      --split_file data/splits/tau2_airline.json --dataset_path data/tau2_airline
  # --parquet PATH  to point at a specific parquet (else auto-found in HF_HOME cache)
"""
import argparse
import glob
import json
import os
from pathlib import Path

import datasets as hfds

from act_prm.environments.act_prm_traces import data as aprm_data


def find_parquet(dataset_repo: str) -> str:
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    org, name = dataset_repo.split("/", 1)
    base = Path(hf_home) / "hub" / f"datasets--{org}--{name}" / "snapshots"
    hits = sorted(glob.glob(str(base / "*" / "**" / "*.parquet"), recursive=True))
    if not hits:
        raise FileNotFoundError(
            f"no cached parquet for {dataset_repo} under {base} — `hf download "
            f"--repo-type dataset {dataset_repo}` from a network-capable shell first"
        )
    return hits[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split_file", required=True)
    ap.add_argument("--dataset_path", required=True)
    ap.add_argument("--parquet", default=None)
    args = ap.parse_args()

    dataset_repo = json.loads(Path(args.split_file).read_text())["dataset"]
    parquet = args.parquet or find_parquet(dataset_repo)
    print(f"[prebuild] split_file={args.split_file}")
    print(f"[prebuild] dataset_repo={dataset_repo}")
    print(f"[prebuild] parquet={parquet}")

    # Redirect the one hub-streaming call (load_grouped -> load_dataset(repo,
    # streaming=True)) to the local parquet; leave every other datasets call intact.
    real_load_dataset = hfds.load_dataset

    def patched(path, *a, **k):
        if path == dataset_repo:
            return real_load_dataset("parquet", data_files=parquet, split="train")
        return real_load_dataset(path, *a, **k)

    hfds.load_dataset = patched
    try:
        train_pool, eval_pool = aprm_data.load_split(args.split_file)
    finally:
        hfds.load_dataset = real_load_dataset

    aprm_data.save_pools(
        args.dataset_path,
        train_pool,
        eval_pool,
        meta={
            "dataset": dataset_repo,
            "split_file": args.split_file,
            "num_trajectories": None,
            "eval_trajectories": None,
            "source": "prebuild_pools_offline",
            "parquet": parquet,
        },
    )
    print(f"[prebuild] wrote {len(train_pool)} train / {len(eval_pool)} eval -> {args.dataset_path}")


if __name__ == "__main__":
    main()
