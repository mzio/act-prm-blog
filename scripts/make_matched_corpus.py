#!/usr/bin/env python
"""Subsample an Act-PRM thought corpus to ONE generation per task.

Control for a confound in the Stage-2 comparison: the Stage-1 relabel emitted up to two
generations per task and the export kept both, so the thought corpora carry 80
trajectories / 1043 assistant steps while the actions_only and expert_thoughts pools carry
49 / 635. The Act-PRM arms therefore trained on ~64% more supervised steps than the arms
they were compared against -- an alternative explanation for their advantage that has
nothing to do with thought quality.

This writes <corpus>_1gen/ keeping the FIRST generation of each uid (deterministic, no
RNG), so the thought arms can be re-trained at matched volume.

Usage: uv run --no-project python scripts/make_matched_corpus.py \
         data/sft_corpus/tau2_retail/policy data/sft_corpus/tau2_retail/policy_1gen
"""
import json
import os
import shutil
import sys

src, dst = sys.argv[1], sys.argv[2]
os.makedirs(dst, exist_ok=True)
for split in ("train", "eval"):
    p = os.path.join(src, f"{split}.json")
    if not os.path.exists(p):
        continue
    data = json.load(open(p))
    seen, keep = set(), []
    for ex in data:
        uid = ex.get("uid")
        if uid in seen:
            continue
        seen.add(uid)
        keep.append(ex)
    steps_in = sum(1 for e in data for m in e["messages"] if m["role"] == "assistant")
    steps_out = sum(1 for e in keep for m in e["messages"] if m["role"] == "assistant")
    json.dump(keep, open(os.path.join(dst, f"{split}.json"), "w"))
    print(f"  {split}: {len(data)} -> {len(keep)} trajectories, {steps_in} -> {steps_out} assistant steps")
# carry the meta forward with a provenance note
mp = os.path.join(src, "meta.json")
if os.path.exists(mp):
    meta = json.load(open(mp))
    meta["subsampled"] = "one generation per uid (first occurrence) to match the "
    meta["subsampled"] += "actions_only / expert_thoughts pools on trajectory + step count"
    meta["subsampled_from"] = src
    json.dump(meta, open(os.path.join(dst, "meta.json"), "w"), indent=1)
