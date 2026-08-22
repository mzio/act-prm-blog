#!/usr/bin/env python
"""Keep the BEST generation per task in an Act-PRM thought corpus.

The Stage-1 relabel emitted several expert generations per task and the export kept them
all, so the thought arms trained on more supervised steps than actions_only /
expert_thoughts (retail 1043 vs 635 -- a 64% imbalance that is an alternative explanation
for their advantage). Naive dedup (keep the first) fixes the volume but throws away the
choice of WHICH generation.

This scores every corpus trajectory by the mean EM likelihood of its committed thoughts --
generations.jsonl records `likelihoods` per candidate and the `best` index the generator
selected, joined back to the corpus on committed-thought text -- and keeps the argmax per
uid. Trajectories with no match fall back to generation_id order, so the result is always
exactly one trajectory per uid.

Usage: uv run --no-project python scripts/make_bestgen_corpus.py <corpus_dir> <out_dir>
"""
import json, os, re, sys, statistics
from collections import defaultdict

src, dst = sys.argv[1], sys.argv[2]
meta_p = os.path.join(src, "meta.json")
gen_p = json.load(open(meta_p)).get("exported_from") if os.path.exists(meta_p) else None

# committed-thought text -> likelihood of that thought
score_of = {}
if gen_p and os.path.exists(gen_p):
    for line in open(gen_p):
        try:
            r = json.loads(line)
        except Exception:
            continue
        th, lk, b = r.get("thoughts") or [], r.get("likelihoods") or [], int(r.get("best", 0) or 0)
        if th and 0 <= b < len(th) and b < len(lk):
            score_of[th[b].strip()] = float(lk[b])
print(f"  scored thoughts available: {len(score_of)}")

from act_prm.environments.act_prm_traces.data import extract_action


def traj_score(ex):
    vals = []
    for m in ex["messages"]:
        if m.get("role") != "assistant":
            continue
        c = m.get("content") or ""
        a = extract_action(c)
        pre = (c[: c.find(a)] if a and c.find(a) >= 0 else "").strip()
        if pre in score_of:
            vals.append(score_of[pre])
    return statistics.mean(vals) if vals else None


os.makedirs(dst, exist_ok=True)
for split in ("train", "eval"):
    p = os.path.join(src, f"{split}.json")
    if not os.path.exists(p):
        continue
    data = json.load(open(p))
    by_uid = defaultdict(list)
    for ex in data:
        by_uid[ex.get("uid")].append(ex)
    keep, n_scored = [], 0
    for uid, exs in by_uid.items():
        scored = [(traj_score(e), e) for e in exs]
        with_s = [(s, e) for s, e in scored if s is not None]
        if with_s:
            n_scored += 1
            keep.append(max(with_s, key=lambda t: t[0])[1])
        else:
            keep.append(sorted(exs, key=lambda e: e.get("generation_id", 0))[0])
    st = lambda d: sum(1 for e in d for m in e["messages"] if m["role"] == "assistant")
    print(f"  {split}: {len(data)} -> {len(keep)} traj ({st(data)} -> {st(keep)} steps); "
          f"{n_scored}/{len(by_uid)} uids picked by EM likelihood")
    json.dump(keep, open(os.path.join(dst, f"{split}.json"), "w"))
if os.path.exists(meta_p):
    m = json.load(open(meta_p))
    m["subsampled"] = "best generation per uid by mean EM likelihood of committed thoughts"
    m["subsampled_from"] = src
    json.dump(m, open(os.path.join(dst, "meta.json"), "w"), indent=1)
