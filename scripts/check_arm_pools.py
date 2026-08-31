#!/usr/bin/env python
"""Assert every arm of a domain shares the SAME eval set.

The arms differ only in the assistant/target content; observations, prompt and the
train/eval partition must be identical or the comparison is meaningless. Two bugs of
exactly this shape already shipped:
  * finance expert_thoughts used data/snorkel_finance_split_expert_thoughts, whose eval
    set shares 3/25 trajectories with the _v3 eval set the other arms used.
  * insurance expert_thoughts crashed because the derived pool path did not exist.
Both were silent at runtime. Run this BEFORE a sweep.

Usage: uv run --no-project python scripts/check_arm_pools.py <base_pool> <pool2> [...]
Exit 1 on any mismatch.
"""
import hashlib
import json
import os
import sys


def key(t):
    """Identity of a trajectory = its non-assistant content (prompt + observations)."""
    return hashlib.md5(
        "|".join((m.get("content") or "")[:300] for m in t["messages"]
                 if m.get("role") != "assistant").encode()
    ).hexdigest()


def load(d, split):
    p = os.path.join(d, f"{split}.json")
    if not os.path.isfile(p):
        return None
    return {key(t) for t in json.load(open(p))}


def main():
    pools = sys.argv[1:]
    if len(pools) < 2:
        raise SystemExit("need a base pool and at least one arm pool")
    base = pools[0]
    ok = True
    ref = {s: load(base, s) for s in ("train", "eval")}
    if ref["eval"] is None:
        raise SystemExit(f"base pool has no eval.json: {base}")
    print(f"base: {base}  train={len(ref['train'])} eval={len(ref['eval'])}")
    for d in pools[1:]:
        cur = {s: load(d, s) for s in ("train", "eval")}
        if cur["eval"] is None:
            print(f"  MISSING  {d}  (no eval.json)")
            ok = False
            continue
        # SET EQUALITY, not containment. Containment ("is every base trajectory present?")
        # passed while data/tau2_retail_expert_thoughts held 10 eval trajectories against
        # the base pool's 8 -- so that arm was scored on 115 steps vs the others' 92 and
        # its PPL was not comparable. Extra trajectories are just as disqualifying as
        # missing ones.
        shared = len(cur["eval"] & ref["eval"])
        extra = len(cur["eval"] - ref["eval"])
        missing = len(ref["eval"] - cur["eval"])
        tr = len(cur["train"] & ref["train"]) if cur["train"] else 0
        tr_extra = len(cur["train"] - ref["train"]) if cur["train"] else 0
        bad = extra or missing
        if bad:
            ok = False
        print(f"  {'BAD ' if bad else 'OK  '} {d}\n"
              f"        eval : {shared} shared, {missing} MISSING, {extra} EXTRA "
              f"(base has {len(ref['eval'])}, arm has {len(cur['eval'])})\n"
              f"        train: {tr} shared, {tr_extra} extra "
              f"(base has {len(ref['train'])}, arm has {len(cur['train'] or [])})")
    if not ok:
        raise SystemExit("ARM POOL MISMATCH -- arms would be scored on different eval sets")
    print("all arms share the base eval set")


if __name__ == "__main__":
    main()
