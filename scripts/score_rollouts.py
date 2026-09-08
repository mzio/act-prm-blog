#!/usr/bin/env python3
"""Split-aware, deduped rollout scoring. Lives in the repo (not /tmp, which gets reaped).

Three corrections this encodes, each of which silently corrupted a reading before:
  * rollouts_per_task.jsonl mixes eval AND train rows (rl.py generates train rollouts even
    in --no_train mode) -> always filter split=="eval".
  * task_id can be null on the snorkel envs (they store tasks as dicts, so getattr(.,"id")
    was None) -> fall back to sample_id, else every task collapses into one bucket.
  * the snorkel gyms run negative_rewards=True, so a 0-score is clamped to -1.0. Summing
    final_reward gives a NEGATIVE count and a math-domain error in the Wilson CI ->
    success is final_reward > 0.
"""
import glob, json, math, os, sys

def wilson(k, n, z=1.96):
    if not n:
        return 0.0, 0.0
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return 100 * (c - h), 100 * (c + h)

for pat in sys.argv[1:]:
    for d in sorted(glob.glob(pat)):
        p = os.path.join(d, "rollouts_per_task.jsonl")
        if not os.path.exists(p):
            continue
        rows = [json.loads(l) for l in open(p) if l.strip()]
        ev = [r for r in rows if r.get("split") == "eval"]
        seen = {}
        for r in ev:
            k = r.get("task_id")
            k = str(k) if k is not None else f's{r.get("sample_id")}'
            seen.setdefault(k, r["final_reward"])
        v = list(seen.values())
        k = sum(1 for x in v if x > 0)
        n = len(v)
        tj = os.path.join(d, "trajectories.jsonl")
        nt = sum(1 for _ in open(tj)) if os.path.exists(tj) else 0
        lo, hi = wilson(k, n)
        tag = os.path.basename(d.rstrip("/")).split("-act-prm")[0]
        print(f"  {tag[:58]:58s} {k:>3}/{n:<3} {100*k/max(n,1):>5.1f}%  [{lo:.1f},{hi:.1f}]  traj={nt}")
