#!/usr/bin/env python
"""Exit 0 if a rollout eval looks real, 1 if it looks like a user-sim outage.

A transient Claude Agent SDK failure kills every episode on its FIRST step. The trainer
records that as 0/N correct, indistinguishable from a genuine 0% unless you look at how
many generate calls were made.

The discriminator is calls-per-task, not `timesteps` (which is 1.00 even on healthy runs):
  healthy retail thoughts_policy : 552 calls / 42 tasks = 13.1 per task
  outage  (08-22 01:35-02:09)    :  42 calls / 42 tasks =  1.0 per task
An episode that never gets past its first turn cannot have solved anything, so refuse to
bank it as a result.
"""
import json, os, sys

MIN_CALLS_PER_TASK = 2.0
d = sys.argv[1] if len(sys.argv) > 1 else ""
m = os.path.join(d, "metrics.jsonl")
if not d or not os.path.exists(m):
    print(f"  check: no metrics at {d!r}")
    sys.exit(1)
rows = [json.loads(l) for l in open(m) if l.strip()]
ev = [r for r in rows if r.get("eval/try_0/final_reward") is not None]
if not ev:
    print("  check: no eval rows")
    sys.exit(1)
r = ev[-1]
correct = int(r.get("eval/try_0/correct") or 0)
total = int(r.get("eval/try_0/total") or 0)
calls = max((x.get("usage/total_generate_calls") or 0) for x in rows)
per_task = calls / total if total else 0.0
if per_task < MIN_CALLS_PER_TASK:
    print(f"  check: INVALID -- {correct}/{total}, only {calls} generate calls "
          f"({per_task:.2f}/task); episodes died on turn 1")
    sys.exit(1)
print(f"  check: ok -- {correct}/{total}, {calls} generate calls ({per_task:.1f}/task)")
sys.exit(0)
