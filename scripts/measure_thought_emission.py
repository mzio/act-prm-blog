#!/usr/bin/env python3
"""How often does a rolled-out checkpoint actually THINK before it acts?

Reads the exported rollout text (results/trajectories/*.jsonl.gz) and reports, per
(domain, arm, snapshot), the fraction of tool-call turns that carry >=10 chars of prose
before the ``<tool_call>`` block. That is the same predicate ``--require_thought`` uses to
select trainable targets (sft_flat.py:117-122), so the number is directly comparable to a
corpus's thought coverage.

WHY THIS EXISTS: every thought arm is trained at 100% thought coverage, but measured at
rollout the arms emit a thought before only ~4-17% of their tool calls -- and
``actions_only``, trained with no thoughts at all, emits them at the same rate or higher.
The Stage-2 corpus therefore barely expresses itself at inference, which is the obvious
candidate explanation for why corpus choice does not move rollout success.

Turns with no ``<tool_call>`` (final natural-language answers) are excluded from the
denominator: they are not action turns and have no thought/action split to measure.
"""
import argparse
import collections
import glob
import gzip
import json
import os
import statistics as st

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAJ = os.path.join(REPO, "results", "trajectories")

# longest first so thoughts_policy_adamw30 is not shadowed by thoughts_policy
ARMS = ["expert_thoughts_all", "expert_thoughts", "thoughts_policy_g8top1",
        "thoughts_policy_g8top2", "thoughts_policy_g8top4", "thoughts_policy_g8top8",
        "thoughts_policy_adamw30", "thoughts_base_adamw30", "thoughts_policy",
        "thoughts_base", "actions_only", "BASE"]

MIN_THOUGHT_CHARS = 10  # matches sft_flat.py's _has_thought


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min_turns", type=int, default=50,
                    help="skip (domain, arm, snapshot) groups with fewer tool-call turns")
    args = ap.parse_args()

    g = collections.defaultdict(lambda: [0, 0, 0, []])  # tool-call, with-thought, no-tool-call, lens
    for fp in sorted(glob.glob(os.path.join(TRAJ, "*.jsonl.gz"))):
        base = os.path.basename(fp)[: -len(".jsonl.gz")]
        domain, tag = base.split("__", 1)
        arm = next((a for a in ARMS if a in tag), None)
        if arm is None:
            continue
        snap = next((s for s in ("step_0020", "step_best", "step_last") if s in tag), "?")
        rec = g[(domain, arm, snap)]
        for line in gzip.open(fp, "rt", encoding="utf-8"):
            for m in json.loads(line)["messages"]:
                if m.get("role") != "assistant":
                    continue
                c = m.get("content") or ""
                i = c.find("<tool_call>")
                if i < 0:
                    rec[2] += 1
                    continue
                rec[0] += 1
                pre = c[:i].strip()
                if len(pre) >= MIN_THOUGHT_CHARS:
                    rec[1] += 1
                    rec[3].append(len(pre))

    hdr = ("domain", "arm", "snapshot", "tool_call_turns", "pct_with_thought",
           "median_thought_chars", "non_action_turns")
    print(f"{hdr[0]:10s} {hdr[1]:24s} {hdr[2]:10s} {hdr[3]:>15s} {hdr[4]:>16s} "
          f"{hdr[5]:>20s} {hdr[6]:>16s}")
    out = []
    for k in sorted(g):
        n, t, na, lens = g[k]
        if n < args.min_turns:
            continue
        med = round(st.median(lens), 1) if lens else ""
        print(f"{k[0]:10s} {k[1]:24s} {k[2]:10s} {n:>15d} {100*t/n:>15.1f}% "
              f"{str(med):>20s} {na:>16d}")
        out.append(dict(zip(hdr, (k[0], k[1], k[2], n, round(100 * t / n, 2), med, na))))

    import csv
    fp = os.path.join(REPO, "results", "thought_emission.csv")
    with open(fp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(hdr))
        w.writeheader()
        w.writerows(out)
    print(f"\n  wrote {fp}  ({len(out)} groups)")


if __name__ == "__main__":
    main()
