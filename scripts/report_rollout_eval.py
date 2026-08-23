#!/usr/bin/env python
"""Task-completion table for the Stage-2 SFT rollout evals.

Unlike the teacher-forced action-subspan metrics, this scores whether the policy actually
COMPLETES tau2 tasks when it acts on its own -- no gold targets, no inferred-thought
corpus, on never-in-logs tasks. Reads metrics.jsonl so it works mid-sweep.

Usage: uv run --no-project python scripts/report_rollout_eval.py [--regime hide|full]
"""
import argparse, glob, json, os, re

ORDER = ["actions_only", "expert_thoughts", "thoughts_policy", "thoughts_base"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", default="hide", choices=["hide", "full"])
    args = ap.parse_args()
    suffix = "_fullctx" if args.regime == "full" else ""
    rows = {}
    # `_fixeval` marks the expert_thoughts_all re-run whose checkpoint was selected against
    # the corrected (train-only require_thought) eval. Keep it as a separate row rather than
    # overwriting the original -- the two are the comparison.
    for d in glob.glob(f"logs/tau2bench_*_rlvr/hf_qwen3_4b_instruct/*_rollout_*_lr3e_3{suffix}*-*/"):
        tag = os.path.basename(d.rstrip("/")).split("-act-prm")[0]
        m = re.match(rf"(\w+?)_rollout_(.+)_lr3e_3{suffix}(_fixeval)?$", tag)
        if not m:
            continue
        dom, var = m.group(1), m.group(2) + (m.group(3) or "")
        try:
            recs = [json.loads(l) for l in open(d + "metrics.jsonl") if l.strip()]
        except Exception:
            continue
        ev = [r for r in recs if r.get("eval/try_0/final_reward") is not None]
        if not ev:
            continue
        r = ev[-1]
        rows[(dom, var)] = (int(r["eval/try_0/correct"]), int(r["eval/try_0/total"]))
    if not rows:
        print(f"no {args.regime}-regime rollout results yet")
        return
    print(f"TASK COMPLETION on never-in-logs tau2 tasks — {args.regime}-observations, lr 3e-3\n")
    for dom in sorted({d for d, _ in rows}):
        base = rows.get((dom, "actions_only"))
        print(f"  {dom}")
        seen = [v for _d, v in rows if _d == dom]
        for v in ORDER + sorted(v for v in seen if v not in ORDER):
            got = rows.get((dom, v))
            if not got:
                continue
            c, t = got
            delta = ""
            if base and v != "actions_only":
                delta = f"   vs baseline {100*(c/t - base[0]/base[1]):+.1f}pp"
            print(f"    {v:18} {c:3}/{t:<3} = {100*c/t:5.1f}%{delta}")
        print()
    print("n.b. one rollout per task; at p~0.15 on 42 tasks the binomial SE is ~5.5pp.")


if __name__ == "__main__":
    main()
