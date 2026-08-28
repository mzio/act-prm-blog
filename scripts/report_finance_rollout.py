#!/usr/bin/env python
"""Task-completion table for the finance v3 rollout evals.

Two sets, reported separately and NEVER pooled:
  fair : the 10 v3 eval questions (the expert solved these; same questions the SFT curves
         score; never in v3 train). Comparable in spirit to the retail/airline rollouts.
  hard : the 29 questions with no successful expert trajectory -- all 123 of their logged
         rollouts are reward=0. Selected for difficulty, so the number is only meaningful
         against other arms on the SAME set, never against `fair` or another domain.

Usage: uv run --no-project python scripts/report_finance_rollout.py
"""
import glob
import json
import os
import re

ORDER = ["actions_only", "expert_thoughts", "expert_thoughts_all", "thoughts_policy", "thoughts_base"]


def main() -> None:
    rows = {}
    for d in glob.glob(
        "logs/act_prm_snorkel_finance_gym/hf_qwen3_4b_instruct/finance_rollout_*_v3_*/"
    ):
        tag = os.path.basename(d.rstrip("/")).split("-act-prm")[0]
        m = re.match(r"finance_rollout_(.+)_v3_(fair|hard)$", tag)
        if not m:
            continue
        arm, setname = m.group(1), m.group(2)
        try:
            recs = [json.loads(l) for l in open(d + "metrics.jsonl") if l.strip()]
        except Exception:
            continue
        ev = [r for r in recs if r.get("eval/try_0/final_reward") is not None]
        if not ev:
            continue
        rows[(setname, arm)] = (int(ev[-1]["eval/try_0/correct"]), int(ev[-1]["eval/try_0/total"]))

    if not rows:
        print("no finance rollout results yet")
        return

    exp = {}
    try:
        sp = json.load(open("data/splits/snorkel_finance_v3.json"))
        exp = {"fair": len(sp["eval_questions"]), "hard": len(sp["hard_rollout_questions"])}
    except Exception:
        pass

    for setname in ("fair", "hard"):
        got = {a: v for (s, a), v in rows.items() if s == setname}
        if not got:
            continue
        want = exp.get(setname)
        note = ""
        totals = {t for _, t in got.values()}
        if want and totals and totals != {want}:
            note = f"   ** expected {want} questions, evaluated {sorted(totals)} **"
        print(f"\n  {setname}{note}")
        base = got.get("actions_only")
        for arm in ORDER + sorted(a for a in got if a not in ORDER):
            if arm not in got:
                continue
            c, t = got[arm]
            delta = ""
            if base and arm != "actions_only" and base[1] == t:
                delta = f"   vs baseline {100 * (c / t - base[0] / base[1]):+.1f}pp"
            print(f"    {arm:22} {c:3}/{t:<3} = {100 * c / t:5.1f}%{delta}")
    print("\n  fair and hard are never pooled: hard is selected for difficulty (all-zero expert reward).")


if __name__ == "__main__":
    main()
