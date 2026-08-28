#!/usr/bin/env python
"""Behavioural comparison of the finance rollout arms, for when completion is 0 everywhere.

Every finance arm scores 0 on the gym: ~75% of episodes exhaust the turn cap without ever
calling respond_user, and of the ~25% that do answer, none is judged correct. Completion
therefore cannot rank the arms. Two things still can, and both are measured here:

  answer rate      -- fraction of episodes that reach respond_user before the cap. Measures
                      whether the arm learned to TERMINATE, which is a real skill the
                      expert demonstrates in 100% of its trajectories.
  calculator usage -- the policy called `calculator` 0 times in 2452 turns while the expert
                      uses it in 15% of its calls (53/363). Inspecting graded answers shows
                      the failure directly: asked for a ratio of tax-rate components, the
                      model retrieves every component correctly and then recites them
                      instead of computing the quotient. Whether an arm recovers the
                      arithmetic step is the substantive question finance can still answer.

Usage: uv run --no-project python scripts/report_finance_behaviour.py
"""
import collections
import glob
import os
import re

from datasets import load_from_disk

EXPERT = {"sql_query": 154, "get_table_info": 102, "calculator": 53, "respond_user": 31,
          "get_descriptions": 23}
TOOLS = ["get_descriptions", "get_table_info", "sql_query", "calculator", "respond_user"]


def main() -> None:
    pat = "checkpoints_lora/act_prm_snorkel_finance_gym/hf_qwen3_4b_instruct/finance_rollout_*_v3_*/replay_buffer"
    by_arm = {}
    for ck in glob.glob(pat):
        tag = ck.split("/")[-2].split("-act-prm")[0]
        m = re.match(r"finance_rollout_(.+)_v3_(fair|hard)$", tag)
        if not m:
            continue
        key = (m.group(1), m.group(2))
        # newest buffer wins if a tag was re-run
        if key in by_arm and os.path.getmtime(ck) <= os.path.getmtime(by_arm[key]):
            continue
        by_arm[key] = ck

    if not by_arm:
        print("no finance rollout replay buffers yet")
        return

    print("FINANCE ROLLOUT — behaviour (completion is 0 for every arm; see module docstring)\n")
    print(f"  {'arm':24} {'set':5} {'eps':>4} {'answered':>9} " + " ".join(f"{t[:9]:>9}" for t in TOOLS))
    for (arm, setname), ck in sorted(by_arm.items()):
        try:
            ds = load_from_disk(ck)
        except Exception:
            continue
        # A buffer row is one STEP, not one episode, and every step of an episode carries
        # the SAME final_outcome. Counting per row multiplies each episode by its length
        # (120 rows for 10 questions). Dedupe on (sample_id, generation_id) first.
        tools = collections.Counter()
        n_ep = n_ans = 0
        seen = set()
        for row in ds:
            if row.get("is_train"):
                continue
            key = (row.get("sample_id"), row.get("generation_id"))
            if key in seen:
                continue
            seen.add(key)
            n_ep += 1
            answered = False
            for msg in row.get("final_outcome") or []:
                if msg.get("role") != "assistant":
                    continue
                names = re.findall(r'"name"\s*:\s*"([a-z_]+)"', msg.get("content") or "")
                if names:
                    tools[names[0]] += 1
                    if names[0] == "respond_user":
                        answered = True
            n_ans += answered
        tot = sum(tools.values()) or 1
        cells = " ".join(f"{100*tools[t]/tot:8.1f}%" for t in TOOLS)
        print(f"  {arm:24} {setname:5} {n_ep:4} {100*n_ans/max(1,n_ep):8.1f}% {cells}")

    tot = sum(EXPERT.values())
    print(f"\n  {'EXPERT (reference)':24} {'--':5} {'--':>4} {'100.0%':>9} "
          + " ".join(f"{100*EXPERT.get(t,0)/tot:8.1f}%" for t in TOOLS))
    print("\n  answered = episode called respond_user before the turn cap.")
    print("  Percentages are share of that arm's tool calls, so they sum to 100 across the row.")


if __name__ == "__main__":
    main()
