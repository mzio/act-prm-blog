#!/usr/bin/env python
"""Question-level finance split (v3) + per-arm SFT pools built by subselection.

Background: only 51 of the 79 finqa_reasoning questions have a SUCCESSFUL expert rollout
(the split's "usable" filter is done + reward>0). The other 28 have 123 trajectories, all
reward=0 -- questions GPT-5-mini failed. The v1 split partitioned uids, but ~3 uids share
each question, so questions leaked across splits (76% of eval, 91% of rl_eval).

v3 partitions the 51 demo-bearing QUESTIONS into train / eval, then filters each arm's
existing pool by question. No Stage-1 relabel and no new generation: pure subselection.

Eval sets this enables:
  - teacher-forced + rollout on the 10 eval questions  (fair: expert solved these)
  - rollout on the 28 expert-failure questions          (hard test, labelled as such)

Usage: uv run --no-project python scripts/make_finance_pools_v3.py [--n-eval 10]
"""
import argparse, csv, json, os, random, re

ap = argparse.ArgumentParser()
ap.add_argument("--n-eval", type=int, default=10)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

FQ = {r["question"].strip(): str(r["id"])
      for r in csv.DictReader(open("data/snorkel_finance/benchmark/finqa_reasoning.csv"))}
ALLQ = set(FQ.values())


def qid(msgs):
    for m in msgs:
        if m.get("role") == "u" + "ser":
            c = m["content"]
            mm = re.search(r"Here is the question\s*:\s*(.+?),\s*Here are", c, re.S)
            return FQ.get((mm.group(1) if mm else c[:120]).strip())
    return None


SRC = [("data/snorkel_finance_split", "actions_only"),
       ("data/snorkel_finance_split_expert_thoughts", "expert_thoughts"),
       ("data/sft_corpus/snorkel_finance_split/policy", "thoughts_policy"),
       ("data/sft_corpus/snorkel_finance_split/base", "thoughts_base")]

# every question that appears anywhere in the built pools == the demo-bearing set
demo_q = set()
pools = {}
for src, arm in SRC:
    pools[arm] = {s: json.load(open(f"{src}/{s}.json")) for s in ("train", "eval")}
    for s in ("train", "eval"):
        for ex in pools[arm][s]:
            q = qid(ex["messages"])
            if q:
                demo_q.add(q)
demo_q = sorted(demo_q, key=int)

rng = random.Random(args.seed)
sh = demo_q[:]
rng.shuffle(sh)
q_eval = set(sh[:args.n_eval])
q_train = set(sh[args.n_eval:])
hard = sorted(ALLQ - set(demo_q), key=int)      # expert-failure questions
assert not (q_eval & q_train)

print(f"  demo-bearing questions: {len(demo_q)}  ->  train {len(q_train)} / eval {len(q_eval)}")
print(f"  expert-FAILURE questions (rollout hard test): {len(hard)}")

for src, arm in SRC:
    dst = src.rstrip("/") + "_v3"
    os.makedirs(dst, exist_ok=True)
    out = {"train": [], "eval": []}
    for s in ("train", "eval"):
        for ex in pools[arm][s]:
            q = qid(ex["messages"])
            if q in q_train:
                out["train"].append(ex)
            elif q in q_eval:
                out["eval"].append(ex)
    for s in ("train", "eval"):
        json.dump(out[s], open(f"{dst}/{s}.json", "w"))
    st = lambda d: sum(1 for e in d for m in e["messages"] if m["role"] == "assistant")
    print(f"   {arm:16} train {len(out['train']):3} traj / {st(out['train']):4} steps"
          f"   eval {len(out['eval']):3} traj / {st(out['eval']):4} steps")
    json.dump({"note": "v3 question-level subselection of the existing pool; no relabel",
               "train_questions": sorted(q_train, key=int), "eval_questions": sorted(q_eval, key=int)},
              open(f"{dst}/meta.json", "w"), indent=1)

json.dump({"note": "v3 QUESTION-level finance split. 51 questions have a successful expert "
                   "rollout; the other 28 have only reward=0 trajectories (expert failures) and "
                   "are reserved as a labelled hard rollout test.",
           "n_demo_questions": len(demo_q), "train_questions": sorted(q_train, key=int),
           "eval_questions": sorted(q_eval, key=int), "hard_rollout_questions": hard},
          open("data/splits/snorkel_finance_v3.json", "w"), indent=1)
print("  wrote data/splits/snorkel_finance_v3.json")
