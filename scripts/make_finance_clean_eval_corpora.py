#!/usr/bin/env python
"""Filter each finance arm's eval corpus to questions the SFT never trained on.

The aprm finance split partitioned uids, but many uids share a finqa_reasoning question,
so 16 of act_prm_eval's 21 questions (76%) also sit in act_prm_train. Every finance
Stage-2 eval number was therefore measured largely on memorised questions.

No retraining is required to fix this -- the checkpoints are fine, the measurement was
wrong. This writes <corpus>_cleaneval/eval.json keeping only trajectories whose question
was never in act_prm_train, so the existing checkpoints can simply be re-scored.

Usage: uv run --no-project python scripts/make_finance_clean_eval_corpora.py
"""
import csv, json, os, re

CLEAN = set(json.load(open("data/splits/snorkel_finance_unseen_by_sft.json"))["teacher_forced"])
FQ = {r["question"].strip(): str(r["id"])
      for r in csv.DictReader(open("data/snorkel_finance/benchmark/finqa_reasoning.csv"))}


def qid_of(msgs):
    for m in msgs:
        if m.get("role") == "user":
            c = m["content"]
            mm = re.search(r"Here is the question\s*:\s*(.+?),\s*Here are", c, re.S)
            return FQ.get((mm.group(1) if mm else c[:120]).strip())
    return None


SRC = [("data/snorkel_finance_split", "actions_only"),
       ("data/snorkel_finance_split_expert_thoughts", "expert_thoughts"),
       ("data/sft_corpus/snorkel_finance_split/policy", "thoughts_policy"),
       ("data/sft_corpus/snorkel_finance_split/base", "thoughts_base")]

for src, arm in SRC:
    p = os.path.join(src, "eval.json")
    if not os.path.exists(p):
        print(f"  {arm:16} MISSING {p}"); continue
    data = json.load(open(p))
    keep = [ex for ex in data if qid_of(ex["messages"]) in CLEAN]
    dst = src.rstrip("/") + "_cleaneval"
    os.makedirs(dst, exist_ok=True)
    json.dump(keep, open(os.path.join(dst, "eval.json"), "w"))
    # eval-only corpus; carry a token train.json so loaders that expect it don't trip
    tp = os.path.join(src, "train.json")
    if os.path.exists(tp):
        json.dump(json.load(open(tp)), open(os.path.join(dst, "train.json"), "w"))
    steps = sum(1 for e in keep for m in e["messages"] if m["role"] == "assistant")
    qs = sorted({qid_of(e["messages"]) for e in keep}, key=lambda x: int(x) if x else -1)
    print(f"  {arm:16} {len(data):3} -> {len(keep):3} trajectories, {steps:4} asst steps, questions={qs}")
    json.dump({"filtered_from": src, "clean_question_ids": sorted(CLEAN, key=int),
               "note": "eval trajectories whose finqa question was never in act_prm_train"},
              open(os.path.join(dst, "meta.json"), "w"), indent=1)
