#!/usr/bin/env python
"""Rebuild the snorkel_finance split at the QUESTION level (v2 / vfix).

The original split partitioned `unique_data_sample_id`s, but many uids map to the SAME
underlying finqa_reasoning question, so questions leaked across splits:
  act_prm_eval  21 questions, 16 of them (76%) also in act_prm_train
  rl_eval       22 questions, 20 of them (91%) also in act_prm_train
Every finance Stage-2 eval number is therefore measured largely on memorised questions,
and the RL hold-out was never held out.

v2 partitions the 79 questions first, then assigns each uid to the split of its question,
so no question can appear in two splits. Deterministic (sorted ids + fixed seed).

Usage: uv run --no-project python scripts/make_finance_split_v2.py [--eval-frac .2] [--rl-frac .2]
"""
import argparse, json, random

ap = argparse.ArgumentParser()
ap.add_argument("--uid-map", default="data/splits/snorkel_finance_uid_to_qid.json")
ap.add_argument("--old-split", default="data/splits/snorkel_finance.json")
ap.add_argument("--out", default="data/splits/snorkel_finance_v2.json")
ap.add_argument("--eval-frac", type=float, default=0.3)
ap.add_argument("--rl-frac", type=float, default=0.0,
                help="Reserve from the 51 demo-bearing questions. Default 0: the 28 questions with "
                     "no usable trajectories are a strictly cleaner RL/rollout hold-out (never seen "
                     "by EM, SFT or anything else), so there is no reason to spend demo-bearing "
                     "questions on it.")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

u2q = {int(k): v for k, v in json.load(open(args.uid_map)).items()}
old = json.load(open(args.old_split))
usable = [u for name in ("act_prm_train", "act_prm_eval", "rl_eval") for u in old[name]]
qids = sorted({u2q[u] for u in usable if u in u2q}, key=int)

rng = random.Random(args.seed)
shuffled = qids[:]
rng.shuffle(shuffled)
n = len(shuffled)
n_eval = max(1, round(n * args.eval_frac))
n_rl = round(n * args.rl_frac)
q_eval = set(shuffled[:n_eval])
q_rl = set(shuffled[n_eval:n_eval + n_rl])
q_train = set(shuffled[n_eval + n_rl:])
assert not (q_eval & q_rl) and not (q_eval & q_train) and not (q_rl & q_train)

out = {"act_prm_train": [], "act_prm_eval": [], "rl_eval": []}
for u in sorted(set(usable)):
    q = u2q.get(u)
    if q is None:
        continue
    out["act_prm_train" if q in q_train else "act_prm_eval" if q in q_eval else "rl_eval"].append(u)

res = {
    "dataset": old["dataset"], "name": "snorkel_finance_v2", "seed": args.seed,
    "note": "QUESTION-LEVEL split. The v1 split partitioned uids, but many uids share a "
            "finqa_reasoning question, leaking 76% of act_prm_eval and 91% of rl_eval into "
            "act_prm_train. Here questions are partitioned first and uids follow their question.",
    "question_counts": {"total": n, "train": len(q_train), "eval": len(q_eval), "rl_eval": len(q_rl)},
    "questions": {"train": sorted(q_train, key=int), "eval": sorted(q_eval, key=int), "rl_eval": sorted(q_rl, key=int)},
    "counts": {"total_usable_tasks": len(set(usable)), **{k: len(v) for k, v in out.items()}},
    **out,
}
json.dump(res, open(args.out, "w"), indent=1)
print(f"  questions: {n} total -> train {len(q_train)} / eval {len(q_eval)} / rl_eval {len(q_rl)}")
print(f"  uids:      {len(set(usable))} total -> train {len(out['act_prm_train'])} / "
      f"eval {len(out['act_prm_eval'])} / rl_eval {len(out['rl_eval'])}")
print(f"  wrote {args.out}")
