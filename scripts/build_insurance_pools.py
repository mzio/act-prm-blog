#!/usr/bin/env python
"""Build Act-PRM pools + a QUESTION-level split from the GPT-5-mini insurance rollouts.

Source: mzio/aprm-insurance-gpt5m_med-gs4-s0-r1-train -- a STEP-level replay buffer
(13,131 steps / 1,000 trajectories / 262 questions, group_size 4). Trajectories are
reconstructed by grouping on (unique_data_sample_id, generation_id) and ordering by
timestep; the terminal step carries reward +1 (success) or -1 (failure).

Only SUCCESSFUL trajectories are kept, per the request: 910/1000 (91%), covering 261 of
262 questions.

The split is by QUESTION, not by trajectory or uid. This is the one hard lesson from
finance, where partitioning `unique_data_sample_id` leaked 76% of the eval questions and
91% of the RL hold-out into train (~2.5 uids share each question), so every finance curve
was scored largely on memorised questions. Here one uid == one question, but the split is
still expressed and verified at question level so the same mistake cannot recur.

Three disjoint sets:
  train   -- Stage-1 EM thought generation + Stage-2 SFT training
  eval    -- Stage-2 held-out action-span PPL / accuracy (the curves)
  rollout -- never seen by Stage 1 or Stage 2; reserved for gym rollout eval

Emits two pools. NOTE the `_split` suffix: data/snorkel_insurance/ is the GYM data root
(resources/, tool_data/, task_data/ from the snorkel benchmark repo), exactly as
data/snorkel_finance/ is for finance. Pools must not be written there.
  data/snorkel_insurance_split/                  actions-only targets (expert baseline)
  data/snorkel_insurance_split_expert_thoughts/  reasoning+action targets (oracle arm)
Each row: {uid, generation_id, system_prompt, messages}.

Usage:
  export https_proxy=http://fwdproxy:8080 http_proxy=http://fwdproxy:8080
  HF_HUB_DISABLE_XET=1 HF_TOKEN=$(cat ~/models/token) \\
    uv run --no-project python scripts/build_insurance_pools.py
"""
import argparse
import collections
import json
import os
import random
import re

REPO = "mzio/aprm-insurance-gpt5m_med-gs4-s0-r1-train"
TOOL_RE = re.compile(r"<tool_call>", re.I)


def split_action(content: str) -> tuple[str, str]:
    """(reasoning_prefix, action). The action starts at the first <tool_call>."""
    m = TOOL_RE.search(content or "")
    if not m:
        return "", content or ""
    return content[: m.start()].rstrip(), content[m.start():]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-eval", type=int, default=40)
    ap.add_argument("--n-rollout", type=int, default=41)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-per-question", type=int, default=0,
                    help="cap successful trajectories kept per question (0 = keep all)")
    args = ap.parse_args()

    from datasets import load_dataset

    d = load_dataset(REPO, split="train")

    # 1. reconstruct trajectories
    traj = collections.defaultdict(list)
    for i in range(len(d)):
        r = d[i]
        traj[(r["unique_data_sample_id"], r["generation_id"])].append(r)
    for k in traj:
        traj[k].sort(key=lambda r: r["timestep"])

    successful = {k: v for k, v in traj.items() if v and v[-1]["done"] and v[-1]["reward"] > 0}
    print(f"trajectories: {len(traj)}   successful: {len(successful)} "
          f"({100*len(successful)/len(traj):.1f}%)")

    by_q = collections.defaultdict(list)
    for (uid, gid), steps in successful.items():
        by_q[uid].append((gid, steps))
    questions = sorted(by_q)
    print(f"questions with >=1 success: {len(questions)}")

    # 2. QUESTION-level split
    rng = random.Random(args.seed)
    shuffled = questions[:]
    rng.shuffle(shuffled)
    n_ev, n_ro = args.n_eval, args.n_rollout
    assert n_ev + n_ro < len(shuffled), "split larger than the question pool"
    ev_q = sorted(shuffled[:n_ev])
    ro_q = sorted(shuffled[n_ev:n_ev + n_ro])
    tr_q = sorted(shuffled[n_ev + n_ro:])
    assert not (set(tr_q) & set(ev_q)) and not (set(tr_q) & set(ro_q)) and not (set(ev_q) & set(ro_q))
    print(f"split by question -> train {len(tr_q)}  eval {len(ev_q)}  rollout {len(ro_q)}")

    # 3. materialise pools
    def build(qids, keep_thoughts: bool):
        rows, n_steps, n_thought = [], 0, 0
        for q in qids:
            gens = sorted(by_q[q])
            if args.max_per_question:
                gens = gens[: args.max_per_question]
            for gid, steps in gens:
                msgs = list(steps[0]["state"])  # the user turn(s) that open the episode
                for st in steps:
                    content = (st["action"] or {}).get("content") or ""
                    reasoning, action = split_action(content)
                    n_steps += 1
                    if reasoning:
                        n_thought += 1
                    msgs.append({"role": "assistant",
                                 "content": content if keep_thoughts else action})
                    for obs in st["next_obs"] or []:
                        msgs.append({"role": obs.get("role", "tool"),
                                     "content": obs.get("content") or ""})
                rows.append({"uid": q, "generation_id": gid,
                             "system_prompt": steps[0]["system_prompt"], "messages": msgs})
        return rows, n_steps, n_thought

    for name, keep in [("data/snorkel_insurance_split", False),
                       ("data/snorkel_insurance_split_expert_thoughts", True)]:
        os.makedirs(name, exist_ok=True)
        for split, qids in [("train", tr_q), ("eval", ev_q)]:
            rows, n_steps, n_thought = build(qids, keep)
            json.dump(rows, open(f"{name}/{split}.json", "w"))
            extra = f"  reasoning-bearing {n_thought}/{n_steps} ({100*n_thought/max(1,n_steps):.0f}%)" if keep else ""
            print(f"  {name}/{split}.json: {len(rows)} traj, {n_steps} steps{extra}")

    # 4. split file
    os.makedirs("data/splits", exist_ok=True)
    out = {
        "note": ("QUESTION-level insurance split from the GPT-5-mini rollouts. Only "
                 "SUCCESSFUL trajectories (terminal reward > 0) are used: 910/1000 (91%), "
                 "covering 261/262 questions. train/eval/rollout are disjoint by question; "
                 "rollout is never seen by Stage-1 EM or Stage-2 SFT."),
        "source": REPO,
        "seed": args.seed,
        "n_questions_with_success": len(questions),
        "train_questions": tr_q,
        "eval_questions": ev_q,
        "rollout_questions": ro_q,
    }
    json.dump(out, open("data/splits/snorkel_insurance.json", "w"), indent=2)
    print(f"\nwrote data/splits/snorkel_insurance.json "
          f"(train {len(tr_q)} / eval {len(ev_q)} / rollout {len(ro_q)} questions)")


if __name__ == "__main__":
    main()
