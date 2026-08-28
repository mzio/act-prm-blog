#!/usr/bin/env python
"""Dump the assistant generations from a rollout eval's replay buffer.

The rollout evals persist full trajectories (is_train=False) to
<ckpt>/replay_buffer, including every assistant message. This reads them back so the
generations can be inspected directly, rather than inferred from token counts.

Usage: uv run --no-project python scripts/dump_rollout_generations.py <domain> [--n 2]
"""
import argparse, glob, json, re, sys
from datasets import load_from_disk
from act_prm.environments.act_prm_traces.data import extract_action

ap = argparse.ArgumentParser()
ap.add_argument("domain")
ap.add_argument("--n", type=int, default=1, help="example assistant turns to print per arm")
ap.add_argument("--arms", default="actions_only,expert_thoughts,thoughts_policy,thoughts_base")
args = ap.parse_args()

for arm in args.arms.split(","):
    pat = (f"checkpoints_lora/tau2bench_{args.domain}_rlvr/hf_qwen3_4b_instruct/"
           f"{args.domain}_rollout_{arm}_lr3e_3-*/replay_buffer")
    dirs = sorted(glob.glob(pat))
    if not dirs:
        print(f"\n### {arm}: no replay buffer"); continue
    d = load_from_disk(dirs[0])
    thought_lens, n_asst, n_thought, samples = [], 0, 0, []
    n_action = n_notool = 0
    for row in d:
        if row.get("is_train"):
            continue
        for m in row.get("final_outcome") or []:
            if m.get("role") != "assistant":
                continue
            c = m.get("content") or ""
            n_asst += 1
            a = extract_action(c)
            if not a or c.find(a) < 0:
                # A turn with no tool call is the agent TALKING to the user, not thinking.
                # Counting its whole content as a "thought" (the earlier bug) conflates
                # customer-facing prose with pre-action reasoning.
                n_notool += 1
                continue
            n_action += 1
            pre = c[:c.find(a)].strip()
            thought_lens.append(len(pre))
            if len(pre) >= 10:
                n_thought += 1
                if len(samples) < args.n:
                    samples.append(c)
    if not n_asst:
        print(f"\n### {arm}: no assistant turns in buffer"); continue
    print(f"\n### {args.domain} / {arm}")
    print(f"  assistant turns: {n_asst}   tool-call turns: {n_action}   talk-only turns: {n_notool}")
    if not thought_lens:
        print("  no tool-call turns"); continue
    mean = sum(thought_lens)/len(thought_lens)
    med = sorted(thought_lens)[len(thought_lens)//2]
    print(f"  of tool-call turns, reasoning BEFORE the call: {n_thought} ({100*n_thought/n_action:.0f}%)"
          f"   chars mean={mean:.0f} median={med}")
    for s in samples:
        print(f"  --- sample generation ---\n{s[:700]}")
