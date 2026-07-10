#!/usr/bin/env python3
"""Emit markdown tables showing how the *selected* thought for fixed held-out
eval steps evolves over training.

Default view: the FIRST step of EVERY eval task (different questions, same step
position) — one table per task. Pass --per-step N to instead show the first N
sequential steps of eval task 0 (same question, evolving context).

Usage:
  uv run --with datasets --with numpy --with "transformers>=4.51" \
         --with python-dotenv --with tinker --with jinja2 \
    python scripts/make_eval_tables.py runs/<log>.json [--every 10] [--per-step N]
"""
import argparse
import importlib.util
import json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("log", nargs="?", default="runs/length_penalty_qwen3_8b_100.json")
ap.add_argument("--every", type=int, default=10, help="show every Nth iteration")
ap.add_argument("--per-step", type=int, default=0,
                help="if >0: show first N steps of eval task 0 instead of step 1 of every task")
ap.add_argument("--trunc", type=int, default=320, help="max chars of thought text per cell")
args = ap.parse_args()

log = json.loads(Path(args.log).read_text())
cfg = log["config"]

# reload the same trajectories the run used (loader is deterministic)
spec = importlib.util.spec_from_file_location(
    "lp", Path(__file__).parent / "act_prm_length_penalty.py")
lp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lp)
trajs = lp.load_trajectories(cfg["num_trajectories"] + cfg["eval_trajectories"],
                             cfg["max_traj_timestep"])
eval_trajs = trajs[cfg["num_trajectories"]:]

iters = [it["iteration"] for it in log["iterations"]]
show = sorted({iters[0], *[i for i in iters if i % args.every == 0], iters[-1]})
by_iter = {it["iteration"]: it for it in log["iterations"]}


def md_escape(s):
    return s.replace("|", "\\|").replace("\n", " ").strip()


def cell(s, n=None):
    s = md_escape(s)
    n = n or args.trunc
    return s if len(s) <= n else s[: n - 1] + "…"


def emit_table(traj_idx, step):
    print("| EM iter | p(x\\|s,ẑ) | \\|ẑ\\| tokens | selected thought ẑ |")
    print("|---:|---:|---:|---|")
    for i in show:
        it = by_iter.get(i)
        if it is None or traj_idx >= len(it["eval_metrics"]):
            continue
        steps = it["eval_metrics"][traj_idx]
        if step >= len(steps):
            continue
        m = steps[step]
        b = m["best"]
        print(f"| {i} | {m['likelihoods'][b]:.4f} | {m['thought_tokens'][b]} "
              f"| {cell(m['thoughts'][b])} |")


print(f"_Held-out tasks, every {args.every} EM iterations; \"selected ẑ\" is the "
      f"penalized-reward argmax of G={cfg['group_size']} sampled thoughts (the one "
      f"committed to context)._\n")

if args.per_step > 0:
    traj = eval_trajs[0]
    actions = [m["content"] for m in traj["messages"] if m["role"] == "assistant"]
    print(f"**Eval question:** {cell(traj['messages'][0]['content'], 400)}\n")
    for step in range(min(args.per_step, len(actions))):
        print(f"\n#### Step {step + 1} — logged action\n")
        print(f"```\n{actions[step].strip()}\n```\n")
        emit_table(0, step)
else:
    for t, traj in enumerate(eval_trajs):
        actions = [m["content"] for m in traj["messages"] if m["role"] == "assistant"]
        print(f"\n#### Task {t + 1}")
        print(f"\n**Question:** {cell(traj['messages'][0]['content'], 400)}\n")
        print(f"**Logged first action:**\n\n```\n{actions[0].strip()}\n```\n")
        emit_table(t, 0)
