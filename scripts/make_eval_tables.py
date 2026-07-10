#!/usr/bin/env python3
"""Emit markdown tables showing, for fixed held-out eval steps, the *selected*
thought (and its action-likelihood / length) at periodic EM iterations — so you
can watch the same thought slot evolve over training.

Usage:
  uv run --with datasets --with numpy python scripts/make_eval_tables.py \
      [runs/length_penalty_qwen3_8b_100.json] > tables.md
"""
import importlib.util
import json
import sys
from pathlib import Path

LOG = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/length_penalty_qwen3_8b_100.json")
EVERY = 10                      # show every Nth iteration (plus first + last)
N_SAMPLES = 5                   # first N steps of eval trajectory 0
TRUNC = 320                     # max chars of thought text per cell

log = json.loads(LOG.read_text())
cfg = log["config"]

# reload the same trajectories the run used (loader is deterministic)
spec = importlib.util.spec_from_file_location(
    "lp", Path(__file__).parent / "act_prm_length_penalty.py")
lp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lp)
trajs = lp.load_trajectories(cfg["num_trajectories"] + cfg["eval_trajectories"],
                             cfg["max_traj_timestep"])
eval_traj = trajs[cfg["num_trajectories"]]          # eval trajectory 0
actions = [m["content"] for m in eval_traj["messages"] if m["role"] == "assistant"]

iters = [it["iteration"] for it in log["iterations"]]
show = sorted({iters[0], *[i for i in iters if i % EVERY == 0], iters[-1]})
by_iter = {it["iteration"]: it for it in log["iterations"]}


def md_escape(s):
    return s.replace("|", "\\|").replace("\n", " ").strip()


def cell(s, n=TRUNC):
    s = md_escape(s)
    return s if len(s) <= n else s[: n - 1] + "…"


print(f"_Same held-out trajectory, every {EVERY} EM iterations; "
      f"\"selected ẑ\" is the penalized-reward argmax of G={cfg['group_size']} "
      f"sampled thoughts (the one committed to context)._\n")
print(f"**Eval question:** {cell(eval_traj['messages'][0]['content'], 400)}\n")

for step in range(N_SAMPLES):
    print(f"\n#### Step {step + 1} — logged action")
    print(f"\n```\n{actions[step].strip()}\n```\n")
    print("| EM iter | p(x\\|s,ẑ) | \\|ẑ\\| tokens | selected thought ẑ |")
    print("|---:|---:|---:|---|")
    for i in show:
        it = by_iter.get(i)
        if it is None or len(it["eval_metrics"]) == 0:
            continue
        steps = it["eval_metrics"][0]           # eval trajectory 0
        if step >= len(steps):
            continue
        m = steps[step]
        b = m["best"]
        print(f"| {i} | {m['likelihoods'][b]:.4f} | {m['thought_tokens'][b]} "
              f"| {cell(m['thoughts'][b])} |")
