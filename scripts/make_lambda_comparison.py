#!/usr/bin/env python3
"""Cross-λ comparison tables: for shared held-out tasks, show the selected
thought under each length-penalty λ at periodic EM iterations.

Layout per task: the prompt + step-1 logged action once, then a table with one
row per iteration and one column per λ (each cell: p(x|s,ẑ), |ẑ| tokens, and
the thought text).

Usage:
  uv run --with datasets --with numpy --with "transformers>=4.51" \
         --with python-dotenv --with tinker --with jinja2 \
    python scripts/make_lambda_comparison.py \
      --run 0.15=runs/length_penalty_qwen3_8b_100.json \
      --run 0.4=runs/length_penalty_qwen3_8b_lam04.json \
      --run 1.0=runs/length_penalty_qwen3_8b_lam10.json \
      [--every 10] [--tasks 2] [--trunc 220]
"""
import argparse
import importlib.util
import json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--run", action="append", required=True,
                help="LABEL=path/to/log.json (repeatable, in display order)")
ap.add_argument("--every", type=int, default=10)
ap.add_argument("--tasks", type=int, default=2,
                help="number of shared held-out tasks to show")
ap.add_argument("--trunc", type=int, default=220)
args = ap.parse_args()

runs = []
for spec_str in args.run:
    label, path = spec_str.split("=", 1)
    runs.append((label, json.loads(Path(path).read_text())))

# all runs share the same deterministic loader ordering; eval tasks start at
# num_trajectories (same across runs — assert to be safe)
cfgs = [log["config"] for _, log in runs]
assert len({c["num_trajectories"] for c in cfgs}) == 1, "runs disagree on train-set size"
n_train = cfgs[0]["num_trajectories"]
n_eval_shared = min(c["eval_trajectories"] for c in cfgs)

spec = importlib.util.spec_from_file_location(
    "lp", Path(__file__).parent / "act_prm_length_penalty.py")
lp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lp)
trajs = lp.load_trajectories(n_train + max(c["eval_trajectories"] for c in cfgs),
                             cfgs[0]["max_traj_timestep"])
eval_trajs = trajs[n_train: n_train + min(args.tasks, n_eval_shared)]

# iteration grid: union of "every N" points across runs, plus each run's last iter
max_iter = max(log["iterations"][-1]["iteration"] for _, log in runs)
grid = sorted({i for i in range(0, max_iter + 1, args.every)}
              | {log["iterations"][-1]["iteration"] for _, log in runs})

by_iter = {label: {it["iteration"]: it for it in log["iterations"]} for label, log in runs}
last_iter = {label: log["iterations"][-1]["iteration"] for label, log in runs}


def md_escape(s):
    return s.replace("|", "\\|").replace("\n", " ").strip()


def cell(label, i, task_idx):
    """Thought cell for run `label` at iteration `i` (exact match only)."""
    it = by_iter[label].get(i)
    if it is None:
        return "—" if i > last_iter[label] else "·"
    if task_idx >= len(it["eval_metrics"]) or not it["eval_metrics"][task_idx]:
        return "·"
    m = it["eval_metrics"][task_idx][0]          # step 1 of the task
    b = m["best"]
    text = md_escape(m["thoughts"][b])
    if len(text) > args.trunc:
        text = text[: args.trunc - 1] + "…"
    p = m["likelihoods"][b]
    p_str = f"{p:.4f}" if p < 0.01 else f"{p:.2f}"
    return f"p={p_str}, {m['thought_tokens'][b]} tok — “{text}”"


labels = [label for label, _ in runs]
print(f"_Selected thought ẑ (penalized-reward argmax of G={cfgs[0]['group_size']}) for the "
      f"**first step** of shared held-out tasks, under different length penalties λ. "
      f"“—” = run already finished (λ=0.15 early-stopped at iter {last_iter[labels[0]]})._\n")

for t, traj in enumerate(eval_trajs):
    actions = [m["content"] for m in traj["messages"] if m["role"] == "assistant"]
    print(f"\n### Task {t + 1}")
    print(f"\n**Prompt:** {md_escape(traj['messages'][0]['content'])[:400]}\n")
    print(f"**Logged action:**\n\n```\n{actions[0].strip()}\n```\n")
    header = "| iter | " + " | ".join(f"λ={la}" for la in labels) + " |"
    print(header)
    print("|---:|" + "---|" * len(labels))
    for i in grid:
        row = [cell(la, i, t) for la in labels]
        if all(r in ("—", "·") for r in row):
            continue
        print(f"| {i} | " + " | ".join(row) + " |")
