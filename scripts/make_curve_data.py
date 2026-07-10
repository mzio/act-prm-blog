#!/usr/bin/env python3
"""Emit assets/js/lenpen-curve-data.js for the blog's interactive Result-2 viz:
the FIRST step of every task (2 held-out + 8 train = 10 samples), with the
selected thought + likelihood at every EM iteration.

Usage:
  uv run --with datasets --with numpy --with "transformers>=4.51" \
         --with python-dotenv --with tinker --with jinja2 \
    python scripts/make_curve_data.py [runs/length_penalty_qwen3_8b_100.json]
"""
import importlib.util
import json
import sys
from pathlib import Path

LOG = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/length_penalty_qwen3_8b_100.json")
OUT = Path("assets/js/lenpen-curve-data.js")
THOUGHT_TRUNC = 300
OBS_TRUNC = 380

log = json.loads(LOG.read_text())
cfg = log["config"]

spec = importlib.util.spec_from_file_location(
    "lp", Path(__file__).parent / "act_prm_length_penalty.py")
lp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lp)
trajs = lp.load_trajectories(cfg["num_trajectories"] + cfg["eval_trajectories"],
                             cfg["max_traj_timestep"])


def trunc(s, n):
    s = " ".join(s.split())
    return s if len(s) <= n else s[: n - 2] + " …"


def task_entry(traj, metrics_key, traj_idx, label):
    msgs = traj["messages"]
    first_action_idx = next(i for i, m in enumerate(msgs) if m["role"] == "assistant")
    per_iter = []
    for it in log["iterations"]:
        em = it[metrics_key]
        if traj_idx >= len(em) or not em[traj_idx]:
            continue
        m = em[traj_idx][0]                      # step 1
        b = m["best"]
        per_iter.append(dict(
            i=it["iteration"],
            p=round(m["likelihoods"][b], 4),
            tok=m["thought_tokens"][b],
            z=trunc(m["thoughts"][b], THOUGHT_TRUNC),
        ))
    return dict(
        label=label,
        question=trunc(msgs[0]["content"], OBS_TRUNC),
        action=msgs[first_action_idx]["content"].strip(),
        per_iter=per_iter,
    )


n_train = cfg["num_trajectories"]
tasks = []
for t in range(cfg["eval_trajectories"]):
    tasks.append(task_entry(trajs[n_train + t], "eval_metrics", t, "held-out"))
for t in range(n_train):
    tasks.append(task_entry(trajs[t], "train_metrics", t, "train"))

data = dict(
    run=str(LOG.name),
    final_iter=log["iterations"][-1]["iteration"],
    eval_mean=[dict(i=it["iteration"], p=round(it["eval"]["mean_likelihood"], 4))
               for it in log["iterations"]],
    tasks=tasks,
)

OUT.write_text("window.LENPEN_CURVE = " + json.dumps(data, ensure_ascii=False) + ";\n")
print(f"wrote {OUT} ({OUT.stat().st_size/1024:.0f} KB): {len(tasks)} tasks "
      f"({cfg['eval_trajectories']} held-out + {n_train} train), step 1, "
      f"{len(tasks[0]['per_iter'])} iterations each")
