#!/usr/bin/env python3
"""Emit assets/js/lenpen-curve-data.js for the blog's interactive Result-2 viz:
for each held-out eval task and step, the selected thought + its likelihood at
EVERY EM iteration of the (λ=0.15) run, plus observations and ground-truth
actions from the trajectories.

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
eval_trajs = trajs[cfg["num_trajectories"]:]


def trunc(s, n):
    s = " ".join(s.split())
    return s if len(s) <= n else s[: n - 2] + " …"


tasks = []
for t, traj in enumerate(eval_trajs):
    msgs = traj["messages"]
    a_idx = [i for i, m in enumerate(msgs) if m["role"] == "assistant"]
    a_idx = a_idx[: cfg["max_steps_per_traj"]]
    steps = []
    for s_i, idx in enumerate(a_idx):
        # the observation is the message right before this action
        obs_msg = msgs[idx - 1] if idx > 0 else msgs[0]
        obs_kind = "prompt" if s_i == 0 else ("tool response" if obs_msg["role"] in ("tool",) else "observation")
        per_iter = []
        for it in log["iterations"]:
            em = it["eval_metrics"]
            if t >= len(em) or s_i >= len(em[t]):
                continue
            m = em[t][s_i]
            b = m["best"]
            per_iter.append(dict(
                i=it["iteration"],
                p=round(m["likelihoods"][b], 4),
                tok=m["thought_tokens"][b],
                z=trunc(m["thoughts"][b], THOUGHT_TRUNC),
            ))
        steps.append(dict(
            obs=trunc(obs_msg["content"], OBS_TRUNC),
            obs_kind=obs_kind,
            action=msgs[idx]["content"].strip(),
            per_iter=per_iter,
        ))
    tasks.append(dict(
        label=f"Task {t + 1}",
        question=trunc(msgs[0]["content"], OBS_TRUNC),
        steps=steps,
    ))

data = dict(
    run=str(LOG.name),
    final_iter=log["iterations"][-1]["iteration"],
    eval_mean=[dict(i=it["iteration"], p=round(it["eval"]["mean_likelihood"], 4))
               for it in log["iterations"]],
    tasks=tasks,
)

OUT.write_text("window.LENPEN_CURVE = " + json.dumps(data, ensure_ascii=False) + ";\n")
size = OUT.stat().st_size
print(f"wrote {OUT} ({size/1024:.0f} KB): {len(tasks)} tasks × "
      f"{len(tasks[0]['steps'])} steps × {len(tasks[0]['steps'][0]['per_iter'])} iterations")
