#!/usr/bin/env python3
"""Emit assets/js/lenpen-curve-data.js for the blog's interactive Result-2 viz:
10 held-out tasks × the saved checkpoints of the λ=0.15 run (first step each),
generated post-hoc by scripts/gen_checkpoint_evals.py. The gray context line
(eval mean at every iteration) comes from the run log.

Usage:
  python3 scripts/make_curve_data.py \
      [runs/ckpt_evals_lam015.json] [runs/length_penalty_qwen3_8b_100.json]
"""
import json
import sys
from pathlib import Path

EVALS = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/ckpt_evals_lam015.json")
LOG = Path(sys.argv[2] if len(sys.argv) > 2 else "runs/length_penalty_qwen3_8b_100.json")
OUT = Path("assets/js/lenpen-curve-data.js")

evals = json.loads(EVALS.read_text())
log = json.loads(LOG.read_text())

data = dict(
    run=evals["run"],
    checkpoints=evals["checkpoints"],
    final_iter=evals["checkpoints"][-1],
    eval_mean=[dict(i=it["iteration"], p=round(it["eval"]["mean_likelihood"], 4))
               for it in log["iterations"]],
    tasks=[dict(label="held-out", **t) for t in evals["tasks"]],
)

OUT.write_text("window.LENPEN_CURVE = " + json.dumps(data, ensure_ascii=False) + ";\n")
print(f"wrote {OUT} ({OUT.stat().st_size/1024:.0f} KB): {len(data['tasks'])} held-out tasks × "
      f"{len(data['checkpoints'])} checkpoints")
