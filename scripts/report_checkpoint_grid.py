#!/usr/bin/env python
"""Per-snapshot selection grid for a Stage-2 run: every viable criterion side by side.

step_best is chosen on eval action-token PPL, but the 08-31 rollout analysis showed the
PPL-optimal checkpoint is NOT the rollout-optimal one -- fitting the SFT target
distribution (bare tool calls, observations hidden) completely is what destroys the
user-directed turns tau2 needs. So we track several criteria per snapshot and let the
choice be explicit:

  eval_actiononly_ppl / accuracy : the SFT objective (what step_best uses)
  median |B@A|                   : how far the TYPICAL adapter coordinate moved. This is
                                   the statistic that separated the conversational SGD
                                   checkpoint (4.7e-05) from the collapsed AdamW ones
                                   (1.5-2.2e-03); peak movement did NOT separate them.
  max |B@A|                      : peak induced weight delta (base weights ~1e-2)

Prose rate needs generation and so is not computed here; measure it with a short rollout
on the snapshots this grid shortlists.

Usage: uv run --no-project python scripts/report_checkpoint_grid.py --run <run_tag_glob>
"""
import argparse
import glob
import json
import os
import re

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True, help="run-tag glob, e.g. 'retail_s2_actions_only_lr1e_4_*'")
ap.add_argument("--env", default="act_prm_tau2_retail")
ap.add_argument("--model", default="hf_qwen3_4b_instruct")
args = ap.parse_args()


def movement(ckpt):
    """max/median |B@A| for one snapshot, via the existing reporter."""
    import subprocess
    out = subprocess.run(
        ["uv", "run", "--no-project", "python", "scripts/report_lora_movement.py", "--glob", ckpt],
        capture_output=True, text=True).stdout
    m = re.search(r"(\d\.\d+e[-+]\d+)\s+(\d\.\d+e[-+]\d+)\s+(\d\.\d+e[-+]\d+)", out)
    return (m.group(2), m.group(3)) if m else ("?", "?")


logs = sorted(glob.glob(f"logs/{args.env}/{args.model}/{args.run}/"), key=os.path.getmtime)
if not logs:
    raise SystemExit(f"no run matching {args.run}")
d = logs[-1]
tag = os.path.basename(d.rstrip("/"))
ck = f"checkpoints_lora/{args.env}/{args.model}/{tag}"

evals = {}
for line in open(d + "metrics.jsonl"):
    line = line.strip()
    if not line:
        continue
    r = json.loads(line)
    if "eval/eval_actiononly_ppl" in r and r.get("progress/batch") is not None:
        evals[r["progress/batch"]] = (r["eval/eval_actiononly_ppl"],
                                      r.get("eval/eval_actiononly_accuracy", 0.0))

print(f"run: {tag[:78]}")
print(f"{'snapshot':<12} {'batch':>6} {'eval ppl':>9} {'eval acc':>9} {'max|B@A|':>10} {'median|B@A|':>12}")
snaps = sorted(glob.glob(f"{ck}/step_[0-9]*")) + [f"{ck}/step_best", f"{ck}/step_last"]
for s in snaps:
    if not os.path.isdir(s):
        continue
    name = os.path.basename(s)
    b = int(name.split("_")[1]) if name.split("_")[1].isdigit() else None
    e = evals.get(b) if b is not None else None
    if e is None and b is not None:  # snapshots land between eval ticks
        near = [k for k in evals if abs(k - b) <= 5]
        e = evals[min(near, key=lambda k: abs(k - b))] if near else None
    mx, med = movement(s)
    ppl = f"{e[0]:.4f}" if e else "-"
    acc = f"{e[1]:.4f}" if e else "-"
    print(f"{name:<12} {str(b or '-'):>6} {ppl:>9} {acc:>9} {mx:>10} {med:>12}")
print("\n  reference: SGD 3e-3 step_best (converses, 21.4% rollout) had median |B@A| = 4.7e-05;"
      "\n             AdamW 1e-3 step_0020 (0% prose) = 1.46e-03, step_best = 2.19e-03.")
