#!/usr/bin/env bash
# DURABILITY DAEMON. `logs/` and `checkpoints_lora/` are GITIGNORED, so anything
# written there dies with the box (we already lost a Stage-3 fleet that way). Every
# INTERVAL seconds this:
#   1. re-exports every logs/<env>/<model>/<run>/metrics.jsonl into the TRACKED
#      results/<env>/<run>/metrics.jsonl (+ config.json when present),
#   2. regenerates results/SUMMARY.csv (one row per run: best eval action-subspan
#      ppl/acc, whole-span ppl/acc, batches, RL success rate when present),
#   3. commits locally (NEVER pushes — github.com is unreachable from the devserver),
#   4. refreshes the dotsync copy via scripts/backup_results.sh (adapters + artifacts
#      under ~/.claude, which IS dotsync'd and survives a box move).
#
# crontab is not permitted for this user -> run detached:
#   setsid nohup ./scripts/backup_results_daemon.sh > /tmp/backup_results.daemon.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
INTERVAL="${1:-1800}"

export_once() {
  .venv/bin/python - <<'PY'
import csv, json, os, shutil
from pathlib import Path

REPO = Path(".")
out_root = REPO / "results"
rows = []

for md in sorted(REPO.glob("logs/*/*/*/metrics.jsonl")):
    run = md.parent.name                      # <run_tag>-<hash>
    env = md.parent.parent.parent.name        # act_prm_<env>
    dest = out_root / env / run
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(md, dest / "metrics.jsonl")
    cfg = md.parent / "config.json"
    if cfg.is_file():
        shutil.copy2(cfg, dest / "config.json")

    recs = [json.loads(l) for l in open(md) if l.strip()]
    if not recs:
        continue
    def pick(rec, suffix):
        v = [rec[k] for k in rec if k.endswith(suffix)]
        return v[0] if v else None

    n_batches = sum(1 for r in recs if "train/loss" in r)
    # SFT: best (min) eval action-subspan ppl; RL: best (max) eval reward
    best_ao_ppl = best_ao_acc = best_ws_ppl = None
    for r in recs:
        p = pick(r, "eval_actiononly_ppl")
        if p is not None and (best_ao_ppl is None or p < best_ao_ppl):
            best_ao_ppl = p
            best_ao_acc = pick(r, "eval_actiononly_accuracy")
            best_ws_ppl = pick(r, "eval_action_ppl")
    best_rl = None
    for r in recs:
        v = pick(r, "eval/try_0/final_reward")
        if v is not None and (best_rl is None or v > best_rl):
            best_rl = v

    rows.append({
        "env": env, "run": run.split("-act-prm")[0], "batches": n_batches,
        "best_eval_actiononly_ppl": round(best_ao_ppl, 4) if best_ao_ppl else "",
        "best_eval_actiononly_acc": round(best_ao_acc, 4) if best_ao_acc else "",
        "eval_wholespan_ppl_at_best": round(best_ws_ppl, 4) if best_ws_ppl else "",
        "best_eval_rl_reward": round(best_rl, 4) if best_rl is not None else "",
        "run_dir": run,
    })

out_root.mkdir(exist_ok=True)
if rows:
    keys = list(rows[0].keys())
    with open(out_root / "SUMMARY.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for r in sorted(rows, key=lambda x: (x["env"], x["run"])): w.writerow(r)
print(f"exported {len(rows)} runs -> results/")
PY
}

while true; do
  export_once
  ./scripts/backup_results.sh >/dev/null 2>&1 || true    # dotsync copy (adapters+artifacts)
  git add -A results/ 2>/dev/null
  if ! git diff --cached --quiet 2>/dev/null; then
    git commit -q -m "results: periodic metrics export ($(date '+%m-%d %H:%M'))" 2>/dev/null \
      && echo "[$(date '+%m-%d %H:%M:%S')] committed results update"
  else
    echo "[$(date '+%m-%d %H:%M:%S')] no results change"
  fi
  sleep "$INTERVAL"
done
