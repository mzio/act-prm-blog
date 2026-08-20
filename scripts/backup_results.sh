#!/usr/bin/env bash
# Durable backup of RUN RESULTS into git.
#
# Why: logs/ and checkpoints_lora/ are gitignored (.gitignore:224-226), so they live
# ONLY on the devserver's local disk. When a devserver dies you lose every result that
# isn't in git (this happened 2026-07-30/31: the whole Stage-3 RLVR fleet was lost).
# This script copies the SMALL, high-value artifacts (metrics.jsonl + a summary) into
# results/ (which IS tracked) and commits them.
#
# It does NOT push: github.com is blocked by fwdproxy from the devserver (403), so run
#   git push
# from a shell that reaches github (your laptop / a github-capable host).
#
# Usage:  ./scripts/backup_results.sh [commit_message_suffix]
set -uo pipefail
cd "$(dirname "$0")/.."
OUT=results
mkdir -p "$OUT"

# 1) Copy every metrics.jsonl, preserving <family>/<run_tag>/ structure (small: ~KBs each).
n=0
while IFS= read -r f; do
  rel=${f#logs/}
  fam=$(echo "$rel" | cut -d/ -f1)
  run=$(echo "$rel" | rev | cut -d/ -f2 | rev)
  dest="$OUT/$fam/$run"
  mkdir -p "$dest"
  cp -f "$f" "$dest/metrics.jsonl" 2>/dev/null && n=$((n+1))
  # config snapshot, if present, so a result is reproducible
  cfg=$(dirname "$f")/config.json
  [ -f "$cfg" ] && cp -f "$cfg" "$dest/config.json" 2>/dev/null
done < <(find logs -name metrics.jsonl 2>/dev/null)

# 2) Machine-readable summary of every run's headline numbers.
python3 - "$OUT" <<'PY'
import json, os, sys, glob, csv
out = sys.argv[1]
rows = []
for f in glob.glob(os.path.join(out, "*", "*", "metrics.jsonl")):
    try:
        recs = [json.loads(l) for l in open(f) if l.strip()]
    except Exception:
        continue
    if not recs:
        continue
    fam, run = f.split(os.sep)[-3], f.split(os.sep)[-2]
    def col(k):
        return [r[k] for r in recs if k in r]
    batch = max([r.get("progress/batch", -1) for r in recs] + [-1])
    ev = col("eval/try_0/final_reward")
    best = col("eval/final_reward_best")
    bstep = col("eval/final_reward_best_step")
    ppl = col("eval/action_ppl")
    rows.append({
        "family": fam, "run": run, "last_batch": batch,
        "eval_final_reward_last": round(ev[-1], 4) if ev else "",
        "eval_final_reward_best": round(max(best), 4) if best else "",
        "best_step": bstep[-1] if bstep else "",
        "eval_action_ppl_last": round(ppl[-1], 4) if ppl else "",
        "n_evals": len(ev),
    })
rows.sort(key=lambda r: (r["family"], r["run"]))
with open(os.path.join(out, "SUMMARY.csv"), "w", newline="") as fh:
    if rows:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
print(f"summarized {len(rows)} runs")
PY

# 3) Commit (no push -- see header).
git add -A "$OUT" >/dev/null 2>&1
if git diff --cached --quiet; then
  echo "backup_results: no changes to commit ($n metrics files checked)"
else
  git commit -q -m "results: backup run metrics${1:+ ($1)} [$(date '+%Y-%m-%d %H:%M')]" && \
    echo "backup_results: committed $n metrics files -> $OUT (remember: git push from a github-capable shell)"
fi
