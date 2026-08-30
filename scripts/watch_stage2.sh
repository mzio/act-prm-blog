#!/usr/bin/env bash
# Append one line per NEW eval point across every Stage-2 arm, and shout when an arm
# early-stops. Exists because early stopping was silently a no-op for SFT until
# configs/trainer/sft.yaml gained early_stop_patience: polling by hand missed it for a
# whole 150-batch arm. `eval/no_improve_evals` present at all == the flag reached the
# trainer; it reaching `patience` and the run ending == the stop actually fired.
set -uo pipefail
cd /home/mzio/projects/act-prm-blog
OUT=/tmp/aprm/stage2_adamw/evals.log
while true; do
  for d in logs/act_prm_*/hf_qwen3_4b_instruct/*_s2_*lr1e_3_adamw_nb100_heldout-*/; do
    [ -f "$d/metrics.jsonl" ] || continue
    uv run --no-project python - "$d" "$OUT" <<'PY' 2>/dev/null
import json, os, sys
d, out = sys.argv[1], sys.argv[2]
tag = os.path.basename(d.rstrip("/")).split("-act-prm")[0]
seen = set()
if os.path.exists(out):
    seen = {l.split()[0] + l.split()[1] for l in open(out) if l.strip() and not l.startswith("#")}
rows = [json.loads(l) for l in open(d + "metrics.jsonl") if l.strip()]
new, done = [], set()
for r in rows:
    if "eval/eval_action_ppl" not in r:
        continue
    b = r.get("progress/batch")
    k = f"{tag}b{b}"
    if k in seen or k in done:
        continue
    done.add(k)
    new.append(f"{tag} b{b} ppl={r['eval/eval_action_ppl']:.4f} "
               f"acc={r.get('eval/eval_action_accuracy', 0):.4f} "
               f"no_improve={r.get('eval/no_improve_evals', 'ABSENT')}")
if new:
    with open(out, "a") as f:
        f.write("\n".join(new) + "\n")
PY
    # an "EARLY STOP" line in the trainer log is the definitive proof it fired
    for l in /tmp/aprm/sft_sweep_*/*lr1e_3_adamw_nb100_heldout.log; do
      [ -f "$l" ] || continue
      grep -h "EARLY STOP" "$l" 2>/dev/null | while read -r line; do
        grep -qF "$line" "$OUT" 2>/dev/null || echo "### $(basename "$l" .log): $line" >> "$OUT"
      done
    done
  done
  sleep 120
done
