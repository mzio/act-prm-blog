#!/usr/bin/env bash
# After the lr 1e-4 work finishes, roll out a LADDER of snapshots and measure
# user-turns/episode -- the only statistic so far that tracks task completion monotonically
# (SGD 4.53 -> 21.4%; lr1e-3 step_0020 3.61 -> 11.9%; lr1e-3 step_best 2.05 -> 0.0%).
# None of the training-time metrics predict it: eval PPL is ANTI-correlated over the
# stretch that matters, and median |B@A| failed (lr1e-3 step_0020 sat 31x above the
# conversational SGD checkpoint and still conversed).
#
# The ladder spans the median-movement range the lr 1e-4 run now covers:
#   step_0020 ~1.7e-04, step_0060 ~3.2e-04, step_best ~4.3e-04   (SGD reference 4.7e-05)
set -uo pipefail
cd /home/mzio/projects/act-prm-blog
L=/tmp/aprm/rollout/snapshot_ladder.log; mkdir -p /tmp/aprm/rollout
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$L"; }

log "waiting for all lr 1e-4 training to finish"
while ps -eo args | grep -qE '[r]un_stage2_flat\.sh|[c]hain_expert_all_rerun\.sh|[r]un_sft_sweep\.sh'; do sleep 60; done
while pgrep -f 'main_pytorch\.py' >/dev/null 2>&1; do sleep 30; done
log "GPU free; starting the snapshot ladder"

for spec in "thoughts_policy_adamw30:step_0020" \
            "thoughts_policy_adamw30:step_0060" \
            "thoughts_policy_adamw30:step_best" \
            "actions_only:step_best" \
            "expert_thoughts_all:step_best"; do
  arm="${spec%%:*}"; step="${spec##*:}"
  log "ROLLOUT $arm @ $step"
  VARIANTS="$arm" CKPT_PAT="lr1e_4_adamw_nb200_flat32" CKPT_TAG="lr1e4" CKPT_STEP="$step" \
    ./scripts/run_sft_rollout_eval.sh >> "$L" 2>&1 || log "  $arm@$step FAILED"
  # the driver launches airline arms too; we only want retail here, so stop after retail
  for p in $(ps -eo pid,args | awk '$2=="bash" && $3 ~ /run_sft_rollout_eval\.sh$/ {print $1}'); do
    kill -9 "$p" 2>/dev/null
  done
  # kill any orphaned child (killing the driver alone leaves it holding the GPU -- this bit
  # us three times today, most recently blocking the lr 1e-4 sweep for 13 minutes)
  for p in $(pgrep -f 'main_pytorch\.py'); do kill -9 "$p" 2>/dev/null; done
  sleep 10
done
log "snapshot ladder complete"
