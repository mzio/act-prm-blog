#!/usr/bin/env bash
# Wait for the in-flight rollout arm to finish, drop the remaining queued rollout work,
# then run the lr 1e-4 Stage-2 sweep on retail with the whole GPU.
#
# Why lr 1e-4: the conversational collapse tracks MEDIAN adapter movement, not peak.
#   SGD 3e-3 step_best (converses, 21.4%): median |B@A| = 4.7e-05
#   AdamW 1e-3 step_0020 (collapsed, 0% prose): 1.46e-03   <- 31x, already at b20
#   AdamW 1e-3 step_best (collapsed, 0/42):     2.19e-03
# Stopping earlier does NOT help (step_0020 is already collapsed), so the lever has to be
# the per-step size itself. save_every is now 10, so we get a fine grid of snapshots to
# find where prose survives.
set -uo pipefail
cd /home/mzio/projects/act-prm-blog
L=/tmp/aprm/stage2_lr1e4/chain.log; mkdir -p /tmp/aprm/stage2_lr1e4
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$L"; }

log "waiting for the in-flight rollout arm to finish"
while pgrep -f 'main_pytorch\.py' >/dev/null 2>&1; do sleep 60; done
log "GPU free; dropping queued rollout drivers"
for p in $(ps -eo pid,args | awk '$2=="bash" && $3 ~ /run_sft_rollout_eval\.sh$/ {print $1}'); do
  kill -9 "$p" 2>/dev/null; log "  killed rollout driver $p"
done
sleep 5
log "launching lr 1e-4 sweep (retail)"
ONLY_DOMAINS="retail" LR=1e-4 OPTIMIZER=adamw NUM_BATCHES=200 EVAL_EVERY=5 \
  STEPS_PER_BATCH=32 PATIENCE=6 TAGSFX=_flat32 \
  ./scripts/run_stage2_flat.sh >> "$L" 2>&1
log "lr 1e-4 sweep exited rc=$?"
