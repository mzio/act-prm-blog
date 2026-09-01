#!/usr/bin/env bash
# Wait for THIS specific rollout process (pid 3420235, the step_0020 actions_only control) to
# exit, then stop the rollout driver and start the lr 1e-4 sweep. The previous chain waited
# for "no main_pytorch", which never happens while a rollout driver is alive -- it launches
# the next arm immediately.
set -uo pipefail
cd /home/mzio/projects/act-prm-blog
L=/tmp/aprm/stage2_lr1e4/chain.log; mkdir -p /tmp/aprm/stage2_lr1e4
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$L"; }
log "waiting on pid 3420235 (step_0020 actions_only control)"
while kill -0 3420235 2>/dev/null; do sleep 30; done
log "control arm done; stopping rollout drivers"
for p in $(ps -eo pid,args | awk '$2=="bash" && $3 ~ /run_sft_rollout_eval\.sh$/ {print $1}'); do
  kill -9 "$p" 2>/dev/null; log "  killed driver $p"
done
sleep 5
log "launching lr 1e-4 sweep (retail, sft_flat, AdamW, nb200, spb32, eval_every 5)"
ONLY_DOMAINS="retail" LR=1e-4 OPTIMIZER=adamw NUM_BATCHES=200 EVAL_EVERY=5   STEPS_PER_BATCH=32 PATIENCE=6 TAGSFX=_flat32   ./scripts/run_stage2_flat.sh >> "$L" 2>&1
log "lr 1e-4 sweep exited rc=$?"
