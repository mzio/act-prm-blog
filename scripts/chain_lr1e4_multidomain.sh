#!/usr/bin/env bash
# Full autonomous chain: wait for the retail snapshot ladder, then train lr 1e-4 on
# airline/finance/insurance (3 arms each) and roll out airline + finance.
# Plan and results: notes/cc-11.0-lr1e4-multidomain-plan.md
set -uo pipefail
cd /home/mzio/projects/act-prm-blog
L=/tmp/aprm/lr1e4_multi/chain.log; mkdir -p /tmp/aprm/lr1e4_multi
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$L"; }

log "waiting for the retail snapshot ladder to finish"
while ps -eo args | grep -qE '[c]hain_snapshot_rollouts\.sh|[r]un_sft_rollout_eval\.sh'; do sleep 120; done
while pgrep -f 'main_pytorch\.py' >/dev/null 2>&1; do sleep 60; done
log "GPU free -> training phase"
./scripts/run_lr1e4_multidomain.sh >> "$L" 2>&1
log "training phase rc=$?  -> rollout phase"
./scripts/run_lr1e4_multidomain_rollouts.sh >> "$L" 2>&1
log "rollout phase rc=$?  -- ALL DONE"
