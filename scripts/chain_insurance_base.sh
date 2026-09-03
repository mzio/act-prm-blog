#!/usr/bin/env bash
# Queued LAST (after chain_expert_all_sgd). Placed at the end deliberately: the four
# existing chains form a strict serial dependency (multidomain -> nb1000 ->
# corpus_ablation -> expert_all), each polling the previous chain's PID. Inserting a new
# chain that waits on a MIDDLE pid would make two chains fire at once and contend for the
# GPU (~89 GiB each), so the only safe insertion point is the tail.
set -uo pipefail
cd /home/mzio/projects/act-prm-blog
WAIT_PID="${WAIT_PID:?set WAIT_PID}"
G=/tmp/aprm/insbase; mkdir -p "$G"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$G/chain.log"; }
log "waiting on pid=$WAIT_PID"
while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
log "pid $WAIT_PID exited"
for p in $(pgrep -f 'main_pytorch\.py'); do kill -9 "$p" 2>/dev/null; done; sleep 20
./scripts/run_insurance_base_rollout.sh >>"$G/chain.log" 2>&1
log "rc=$? ; === insurance base complete ==="
