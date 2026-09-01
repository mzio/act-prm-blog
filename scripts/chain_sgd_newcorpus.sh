#!/usr/bin/env bash
# Run the SGD-on-new-corpus experiment once the retail ladder's last rollout finishes.
set -uo pipefail
cd /home/mzio/projects/act-prm-blog
L=/tmp/aprm/sgd_flat/chain.log; mkdir -p /tmp/aprm/sgd_flat
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$L"; }
log "waiting for the retail snapshot ladder to finish"
while ps -eo args | grep -qE '[c]hain_snapshot_rollouts\.sh'; do sleep 120; done
while pgrep -f 'main_pytorch\.py' >/dev/null 2>&1; do sleep 60; done
log "GPU free -> SGD-on-new-corpus experiment"
./scripts/run_sgd_flat.sh >> "$L" 2>&1
log "experiment rc=$? -- done"
