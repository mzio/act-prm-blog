#!/usr/bin/env bash
# Re-run the four SGD 1e-3 rollouts that were skipped by a bad checkpoint glob.
# The driver looked for ${lrtag}_sgd_nb200_flat32sgd but run_sft_sweep builds
# ${LRTAG}${NBTAG}${TAGSFX} = lr1e_3_nb200_flat32sgd -- no "_sgd_" component. All four
# logged "no checkpoint, skip", the driver exited clean, and the 3e-3 chain fired early.
# Runs AFTER the in-flight 3e-3 training so nothing contends for the GPU.
set -uo pipefail
cd /home/mzio/projects/act-prm-blog
L=/tmp/aprm/sgd_flat/rollouts_1e3.log
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$L"; }
log "waiting for the 3e-3 training pass to finish"
while ps -eo args | grep -qE '[r]un_sgd_flat\.sh|[c]hain_sgd3e3\.sh'; do sleep 120; done
while pgrep -f 'main_pytorch\.py' >/dev/null 2>&1; do sleep 60; done
log "GPU free -> SGD 1e-3 rollouts"
rm -f /tmp/aprm/sgd_flat/roll_lr1e_3.*.done
ARMS="thoughts_policy_adamw30 actions_only" LRS="1e-3" ./scripts/run_sgd_flat.sh >> "$L" 2>&1
log "rc=$?"
