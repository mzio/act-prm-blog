#!/usr/bin/env bash
# Retry the insurance expert_thoughts arm once the current sweep frees the GPU.
#
# It failed 2026-08-30 12:41 in 26s with KeyError: 'dataset' -- train_sft.sh looked for
# data/snorkel_insurance_expert_thoughts (ENVNAME=snorkel_insurance) but the pool was built
# as snorkel_insurance_split_expert_thoughts, so the env tried to rebuild from split_file
# and that JSON has no "dataset" key. Fixed by a symlink; this re-runs the arm.
#
# Also removes the domain-level marker: run_stage2_adamw.sh marks insurance done when the
# sweep exits 0 (which it does even with a failed arm), and the watchdog waits for 12 arm
# markers -- so without this the watchdog would relaunch a driver that skips everything and
# exits, in a 2-minute loop.
set -uo pipefail
cd /home/mzio/projects/act-prm-blog
L=/tmp/aprm/stage2_adamw/insurance_expert_retry.log
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" >> "$L"; }
log "waiting for the current sweep to finish"
while pgrep -f 'main_pytorch\.py' >/dev/null || pgrep -f 'run_sft_sweep\.sh' >/dev/null; do sleep 120; done
log "GPU free; retrying insurance expert_thoughts"
rm -f /tmp/aprm/stage2_adamw/insurance.done
. scripts/wandb_preflight.sh >/dev/null 2>&1
export PYTHONUNBUFFERED=1 HF_HOME=/data/users/mzio/models/hf_cache
LR=1e-3 OPTIMIZER=adamw NUM_BATCHES=100 EVAL_EVERY=10 REGIMES=hide \
  CORPUS_VARIANTS="_adamw30" VARIANTS="expert_thoughts" \
  ./scripts/run_sft_sweep.sh act_prm/snorkel_insurance >> "$L" 2>&1
log "retry exited rc=$?"
