#!/usr/bin/env bash
# Re-run expert_thoughts_all once the current lr 1e-4 sweep finishes. The first attempt
# silently ran as a duplicate of expert_thoughts (require_thought was unimplemented in
# sft_flat); now fixed, so the arm trains on the 317 reasoning-bearing turns rather than
# all 635.
set -uo pipefail
cd /home/mzio/projects/act-prm-blog
L=/tmp/aprm/stage2_lr1e4/expert_all_rerun.log
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$L"; }
log "waiting for the lr 1e-4 sweep driver to finish"
while ps -eo args | grep -q '[r]un_stage2_flat\.sh'; do sleep 60; done
while pgrep -f 'main_pytorch\.py' >/dev/null 2>&1; do sleep 30; done
log "GPU free; re-running expert_thoughts_all with require_thought active"
[ -f scripts/wandb_preflight.sh ] && . scripts/wandb_preflight.sh
export PYTHONUNBUFFERED=1 HF_HOME=/data/users/mzio/models/hf_cache
LR=1e-4 OPTIMIZER=adamw NUM_BATCHES=200 EVAL_EVERY=5 STEPS_PER_BATCH=32 PATIENCE=6 \
  TRAINER_CFG=sft_flat EXPERT_POOL=data/tau2_retail_expert_thoughts_matched \
  VARIANTS="expert_thoughts_all" REGIMES=hide CORPUS_VARIANTS="_adamw30" TAGSFX=_flat32 \
  ./scripts/run_sft_sweep.sh act_prm/tau2_retail >> "$L" 2>&1
log "expert_thoughts_all re-run exited rc=$?"
