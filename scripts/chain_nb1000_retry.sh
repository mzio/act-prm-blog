#!/usr/bin/env bash
# Retry of the SGD 1e-3 nb1000 run, queued BEHIND the cc-13.0 multidomain chain.
#
# WHY IT FAILED THE FIRST TIME: chain_sgd_long.sh passed VARIANTS="thoughts_policy", but
# with CORPUS_VARIANTS="_adamw30" the sweep's label is "thoughts_policy_adamw30" and the
# VARIANTS filter matches the LABEL. Nothing matched; the sweep enumerated its arms and
# exited 0 in 20 seconds, so rc=0 looked like success and the chain moved on to a rollout
# whose checkpoint had never been produced. Fixed here by passing the full label.
#
# WHY IT IS DEPRIORITISED: its purpose was to discriminate (A) "degradation tracks fit
# depth" from (B) "SGD's update shape is protective" by driving SGD to AdamW-level fit.
# The 3e-3 rollouts answered that in the meantime -- SGD 3e-3 at PPL 2.28 scored 9.5%,
# WORSE than AdamW 1e-3 step_0020 at PPL 1.78 (11.9%) -- so (B) is dead and (A) holds.
# This run now only sharpens the tail of a curve that is already well determined, so it
# must not block the multidomain sweep.
set -uo pipefail
cd /home/mzio/projects/act-prm-blog
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/usr/local/bin:$PATH"
[ -f scripts/wandb_preflight.sh ] && . scripts/wandb_preflight.sh
export PYTHONUNBUFFERED=1 HF_HOME=/data/users/mzio/models/hf_cache
export ACT_PRM_DUMP_TRAJECTORIES=1

WAIT_PID="${WAIT_PID:?set WAIT_PID}"
G=/tmp/aprm/nb1000; mkdir -p "$G"
LOG="$G/chain.log"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
reap(){ for p in $(pgrep -f 'main_pytorch\.py'); do kill -9 "$p" 2>/dev/null; done; sleep 20; }

log "waiting on pid=$WAIT_PID"
while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
log "pid $WAIT_PID exited"; reap

if [ ! -f "$G/train.done" ]; then
  log "TRAIN SGD 1e-3 nb1000 (thoughts_policy_adamw30) -- ~8.5h"
  TRAINER_CFG=sft_flat LR=1e-3 OPTIMIZER=sgd NUM_BATCHES=1000 EVAL_EVERY=10 \
  STEPS_PER_BATCH=32 PATIENCE=8 VARIANTS="thoughts_policy_adamw30" \
  REGIMES=hide CORPUS_VARIANTS="_adamw30" TAGSFX=_flat32sgd1k \
    ./scripts/run_sft_sweep.sh act_prm/tau2_retail >>"$G/train.log" 2>&1
  log "  rc=$?"
  # Verify the run ACTUALLY produced checkpoints before marking done -- rc=0 from a sweep
  # that matched no arms is exactly what fooled the first attempt.
  n=$(ls -d checkpoints_lora/act_prm_tau2_retail/hf_qwen3_4b_instruct/retail_s2_thoughts_policy_adamw30_lr1e_3_nb1000_flat32sgd1k_heldout-*/step_* 2>/dev/null | wc -l)
  log "  produced $n snapshot(s)"
  [ "$n" -gt 0 ] && touch "$G/train.done" || { log "NO SNAPSHOTS -- aborting"; exit 1; }
  reap
fi

for step in step_0020 step_best; do
  m="$G/roll.$step.done"; [ -f "$m" ] && { log "roll $step done, skip"; continue; }
  ck=$(ls -d checkpoints_lora/act_prm_tau2_retail/hf_qwen3_4b_instruct/retail_s2_thoughts_policy_adamw30_lr1e_3_nb1000_flat32sgd1k_heldout-*/$step 2>/dev/null | head -1)
  [ -z "$ck" ] && { log "$step MISSING, skip"; continue; }
  log "ROLLOUT nb1000/$step"
  VARIANTS="thoughts_policy_adamw30" CKPT_PAT="lr1e_3_nb1000_flat32sgd1k" \
  CKPT_STEP="$step" CKPT_TAG="sgd1e_3_nb1k" ./scripts/run_sft_rollout_eval.sh >>"$LOG" 2>&1
  log "  rc=$?"; touch "$m"; reap
done
log "=== nb1000 retry complete ==="
