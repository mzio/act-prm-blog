#!/usr/bin/env bash
# cc-13.0: SGD sft_flat Stage-2 + rollouts for airline / finance / insurance.
# Queued behind the cc-12.0 chain (chain_sgd_long.sh). See notes/cc-13.0-multidomain-sgd.md.
#
# Order: wait -> insurance gym SMOKE -> 6 training runs -> rollouts (airline, finance,
# insurance). Smoke first so a broken insurance harness costs 5 min, not 10h.
#
# WAIT: poll `kill -0 <PID>`. NOT `ps | grep <pattern>` -- that deadlocked a previous
# chain for 3.5h because this shell's own cmdline contained the literal pattern.
set -uo pipefail
cd /home/mzio/projects/act-prm-blog
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/usr/local/bin:$PATH"
[ -f scripts/wandb_preflight.sh ] && . scripts/wandb_preflight.sh
export PYTHONUNBUFFERED=1 HF_HOME=/data/users/mzio/models/hf_cache
export ACT_PRM_DUMP_TRAJECTORIES=1

WAIT_PID="${WAIT_PID:?set WAIT_PID}"
G=/tmp/aprm/multidom; mkdir -p "$G"
LOG="$G/chain.log"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
reap(){ for p in $(pgrep -f 'main_pytorch\.py'); do kill -9 "$p" 2>/dev/null; done; sleep 20; }

log "waiting on pid=$WAIT_PID"
while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
log "pid $WAIT_PID exited"; reap

# LR chosen from the retail 1e-3 vs 3e-3 rollout comparison; see cc-13.0 "LR selection".
LR=$(cat /tmp/aprm/chosen_lr.txt 2>/dev/null | tr -d '[:space:]')
if [ -z "$LR" ]; then LR=1e-3; log "WARNING: /tmp/aprm/chosen_lr.txt absent -> defaulting to LR=$LR"; fi
LRTAG="lr${LR//[-.]/_}"
# CKPT_TAG must match what the retail chains already used, so retail arms hit their
# existing .done markers and skip instead of re-running (run_sft_rollout_eval.sh has no
# domain filter -- it always attempts retail then airline).
case "$LR" in 3e-3) RTAG_RETAIL=sgd3e_3flat;; *) RTAG_RETAIL="sgd${LRTAG}";; esac
log "=== cc-13.0 multidomain: LR=$LR ($LRTAG), ckpt_tag=$RTAG_RETAIL ==="

# ------------------------------------------------------------------ 0. insurance smoke
if [ ! -f /tmp/aprm/insurance_rollout/SMOKE_OK ]; then
  log "insurance gym smoke test (1 task, base model)"
  ./scripts/smoke_insurance_gym.sh >>"$LOG" 2>&1; reap
  [ -f /tmp/aprm/insurance_rollout/SMOKE_OK ] \
    && log "  smoke PASS -- insurance rollouts enabled" \
    || log "  smoke FAIL -- insurance TRAINING still runs; its rollout will be skipped"
fi

# ------------------------------------------------------------------------- 1. training
train_dom(){  # $1=env config  $2=short name
  local env=$1 name=$2
  local m="$G/train.${name}.done"
  [ -f "$m" ] && { log "train $name done, skip"; return; }
  log "TRAIN $name (sgd $LR, sft_flat, nb200, spb32, 2 arms) -- ~3.2h"
  TRAINER_CFG=sft_flat LR="$LR" OPTIMIZER=sgd NUM_BATCHES=200 EVAL_EVERY=5 \
  STEPS_PER_BATCH=32 PATIENCE=6 VARIANTS="thoughts_policy_adamw30 actions_only" \
  REGIMES=hide CORPUS_VARIANTS="_adamw30" TAGSFX=_flat32sgd \
    ./scripts/run_sft_sweep.sh "$env" >>"$G/train.${name}.log" 2>&1
  log "  $name rc=$?"; touch "$m"; reap
}
train_dom act_prm/tau2_airline          airline
train_dom act_prm/snorkel_finance_split finance
train_dom act_prm/snorkel_insurance     insurance

# ------------------------------------------------------------------------ 2. rollouts
CKPAT="${LRTAG}_nb200_flat32sgd"

# -- airline (tau2 harness; also attempts retail, which skips on existing markers)
for step in step_0020 step_best; do
  m="$G/roll.airline.$step.done"; [ -f "$m" ] && { log "roll airline $step done, skip"; continue; }
  log "ROLLOUT airline/$step"
  VARIANTS="thoughts_policy_adamw30 actions_only" CKPT_PAT="$CKPAT" \
  CKPT_STEP="$step" CKPT_TAG="$RTAG_RETAIL" ./scripts/run_sft_rollout_eval.sh >>"$LOG" 2>&1
  log "  rc=$?"; touch "$m"; reap
done

# -- finance (fair set only: the cross-domain-comparable one; hard is difficulty-selected)
for step in step_0020 step_best; do
  m="$G/roll.finance.$step.done"; [ -f "$m" ] && { log "roll finance $step done, skip"; continue; }
  log "ROLLOUT finance/$step (fair set)"
  ARMS="thoughts_policy_adamw30 actions_only" SETS=fair CKPT_PAT="$CKPAT" \
  CKPT_STEP="$step" CKPT_TAG="sgd${LRTAG}" ./scripts/run_finance_rollout.sh >>"$LOG" 2>&1
  log "  rc=$?"; touch "$m"; reap
done

# -- insurance (only if the smoke test passed)
if [ -f /tmp/aprm/insurance_rollout/SMOKE_OK ]; then
  for step in step_0020 step_best; do
    m="$G/roll.insurance.$step.done"; [ -f "$m" ] && { log "roll insurance $step done, skip"; continue; }
    log "ROLLOUT insurance/$step"
    ARMS="thoughts_policy_adamw30 actions_only" CKPT_PAT="$CKPAT" \
    CKPT_STEP="$step" CKPT_TAG="sgd${LRTAG}" ./scripts/run_insurance_rollout.sh >>"$LOG" 2>&1
    log "  rc=$?"; touch "$m"; reap
  done
else
  log "insurance rollouts SKIPPED (smoke test did not pass)"
fi
log "=== cc-13.0 chain complete ==="
