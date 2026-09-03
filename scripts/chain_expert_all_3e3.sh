#!/usr/bin/env bash
# expert_thoughts_all at SGD 3e-3 / sft_flat, across all four domains + rollouts.
#
# GAP THIS FILLS: cc-13.0 deliberately ran only 2 arms (actions_only, thoughts_policy), so
# there were ZERO expert_thoughts_all runs at SGD 1e-3 in any domain -- every one on disk
# is the old SGD-era lr3e_3/nb150 generation or a single retail lr1e_4 AdamW run.
#
# EXPERT POOL OVERRIDES ARE MANDATORY. train_sft.sh defaults to
# data/${ENVNAME}_expert_thoughts, and that default is WRONG for two domains. Gated
# 2026-09-02 with scripts/check_arm_pools.py against each domain's base pool (eval sets
# must be SET-EQUAL or the arm is not comparable):
#
#   retail    default  8 shared, 0 MISSING, 2 EXTRA (10 vs 8)          -> BROKEN
#             _matched 8 shared, 0 MISSING, 0 EXTRA                    -> use this
#   airline   default  4 shared, 0 MISSING, 0 EXTRA                    -> default OK
#   finance   default  3 shared, 22 MISSING, 22 EXTRA                  -> CATASTROPHIC
#             _all_v3  25 shared, 0 MISSING, 0 EXTRA                   -> use this
#   insurance default  40 shared, 0 MISSING, 0 EXTRA                   -> default OK
#
# The finance default is the known pre-v3 pool whose eval set overlaps the real one by
# 3/25; it silently shifted that arm's score before. Do not remove these overrides.
#
# expert_thoughts_all = --keep_expert_thoughts --require_thought: expert targets filtered
# to reasoning-bearing turns. (Plain expert_thoughts is ~50% bare <tool_call>, which
# teaches the model not to think.)
set -uo pipefail
cd /home/mzio/projects/act-prm-blog
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/usr/local/bin:$PATH"
[ -f scripts/wandb_preflight.sh ] && . scripts/wandb_preflight.sh
export PYTHONUNBUFFERED=1 HF_HOME=/data/users/mzio/models/hf_cache
export ACT_PRM_DUMP_TRAJECTORIES=1

WAIT_PID="${WAIT_PID:?set WAIT_PID}"
G=/tmp/aprm/expert_all_3e3; mkdir -p "$G"
LOG="$G/chain.log"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
reap(){ for p in $(pgrep -f 'main_pytorch\.py'); do kill -9 "$p" 2>/dev/null; done; sleep 20; }

log "waiting on pid=$WAIT_PID"
while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
log "pid $WAIT_PID exited"; reap

CKPAT="lr3e_3_nb200_flat32sgd"

train_one(){  # $1=envcfg  $2=short  $3=EXPERT_POOL ("" = use default)
  local env=$1 name=$2 pool=$3
  local m="$G/train.${name}.done"; [ -f "$m" ] && { log "train $name done, skip"; return; }
  log "TRAIN $name expert_thoughts_all (sgd 1e-3, sft_flat, nb200) pool=${pool:-<default>}"
  env ${pool:+EXPERT_POOL=$pool} \
    TRAINER_CFG=sft_flat LR=3e-3 OPTIMIZER=sgd NUM_BATCHES=200 EVAL_EVERY=5 \
    STEPS_PER_BATCH=32 PATIENCE=6 VARIANTS="expert_thoughts_all" \
    REGIMES=hide CORPUS_VARIANTS="_adamw30" TAGSFX=_flat32sgd \
    ./scripts/run_sft_sweep.sh "$env" >>"$G/train.${name}.log" 2>&1
  log "  rc=$?"
  # A sweep matching zero arms exits 0 -- verify snapshots actually appeared.
  local n; n=$(ls -d checkpoints_lora/*/hf_qwen3_4b_instruct/*_s2_expert_thoughts_all_${CKPAT}_heldout-*/step_* 2>/dev/null | wc -l)
  log "  cumulative expert_thoughts_all snapshots: $n"
  touch "$m"; reap
}
train_one act_prm/tau2_retail          retail    data/tau2_retail_expert_thoughts_matched
train_one act_prm/tau2_airline         airline   ""
train_one act_prm/snorkel_finance_split finance  data/snorkel_finance_split_expert_thoughts_all_v3
train_one act_prm/snorkel_insurance    insurance ""

# ------------------------------------------------------------------------- rollouts
# retail + airline share the tau2 driver (it always attempts both; the one without a
# matching checkpoint logs "no checkpoint, skip").
for step in step_0020 step_best; do
  m="$G/roll.tau2.$step.done"; [ -f "$m" ] && { log "roll tau2 $step done, skip"; continue; }
  log "ROLLOUT retail+airline expert_thoughts_all/$step"
  VARIANTS="expert_thoughts_all" CKPT_PAT="$CKPAT" CKPT_STEP="$step" \
  CKPT_TAG="sgdlr3e_3" ./scripts/run_sft_rollout_eval.sh >>"$LOG" 2>&1
  log "  rc=$?"; touch "$m"; reap
done

for step in step_0020 step_best; do
  m="$G/roll.finance.$step.done"; [ -f "$m" ] && { log "roll finance $step done, skip"; continue; }
  log "ROLLOUT finance expert_thoughts_all/$step (fair)"
  ARMS="expert_thoughts_all" SETS=fair CKPT_PAT="$CKPAT" CKPT_STEP="$step" \
  CKPT_TAG="sgdlr3e_3" ./scripts/run_finance_rollout.sh >>"$LOG" 2>&1
  log "  rc=$?"; touch "$m"; reap
done

if [ -f /tmp/aprm/insurance_rollout/SMOKE_OK ]; then
  for step in step_0020 step_best; do
    m="$G/roll.insurance.$step.done"; [ -f "$m" ] && { log "roll insurance $step done, skip"; continue; }
    log "ROLLOUT insurance expert_thoughts_all/$step"
    ARMS="expert_thoughts_all" CKPT_PAT="$CKPAT" CKPT_STEP="$step" \
    CKPT_TAG="sgdlr3e_3" ./scripts/run_insurance_rollout.sh >>"$LOG" 2>&1
    log "  rc=$?"; touch "$m"; reap
  done
else
  log "insurance expert rollouts SKIPPED (no smoke marker)"
fi
log "=== expert_thoughts_all SGD chain complete ==="
