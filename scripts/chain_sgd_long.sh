#!/usr/bin/env bash
# After the in-flight SGD 1e-3 rollout driver exits:
#   1. rollouts for the NEW SGD 3e-3 sft_flat checkpoints (step_best, both arms)
#   2. SGD 1e-3 training to nb1000 with eval-PPL early stopping -- the (A)/(B) discriminator
#   3. rollout of that run's step_best
#
# WAIT MECHANISM: poll `kill -0 <PID>`, NOT `ps | grep <pattern>`. The grep form deadlocked
# chain_sgd1e3_rollouts.sh for 3.5h -- the bracket trick ([r]un_sgd_flat) only defeats
# grep's own argv, but the Claude Code bash wrapper's cmdline contained the whole heredoc
# that created the script, including the literal pattern, so the match never cleared.
#
# Calls run_sft_sweep.sh DIRECTLY rather than run_sgd_flat.sh: the latter hardcodes
# NUM_BATCHES=200 / EVAL_EVERY=5 / PATIENCE=6 / TAGSFX=_flat32sgd and globs nb200_flat32sgd,
# so a nb1000 request passed via env would silently have produced another nb200 run.
# run_sgd_flat.sh is also mid-execution by the driver we are waiting on, and bash reads
# scripts lazily -- editing it in place could corrupt that running loop.
set -uo pipefail
cd /home/mzio/projects/act-prm-blog
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/usr/local/bin:$PATH"
[ -f scripts/wandb_preflight.sh ] && . scripts/wandb_preflight.sh
export PYTHONUNBUFFERED=1 HF_HOME=/data/users/mzio/models/hf_cache
export ACT_PRM_DUMP_TRAJECTORIES=1

WAIT_PID="${WAIT_PID:?set WAIT_PID to the in-flight driver pid}"
G=/tmp/aprm/sgd_long; mkdir -p "$G"
LOG="$G/chain.log"
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
reap() { for p in $(pgrep -f 'main_pytorch\.py'); do kill -9 "$p" 2>/dev/null; done; sleep 20; }

log "waiting on driver pid=$WAIT_PID"
while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
log "driver $WAIT_PID exited"
reap

# ------------------------------------------------------------- 1. 3e-3 step_best rollouts
# CKPT_PAT excludes _heldout: the glob is ${dom}_s2_${v}_${CKPT_PAT}_heldout-*/${CKPT_STEP}.
# Both arms verified FOUND before launch. airline has no such checkpoint and will skip.
for arm in thoughts_policy_adamw30 actions_only; do
  m="$G/roll3e3.${arm}.done"; [ -f "$m" ] && { log "roll 3e-3 $arm done, skip"; continue; }
  log "ROLLOUT 3e-3 $arm/step_best"
  VARIANTS="$arm" CKPT_PAT="lr3e_3_nb200_flat32sgd" CKPT_STEP="step_best" \
  CKPT_TAG="sgd3e_3flat" ./scripts/run_sft_rollout_eval.sh >>"$LOG" 2>&1
  log "  rc=$?"; touch "$m"; reap
done

# -------------------------------------------- 2. SGD 1e-3 to nb1000 with early stopping
# eval_every 10 (not 5): 1000 batches would otherwise be 200 evals. patience 8 => stop
# after 80 batches with no eval_actiononly_ppl gain. best_metric is hardcoded to
# eval_actiononly_ppl inside run_sft_sweep.sh, which is what we want.
# save_every stays at the sft_flat.yaml value of 10 -> ~100 LoRA snapshots (~30 MB each).
m="$G/train_1e3_nb1000.done"
if [ ! -f "$m" ]; then
  log "TRAIN SGD 1e-3 nb1000 (thoughts_policy_adamw30) -- ~8.5h, early stop on eval_actiononly_ppl"
  TRAINER_CFG=sft_flat EXPERT_POOL=data/tau2_retail_expert_thoughts_matched \
  LR=1e-3 OPTIMIZER=sgd NUM_BATCHES=1000 EVAL_EVERY=10 STEPS_PER_BATCH=32 PATIENCE=8 \
  VARIANTS="thoughts_policy" REGIMES=hide CORPUS_VARIANTS="_adamw30" TAGSFX=_flat32sgd1k \
    ./scripts/run_sft_sweep.sh act_prm/tau2_retail >>"$G/train_nb1000.log" 2>&1
  log "  rc=$?"; touch "$m"; reap
fi

# ---------------------------------------------------- 3. rollout the nb1000 step_best
m="$G/roll_nb1000.done"
if [ ! -f "$m" ]; then
  ck=$(ls -d checkpoints_lora/act_prm_tau2_retail/hf_qwen3_4b_instruct/retail_s2_thoughts_policy_adamw30_lr1e_3_nb1000_flat32sgd1k_heldout-*/step_best 2>/dev/null | head -1)
  if [ -z "$ck" ]; then
    log "nb1000 step_best MISSING -- not rolling out (check $G/train_nb1000.log)"
  else
    log "ROLLOUT nb1000 thoughts_policy/step_best"
    VARIANTS="thoughts_policy_adamw30" CKPT_PAT="lr1e_3_nb1000_flat32sgd1k" \
    CKPT_STEP="step_best" CKPT_TAG="sgd1e_3_nb1k" ./scripts/run_sft_rollout_eval.sh >>"$LOG" 2>&1
    log "  rc=$?"; reap
  fi
  touch "$m"
fi
log "=== chain complete ==="
