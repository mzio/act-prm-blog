#!/usr/bin/env bash
# Stage-2 SFT on the BASE-SCORED thought corpora (thoughts_base_adamw30), 3 domains.
#
# WHAT THIS TESTS. Every thought corpus used so far (`policy_adamw30`) rewarded candidate
# thoughts by p(x|s,z) under the *current LoRA policy*. The `base_adamw30` corpora --
# generated 2026-09-05/06 by chain_stage1_base.sh with --score_with_base -- reward them
# under the *frozen base model* instead. Same tasks, same action points, same 4 candidates
# per row; only the scorer differs (verified: train counts match policy exactly, 24/52/180).
# So this asks: does the EM's reward signal degrade as the policy drifts?
#
# CHECKPOINTING (unchanged from every other Stage-2 arm, so results stay comparable):
#   save_every 10 + keep_step_checkpoints -> step_0010 ... step_0200
#   step_best selected on --best_metric eval_actiononly_ppl (the ACTION-token-only span;
#     run_sft_sweep.sh:101 passes this explicitly, overriding sft_flat.yaml's final_reward)
#   step_last also kept  => 22 checkpoints per arm
#   sft_flat, SGD 1e-3, nb 200, spb 32, eval_every 5, patience 6, hide-observations, r8/a16
#
# ROLLOUTS. Only insurance can actually resolve a difference (noise ~2.5pt same-seed /
# ~3.1pt seed-varied, n=40, base 30.0%). retail (~16pt) and airline (~22pt) single-run
# noise exceeds any plausible effect, so their rollouts are recorded for completeness but
# should be read as direction-only. Finance has no base corpus (dropped from Stage-1).
set -uo pipefail
cd /home/mzio/projects/act-prm-blog
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/usr/local/bin:$PATH"
[ -f scripts/wandb_preflight.sh ] && . scripts/wandb_preflight.sh
export PYTHONUNBUFFERED=1
export HF_HOME=/data/users/mzio/models/hf_cache
export HF_HUB_OFFLINE=1
export WANDB_ERROR_REPORTING=false
export ACT_PRM_DUMP_TRAJECTORIES=1
G=/tmp/aprm/thoughts_base; mkdir -p "$G"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$G/chain.log"; }
reap(){ for p in $(pgrep -f 'main_pytorch\.py'); do kill -9 "$p" 2>/dev/null; done; sleep 15; }

CKPAT="lr1e_3_nb200_flat32sgd"

train_one(){  # $1=envcfg $2=name
  local env=$1 name=$2
  [ -f "$G/train.$name.done" ] && { log "train $name done, skip"; return; }
  log "TRAIN $name thoughts_base_adamw30 (sgd 1e-3, sft_flat, nb200, save_every 10)"
  TRAINER_CFG=sft_flat LR=1e-3 OPTIMIZER=sgd NUM_BATCHES=200 EVAL_EVERY=5 \
  STEPS_PER_BATCH=32 PATIENCE=6 VARIANTS="thoughts_base_adamw30" \
  REGIMES=hide CORPUS_VARIANTS="_adamw30" TAGSFX=_flat32sgd \
    ./scripts/run_sft_sweep.sh "$env" >>"$G/train.$name.log" 2>&1
  log "  rc=$?"
  # A sweep that matches ZERO arms exits 0 -- verify snapshots really appeared.
  local n; n=$(ls -d checkpoints_lora/*/hf_qwen3_4b_instruct/*_s2_thoughts_base_adamw30_${CKPAT}_heldout-*/step_* 2>/dev/null | wc -l)
  log "  cumulative thoughts_base snapshots: $n"
  touch "$G/train.$name.done"; reap
}

log "=== thoughts_base_adamw30 Stage-2: airline -> retail -> insurance ==="
train_one act_prm/tau2_airline      airline
train_one act_prm/tau2_retail       retail
train_one act_prm/snorkel_insurance insurance

# --- rollouts: insurance first (the only domain that resolves), then tau2 for the record
for step in step_0020 step_best; do
  m="$G/roll.ins.$step.done"; [ -f "$m" ] && continue
  log "ROLLOUT insurance thoughts_base/$step"
  ARMS="thoughts_base_adamw30" CKPT_PAT="$CKPAT" CKPT_STEP="$step" \
  CKPT_TAG="sgdlr1e_3" ./scripts/run_insurance_rollout.sh >>"$G/chain.log" 2>&1
  log "  rc=$?"; touch "$m"; reap
done
for step in step_0020 step_best; do
  m="$G/roll.tau2.$step.done"; [ -f "$m" ] && continue
  log "ROLLOUT retail+airline thoughts_base/$step"
  VARIANTS="thoughts_base_adamw30" CKPT_PAT="$CKPAT" CKPT_STEP="$step" \
  CKPT_TAG="sgdlr1e_3" ./scripts/run_sft_rollout_eval.sh >>"$G/chain.log" 2>&1
  log "  rc=$?"; touch "$m"; reap
done
log "=== thoughts_base chain complete ==="
