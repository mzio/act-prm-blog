#!/usr/bin/env bash
# Stage-3 RL, STRONGER + RLVR config — the underpowered first matrix (group 4, bs 2,
# max_turns 20, GRPO mean-centering over {+1,-1}, discount 0.9) barely learned: most
# groups mean-centered to ~zero advantage, so <30% of batches produced a gradient.
#
# This run switches to pure RLVR on 4 core arms:
#   * reward = {+1 correct, 0 fail}      (env retail_rlvr, negative_rewards: false)
#   * advantage = raw reward, NO baseline (generator hf_rlvr, mean_center: false)
#   * discount_factor 1.0                (undiscounted -> every token in a winning
#                                         trajectory gets advantage 1; failures 0)
#   * group_size 8, batch_size 1         (8 concurrent rollouts; ~fits 95 GiB w/ ckpting)
#   * max_turns 30, num_batches 100      (more turns + steps to actually move the policy)
#   * gradient_checkpointing             (RL is user-sim-latency-bound -> ~free)
#   * eval on 20 hold-out tasks during RL (bigger, more stable early-stop signal),
#     full 42 never-in-logs eval at the end.
# Only all-fail groups (P≈0.7^8≈6% at ~30% success) waste a step now.
#
# Arms: base (base-direct floor), actions_only, thoughts_base, thoughts_policy.
# Distinct run_tags (retail_rlvr_*) so nothing collides with the weak retail_rl_* runs.
#
# Usage: CUDA_VISIBLE_DEVICES=0 nohup ./scripts/run_retail_rl_strong.sh > /tmp/aprm/rlvr/run.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/usr/local/bin:$PATH"

MODEL="${MODEL_CFG:-hf_qwen3_4b_instruct}"; export MODEL_CFG="$MODEL"
CKROOT="checkpoints_lora/act_prm_tau2_retail/$MODEL"   # SFT (Stage-2) inits
S3ROOT="checkpoints_lora/tau2bench_retail/$MODEL"      # RL checkpoints
S3LOGROOT="logs/tau2bench_retail/$MODEL"               # RL logs (metrics.jsonl)
MDIR=/tmp/aprm/rlvr; mkdir -p "$MDIR"
DRY="${DRY:-0}"
# HIDE=1 (default): hide-observations RL, warm-started ONLY from HIDE SFT checkpoints
#   (retail_s2_<v>_heldout, no _fullctx), appends --hide_observations.
# HIDE=0: full-context RL, warm-started ONLY from FULL-context SFT (_fullctx).
# The regime is baked into the run_tag (_hide/_full) so the two never collide and
# each RL arm starts from the MATCHING SFT regime (never cross regimes).
HIDE="${HIDE:-1}"
NUM_BATCHES="${NUM_BATCHES:-40}"
MAX_TURNS="${MAX_TURNS:-20}"
GROUP_SIZE="${GROUP_SIZE:-4}"
BATCH_SIZE="${BATCH_SIZE:-2}"      # unique tasks/step; group4*bs2=8 rollouts fits WITHOUT grad-ckpt
GRAD_CKPT="${GRAD_CKPT:-0}"        # 0=off (fast). Only needed for group8 (VRAM); grad-ckpt ~2x slower.
EVAL_EVERY="${EVAL_EVERY:-20}"     # eval on the 20-task hold-out subset every N batches
PATIENCE="${PATIENCE:-3}"          # early-stop if eval success (final_reward) doesn't improve for N evals
if [ "$HIDE" = 1 ]; then REG=hide; HIDE_ARGS=(--hide_observations); else REG=full; HIDE_ARGS=(); fi
if [ "$GRAD_CKPT" = 1 ]; then CKPT_ARGS=(--gradient_checkpointing); else CKPT_ARGS=(); fi
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$MDIR/rlvr.log"; }
wait_gpu_free(){ while pgrep -f '[m]ain_pytorch.py' >/dev/null 2>&1; do sleep 60; done; sleep 10; }
newest(){ ls -dt $1 2>/dev/null | head -1; }

# Common RLVR overrides (appended last -> argparse last-wins beats train_rl_from_sft.sh's
# hardcoded --generator_config hf_grpo / --env_config tau2bench/retail / group/bs/turns).
RLVR_ARGS=(--generator_config hf_rlvr --env_config tau2bench/retail_rlvr
           --group_size "$GROUP_SIZE" --batch_size "$BATCH_SIZE" --max_turns "$MAX_TURNS" --max_tokens 2048
           --num_batches "$NUM_BATCHES" --eval_every "$EVAL_EVERY" --early_stop_patience "$PATIENCE"
           --discount_factor 1.0 "${CKPT_ARGS[@]}" "${HIDE_ARGS[@]}")

log "=== RLVR strong Stage-3 start (regime=$REG group=$GROUP_SIZE bs=$BATCH_SIZE turns=$MAX_TURNS batches=$NUM_BATCHES grad_ckpt=$GRAD_CKPT eval_every=$EVAL_EVERY patience=$PATIENCE) ==="

# --- task split (train 72 logged; eval-during-RL first 20 never-in-logs; final 42) ---
SPLIT_JSON="data/splits/tau2_retail_uid_to_tau2id.json"
TRAIN_IDS=$(python3 -c "import json;print(' '.join(json.load(open('$SPLIT_JSON'))['coverage']['covered_tau2_ids']))")
EVAL_IDS=$(python3 -c "import json;print(' '.join(json.load(open('$SPLIT_JSON'))['never_in_logs']['ids']))")
EVAL_IDS_TRAIN=$(python3 -c "import json;print(' '.join(json.load(open('$SPLIT_JSON'))['never_in_logs']['ids'][:20]))")
[ -n "$TRAIN_IDS" ] && [ -n "$EVAL_IDS" ] && [ -n "$EVAL_IDS_TRAIN" ] || { log "FATAL: bad split json"; exit 1; }
log "split: train=$(echo $TRAIN_IDS|wc -w) eval-during-RL=$(echo $EVAL_IDS_TRAIN|wc -w) final-eval=$(echo $EVAL_IDS|wc -w)"

# --- resolve SFT inits STRICTLY within the regime (NO cross-regime fallback) ---
#   hide RL  <- retail_s2_<v>_heldout-*         (exclude _fullctx)
#   full RL  <- retail_s2_<v>_heldout_fullctx-*
declare -A CKPT
for v in actions_only thoughts_policy thoughts_base; do
  if [ "$REG" = hide ]; then
    c=$(ls -dt $CKROOT/retail_s2_${v}_heldout-*/step_best/adapter_model.safetensors 2>/dev/null | grep -v _fullctx | head -1)
  else
    c=$(newest "$CKROOT/retail_s2_${v}_heldout_fullctx-*/step_best/adapter_model.safetensors")
  fi
  c=${c%/adapter_model.safetensors}
  CKPT[$v]="$c"; log "  init[$v,$REG] = ${c:-<none>}"
done

declare -A INIT
INIT[retail_rlvr_base_${REG}]="base"
INIT[retail_rlvr_actions_only_${REG}]="${CKPT[actions_only]:-}"
INIT[retail_rlvr_thoughts_base_${REG}]="${CKPT[thoughts_base]:-}"
INIT[retail_rlvr_thoughts_policy_${REG}]="${CKPT[thoughts_policy]:-}"

# --- SMOKE: validate the RLVR path cheaply (1 batch, 1 task, few turns) ---
if [ "$DRY" != 1 ]; then
  SMOKE_CK="${CKPT[actions_only]:-}"
  [ -z "$SMOKE_CK" ] && { log "no SFT ckpt for smoke — abort"; exit 1; }
  log "RLVR SMOKE (1 batch, 1 task, group 2, 4 turns) ..."
  wait_gpu_free
  ./scripts/train_rl_from_sft.sh retail "$SMOKE_CK" --run_tag "retail_rlvr_smoke_${REG}" \
    --generator_config hf_rlvr --env_config tau2bench/retail_rlvr \
    --num_batches 1 --batch_size 1 --group_size 2 --max_turns 4 --discount_factor 1.0 \
    "${HIDE_ARGS[@]}" --num_test_tasks 1 --no_initial_eval > "$MDIR/smoke.log" 2>&1
  SMOKE_RC=$?
  SMOKE_M=$(newest "$S3LOGROOT/retail_rlvr_smoke_${REG}-*/metrics.jsonl")
  if [ "$SMOKE_RC" -ne 0 ] || { [ -z "$SMOKE_M" ] && ! grep -q "Rewards:" "$MDIR/smoke.log" 2>/dev/null; }; then
    log "RLVR SMOKE FAILED (rc=$SMOKE_RC). Diagnose: $MDIR/smoke.log"; exit 1
  fi
  log "RLVR SMOKE OK -> launching 4-arm matrix"
fi

# --- RL matrix (4 arms), serial + resumable (skip an arm whose adapter file exists) ---
run_one(){
  local tag="$1" init="$2"
  [ -z "$init" ] && { log "RL $tag: no SFT ckpt, skip"; return; }
  [ -n "$(newest "$S3ROOT/${tag}-*/step_best/adapter_model.safetensors")" ] && { log "RL $tag: done, skip"; return; }
  if [ "$DRY" = 1 ]; then
    echo "DRY $tag: train_rl_from_sft.sh retail $init --run_tag $tag ${RLVR_ARGS[*]} --train_task_ids <72> --eval_task_ids <20>"
    return
  fi
  log "RL $tag from $init ..."
  wait_gpu_free
  ./scripts/train_rl_from_sft.sh retail "$init" --run_tag "$tag" \
    "${RLVR_ARGS[@]}" --train_task_ids $TRAIN_IDS --eval_task_ids $EVAL_IDS_TRAIN \
    > "$MDIR/${tag}.log" 2>&1 \
    && log "RL $tag: done" || log "RL $tag: FAILED (see $MDIR/${tag}.log)"
  ./scripts/snapshot.sh "Stage-3 RLVR: $tag (retail) checkpoint" >/dev/null 2>&1 || true
}

for tag in retail_rlvr_thoughts_policy_${REG} retail_rlvr_thoughts_base_${REG} retail_rlvr_actions_only_${REG} retail_rlvr_base_${REG}; do
  run_one "$tag" "${INIT[$tag]}"
done

# --- final full-42 hold-out eval per arm (eval-only: --no_train, 1 batch = 1 eval) ---
run_eval42(){
  local tag="$1"
  local ckpt; ckpt=$(newest "$S3ROOT/${tag}-*/step_best/adapter_model.safetensors"); ckpt=${ckpt%/adapter_model.safetensors}
  [ -z "$ckpt" ] && { log "EVAL42 $tag: no RL step_best, skip"; return; }
  [ -n "$(newest "$S3LOGROOT/${tag}_eval42-*/metrics.jsonl")" ] && { log "EVAL42 $tag: done, skip"; return; }
  [ "$DRY" = 1 ] && { echo "DRY EVAL42 $tag from $ckpt"; return; }
  log "EVAL42 $tag from $ckpt (full 42 hold-out) ..."
  wait_gpu_free
  ./scripts/train_rl_from_sft.sh retail "$ckpt" --run_tag "${tag}_eval42" \
    --generator_config hf_rlvr --env_config tau2bench/retail_rlvr \
    --no_train --num_batches 1 --eval_every 1 --max_turns "$MAX_TURNS" --discount_factor 1.0 \
    "${HIDE_ARGS[@]}" --train_task_ids $TRAIN_IDS --eval_task_ids $EVAL_IDS > "$MDIR/${tag}_eval42.log" 2>&1 \
    && log "EVAL42 $tag: done" || log "EVAL42 $tag: FAILED (see $MDIR/${tag}_eval42.log)"
}
for tag in retail_rlvr_thoughts_policy_${REG} retail_rlvr_thoughts_base_${REG} retail_rlvr_actions_only_${REG} retail_rlvr_base_${REG}; do
  run_eval42 "$tag"
done
log "=== RLVR strong Stage-3 done ==="
