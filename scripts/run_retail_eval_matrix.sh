#!/usr/bin/env bash
# Stage-3 EVAL MATRIX (retail, hide regime) — the missing step-0 baselines + the
# full 42-task hold-out, in one pass per checkpoint.
#
# WHY: the RLVR matrix ran with --no_initial_eval, so its first eval is at batch 20.
# We therefore cannot tell whether RL moved anything relative to its own SFT init —
# the b=20/b=39 numbers may just be three near-identical SFT checkpoints plus noise
# (20 tasks, 1 try => ~10pp SE at p~0.3). This script evaluates every RL *starting
# point* (step 0) and every RL *result* (step_best) on the SAME task set.
#
# Eval set = all 42 tau2 retail tasks that are NEVER in the expert logs (unseen by
# Stage-1 EM, Stage-2 SFT and Stage-3 RL training). The 20 tasks the RLVR runs
# evaluated on are the first 20 of those 42, so a single 42-task pass yields BOTH
# the new 42-task number and the 20-subset that is directly comparable to the
# existing b=20/b=39 evals — via the per-task dump (see below). 42 tasks ~= 70 min.
#
# Per-task records land in <log_path>/rollouts_per_task.jsonl (added in
# trainer/train.py) — one line per rollout with the resolved tau2 task_id. That is
# what makes the 20-subset recoverable and lets us bootstrap CIs / diff arms
# task-by-task; the aggregate metrics.jsonl throws all of it away.
#
# Eval-only mode = --no_train --num_batches 1 --eval_every 1: batch 0 is also the
# last batch, so the trainer evals, then generates one throwaway train batch
# (group 2 x 1 task, ~5 min) and skips the optimizer step. Cheapest path that
# needs no new trainer code.
#
# Usage: CUDA_VISIBLE_DEVICES=0 nohup ./scripts/run_retail_eval_matrix.sh > /tmp/aprm/eval42/run.log 2>&1 &
#   DRY=1   print what would run, launch nothing
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/usr/local/bin:$PATH"

MODEL="${MODEL_CFG:-hf_qwen3_4b_instruct}"; export MODEL_CFG="$MODEL"
SFTROOT="checkpoints_lora/act_prm_tau2_retail/$MODEL"        # Stage-2 SFT inits
RLROOT="checkpoints_lora/tau2bench_retail_rlvr/$MODEL"       # Stage-3 RLVR results
LOGROOT="logs/tau2bench_retail_rlvr/$MODEL"                  # where eval runs log
MDIR=/tmp/aprm/eval42; mkdir -p "$MDIR"
DRY="${DRY:-0}"
MAX_TURNS="${MAX_TURNS:-20}"   # must match the RLVR runs for comparability
REG=hide                       # matches the RLVR matrix; hide RL <- hide SFT

log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$MDIR/eval42.log"; }
wait_gpu_free(){ while pgrep -f '[m]ain_pytorch.py' >/dev/null 2>&1; do sleep 60; done; sleep 10; }
newest(){ ls -dt $1 2>/dev/null | head -1; }

SPLIT_JSON="data/splits/tau2_retail_uid_to_tau2id.json"
TRAIN_IDS=$(python3 -c "import json;print(' '.join(json.load(open('$SPLIT_JSON'))['coverage']['covered_tau2_ids']))")
EVAL_IDS=$(python3 -c "import json;print(' '.join(json.load(open('$SPLIT_JSON'))['never_in_logs']['ids']))")
[ -n "$TRAIN_IDS" ] && [ -n "$EVAL_IDS" ] || { log "FATAL: bad split json"; exit 1; }
# One train id only: the throwaway train batch must be cheap and must not touch
# an eval task. Task 0 is in the logged/train set.
THROWAWAY_TRAIN_ID=$(echo $TRAIN_IDS | awk '{print $1}')
log "eval set: $(echo $EVAL_IDS | wc -w) never-in-logs tasks; throwaway train id=$THROWAWAY_TRAIN_ID"

# eval_one <run_tag> <ckpt-dir-or-'base'>
eval_one(){
  local tag="$1" ckpt="$2"
  [ -z "$ckpt" ] && { log "EVAL $tag: no checkpoint found, SKIP"; return; }
  if [ -n "$(newest "$LOGROOT/${tag}-*/metrics.jsonl")" ]; then
    log "EVAL $tag: already done, skip"; return
  fi
  if [ "$DRY" = 1 ]; then echo "DRY EVAL $tag <- $ckpt"; return; fi
  log "EVAL $tag <- $ckpt ..."
  wait_gpu_free
  ./scripts/train_rl_from_sft.sh retail "$ckpt" --run_tag "$tag" \
    --generator_config hf_rlvr --env_config tau2bench/retail_rlvr \
    --no_train --num_batches 1 --eval_every 1 \
    --group_size 2 --batch_size 1 \
    --max_turns "$MAX_TURNS" --max_tokens 2048 --discount_factor 1.0 --hide_observations \
    --train_task_ids $THROWAWAY_TRAIN_ID --eval_task_ids $EVAL_IDS \
    > "$MDIR/${tag}.log" 2>&1 \
    && log "EVAL $tag: done" || log "EVAL $tag: FAILED (see $MDIR/${tag}.log)"
}

log "=== retail eval matrix start (regime=$REG turns=$MAX_TURNS) ==="

# --- 1. step-0 baselines: the RL starting points, evaluated BEFORE any RL ---
# hide-regime SFT only (exclude _fullctx) — never cross regimes.
for v in actions_only thoughts_policy thoughts_base expert_thoughts; do
  c=$(ls -dt $SFTROOT/retail_s2_${v}_heldout-*/step_best/adapter_model.safetensors 2>/dev/null \
        | grep -v _fullctx | head -1)
  eval_one "retail_step0_${v}_${REG}_eval42" "${c%/adapter_model.safetensors}"
done
# Raw base model, no SFT, no RL — the absolute floor.
eval_one "retail_step0_base_${REG}_eval42" "base"

# --- 2. the RL results (step_best of the nb=40 post-hide_obs-fix matrix) ---
for v in actions_only thoughts_policy thoughts_base; do
  c=$(newest "$RLROOT/retail_rlvr_${v}_${REG}-*nb=40*/step_best/adapter_model.safetensors")
  eval_one "retail_rlvr_${v}_${REG}_eval42" "${c%/adapter_model.safetensors}"
done

log "=== retail eval matrix done ==="
