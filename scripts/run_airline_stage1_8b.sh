#!/usr/bin/env bash
# Qwen3-8B Stage-1 Act-PRM EM thought-gen for airline (policy + base), to test
# whether a bigger model yields IMPROVING held-out thoughts (the 4B's EM eval reward
# was flat/declining). WAITS for the GPUs to free (i.e. the 4B sweep to finish),
# then runs both scorers in parallel (one per GPU). Offline, uncapped, grad-ckpt.
# Resumable: skips a scorer whose step_best already exists.
#
# Usage:  nohup ./scripts/run_airline_stage1_8b.sh > /tmp/aprm/airline_8b/run.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/usr/local/bin:$PATH"
export HF_HOME="${HF_HOME:-/data/users/$USER/models/hf_cache}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1 UV_FROZEN=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

MODEL=hf_qwen3_8b
CKR=checkpoints_lora/act_prm_tau2_airline/$MODEL
MDIR=/tmp/aprm/airline_8b; mkdir -p "$MDIR"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$MDIR/orchestrator.log"; }
newest(){ ls -dt $1 2>/dev/null | head -1; }
wait_gpu_free(){ while pgrep -f '[m]ain_pytorch.py' >/dev/null 2>&1; do sleep 60; done; sleep 10; }

em(){  # scorer gpu
  local scorer=$1 gpu=$2 swb tag="airline_s1_8b_$1"
  [ "$scorer" = base ] && swb=--score_with_base || swb=--no-score_with_base
  if [ -n "$(newest "$CKR/${tag}-*/step_best")" ]; then log "8B $scorer: step_best exists, skip"; return 0; fi
  log "8B EM $scorer (GPU $gpu)"
  CUDA_VISIBLE_DEVICES="$gpu" ./scripts/train.sh --env_config act_prm/tau2_airline \
    --generator_config act_prm --trainer_config pg --model_config $MODEL \
    --lora_config r8_a16_linear --replay_buffer_config default $swb \
    --group_size 4 --batch_size 4 --num_batches 25 --eval_every 5 --no_initial_eval \
    --length_penalty 0.15 --gradient_checkpointing --save_generations \
    --run_tag "$tag" --verbose > "$MDIR/s1_8b_${scorer}.log" 2>&1 \
    && log "8B $scorer: done" || log "8B $scorer: FAILED (see $MDIR/s1_8b_${scorer}.log)"
}

log "=== 8B Stage-1 EM: waiting for GPUs to free (4B sweep to finish) ==="
wait_gpu_free
log "GPUs free — launching 8B EM (policy GPU0, base GPU1)"
em policy 0 & p0=$!
sleep 60   # stagger the two 8B model loads (16GB each) to avoid a load-time spike
em base   1 & p1=$!
wait $p0; wait $p1
log "=== 8B Stage-1 EM complete — analyze curves, then decide on relabel+SFT ==="
