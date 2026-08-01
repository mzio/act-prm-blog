#!/usr/bin/env bash
# Stage-3 airline RL — V2 (RLVR + hide-obs). All 5 arms under one config so they are
# directly comparable: base (no SFT) + 4 SFT warm-starts.
#
# Config:
#   --group_size 8 --batch_size 1   (8 concurrent rollouts)
#   --max_turns 30
#   --learning_rate 1e-4            (RLVR 0/1 halves the GRPO advantage vs +/-1, so 2x LR)
#   --env_config tau2bench/airline_rlvr   (negative_rewards:false + hide_observations:true)
#   eval = the 18 never-seen tau2 tasks; 100 batches; early-stop patience 3.
# Warm-starts are the HIDE-OBS SFT ckpts (airline_s2_*_heldout); the glob deliberately
# EXCLUDES *_heldout_fullctx so a hide-obs RL run can never start from a full-context SFT.
#
# GPU_LIST env var controls placement (default: all visible GPUs). One run per GPU:
# a single run peaks ~60 GiB (M-step backward over 8 rollouts) on an 80 GiB card.
#
# Usage:  nohup ./scripts/run_airline_rl_v2.sh > /tmp/aprm/airline_rl_v2/run.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/usr/local/bin:$PATH"
export UV_PROJECT_ENVIRONMENT=.venv-tau2 UV_FROZEN=1
export HF_HOME="${HF_HOME:-/data/users/$USER/models/hf_cache}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1          # policy model offline
export https_proxy=http://fwdproxy:8080 http_proxy=http://fwdproxy:8080      # user-sim (claude_agent_sdk)
export TAU2_DATA_DIR="$(pwd)/tau2-bench/data"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

MODEL=hf_qwen3_4b_instruct
SFTR=checkpoints_lora/act_prm_tau2_airline/$MODEL
CKR=checkpoints_lora/tau2bench_airline_rlvr/$MODEL
MDIR=/tmp/aprm/airline_rl_v2; mkdir -p "$MDIR"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$MDIR/orchestrator.log"; }
newest(){ ls -dt $1 2>/dev/null | head -1; }
sft_best(){ newest "$SFTR/airline_s2_${1}_heldout-*/step_best"; }   # hide-obs only (no _fullctx)

rl(){  # tag  gpu  resume_ckpt(or BASE)
  local tag=$1 gpu=$2 ckpt=$3
  if [ -n "$(newest "$CKR/${tag}-*/step_best")" ]; then log "RL $tag: skip (exists)"; return 0; fi
  local resume=()
  if [ "$ckpt" != BASE ]; then
    [ -n "$ckpt" ] && [ -f "$ckpt/adapter_model.safetensors" ] || { log "RL $tag: no ckpt ($ckpt), skip"; return 1; }
    resume=(--resume_from "$ckpt")
  fi
  log "RL $tag (GPU $gpu) ${resume:+warm-start }${ckpt##*/act_prm_tau2_airline/}"
  CUDA_VISIBLE_DEVICES="$gpu" uv run --no-sync python main_pytorch.py \
    --env_config tau2bench/airline_rlvr --model_config $MODEL --lora_config r8_a16_linear \
    --generator_config hf_grpo --trainer_config pg --replay_buffer_config default "${resume[@]}" \
    --group_size 8 --batch_size 1 --max_turns 30 --max_tokens 2048 --learning_rate 1e-4 \
    --num_batches 100 --eval_every 10 --no_initial_eval --gradient_checkpointing \
    --best_metric final_reward --early_stop_patience 3 --run_tag "$tag" --verbose > "$MDIR/${tag}.log" 2>&1 \
    && log "RL $tag: done" || log "RL $tag: FAILED (see $MDIR/${tag}.log)"
}
stream(){ local gpu=$1; shift; for j in "$@"; do IFS='|' read -r tag ckpt <<< "$j"; rl "$tag" "$gpu" "$ckpt"; done; }

# Job list, most important first (so a 1-GPU box still gets the key comparison soonest).
JOBS=(
  "airline_rlvr_thoughts_policy|$(sft_best thoughts_policy)"
  "airline_rlvr_actions_only|$(sft_best actions_only)"
  "airline_rlvr_expert_thoughts|$(sft_best expert_thoughts)"
  "airline_rlvr_thoughts_base|$(sft_best thoughts_base)"
  "airline_rlvr_base|BASE"
)
GPUS=(${GPU_LIST:-$(nvidia-smi --query-gpu=index --format=csv,noheader | tr '\n' ' ')})
NG=${#GPUS[@]}
log "=== airline RL v2 (RLVR + hide-obs): ${#JOBS[@]} arms over $NG GPU(s): ${GPUS[*]} ==="
pids=()
for ((g=0; g<NG; g++)); do
  # round-robin the job list across GPUs; each GPU runs its slice serially
  slice=(); for ((j=g; j<${#JOBS[@]}; j+=NG)); do slice+=("${JOBS[j]}"); done
  stream "${GPUS[g]}" "${slice[@]}" & pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
log "=== airline RL v2 fleet complete ==="
