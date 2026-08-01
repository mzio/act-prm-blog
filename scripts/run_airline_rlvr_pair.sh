#!/usr/bin/env bash
# Stage-3 airline RLVR — the two-arm head-to-head:
#     thoughts_policy (Act-PRM)  vs  actions_only (baseline)
# both hide-obs, both warm-started from their MATCHING hide-obs SFT checkpoint.
#
# Config (as specified):
#   --group_size 8 --batch_size 1        8 rollouts/group (~60 GiB peak at the M-step on an
#                                        80 GiB card, so ONE run per GPU -- do not co-locate)
#   --max_turns 30                       long rollouts
#   --num_batches 100                    100 RL updates ("max_steps 100")
#   hide-obs                             env airline_rlvr (hide_observations:true, last_obs_to_show:1)
#   RLVR                                 generator hf_rlvr (mean_center:false, discount_factor:1.0)
#                                        + env negative_rewards:false -> advantage == +1 success / 0 fail
#   --gradient_checkpointing             on (RL is user-sim-latency-bound; backward cost ~free)
#   no observation truncation            (no --obs_max_chars)
#   eval = the 18 never-seen tau2 tasks, every 10 batches, early-stop patience 3
#
# REGIME GUARD: sft_best() globs airline_s2_<v>_heldout-* which EXCLUDES *_heldout_fullctx,
# so a hide-obs RL run can never warm-start from a full-context SFT checkpoint.
#
# This box has ONE GPU, so the arms run SERIALLY (thoughts_policy first -- it's the arm
# under test). Set GPU_LIST="0 1" on a 2-GPU box to run them in parallel instead.
#
# Usage:  setsid nohup ./scripts/run_airline_rlvr_pair.sh > /tmp/aprm/airline_rlvr_pair/run.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/usr/local/bin:$PATH"
export UV_PROJECT_ENVIRONMENT=.venv-tau2 UV_FROZEN=1
export HF_HOME="${HF_HOME:-/data/users/$USER/models/hf_cache}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1
export https_proxy=http://fwdproxy:8080 http_proxy=http://fwdproxy:8080
export TAU2_DATA_DIR="$(pwd)/tau2-bench/data"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# uv is per-box and may be absent; `uv run --no-sync` just selects this interpreter anyway.
if command -v uv >/dev/null 2>&1; then PY=(uv run --no-sync python); else PY=(.venv-tau2/bin/python); fi

MODEL=hf_qwen3_4b_instruct
SFTR=checkpoints_lora/act_prm_tau2_airline/$MODEL
CKR=checkpoints_lora/tau2bench_airline_rlvr/$MODEL
MDIR=/tmp/aprm/airline_rlvr_pair; mkdir -p "$MDIR"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$MDIR/orchestrator.log"; }
newest(){ ls -dt $1 2>/dev/null | head -1; }
sft_best(){ newest "$SFTR/airline_s2_${1}_heldout-*/step_best"; }   # hide-obs only, never _fullctx

rl(){  # tag  gpu  resume_ckpt
  local tag=$1 gpu=$2 ckpt=$3
  if [ -n "$(newest "$CKR/${tag}-*/step_best")" ]; then log "RL $tag: skip (already has step_best)"; return 0; fi
  if [ -z "$ckpt" ] || [ ! -f "$ckpt/adapter_model.safetensors" ]; then
    log "RL $tag: FAILED -- no hide-obs SFT ckpt ($ckpt)"; return 1
  fi
  log "RL $tag (GPU $gpu) warm-start <- $(basename "$(dirname "$ckpt")" | cut -c1-46)"
  CUDA_VISIBLE_DEVICES="$gpu" "${PY[@]}" main_pytorch.py \
    --env_config tau2bench/airline_rlvr --model_config $MODEL --lora_config r8_a16_linear \
    --generator_config hf_rlvr --trainer_config pg --replay_buffer_config default \
    --resume_from "$ckpt" \
    --group_size 8 --batch_size 1 --max_turns 30 --max_tokens 2048 --learning_rate 1e-4 \
    --num_batches 100 --eval_every 10 --no_initial_eval --gradient_checkpointing \
    --best_metric final_reward --early_stop_patience 3 --run_tag "$tag" --verbose \
    > "$MDIR/${tag}.log" 2>&1 \
    && log "RL $tag: done" || log "RL $tag: FAILED (see $MDIR/${tag}.log)"
}
stream(){ local gpu=$1; shift; for j in "$@"; do IFS='|' read -r tag ckpt <<< "$j"; rl "$tag" "$gpu" "$ckpt"; done; }

# ARMS selects which arms to run (space/comma separated), so a second box can take one
# arm while the first runs the other:
#   ARMS=actions_only    ./scripts/run_airline_rlvr_pair.sh      # box B
#   ARMS=thoughts_policy ./scripts/run_airline_rlvr_pair.sh      # box A
# Default runs both, in order. Valid: thoughts_policy actions_only expert_thoughts thoughts_base
ARMS="${ARMS:-thoughts_policy actions_only}"
JOBS=()
for a in ${ARMS//,/ }; do
  ck="$(sft_best "$a")"
  JOBS+=("airline_rlvr_${a}|$ck")
done
GPUS=(${GPU_LIST:-$(nvidia-smi --query-gpu=index --format=csv,noheader | tr -d ' ' | tr '\n' ' ')})
NG=${#GPUS[@]}
log "=== airline RLVR pair (hide-obs, gs8 mt30 nb100): ${#JOBS[@]} arms over $NG GPU(s): ${GPUS[*]} ==="
pids=()
for ((g=0; g<NG; g++)); do
  slice=(); for ((j=g; j<${#JOBS[@]}; j+=NG)); do slice+=("${JOBS[j]}"); done
  [ ${#slice[@]} -eq 0 ] && continue
  stream "${GPUS[g]}" "${slice[@]}" & pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
log "=== airline RLVR pair complete ==="
