#!/usr/bin/env bash
# Stage-3 airline RL fleet — V2 (RLVR). Full re-run of all 5 arms under one config so
# they're directly comparable:  base (no SFT) + 4 SFT warm-starts.
#
# v2 changes vs run_airline_rl.sh:
#   --group_size 8 --batch_size 1   (8 concurrent rollouts, same footprint as v1's 4x2)
#   --max_turns 30                  (longer rollouts; was 20)
#   --learning_rate 1e-4            (compensates: RLVR 0/1 halves GRPO advantage vs +/-1)
#   --env_config tau2bench/airline_rlvr   (negative_rewards: false -> +1 success / 0 else)
# eval stays the 18 never-seen tau2 tasks; --num_batches 100, early-stop patience 3.
# gradient_checkpointing on (RL is user-sim-latency-bound, extra backward ~free).
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

# uv isn't installed on every box (it's per-box, not dotsynced). Fall back to the
# venv interpreter directly -- `uv run --no-sync` only selects this same interpreter.
if command -v uv >/dev/null 2>&1; then PY=(uv run --no-sync python); else PY=(.venv-tau2/bin/python); fi

MODEL=hf_qwen3_4b_instruct
SFTR=checkpoints_lora/act_prm_tau2_airline/$MODEL
CKR=checkpoints_lora/act_prm_tau2bench_airline/$MODEL
MDIR=/tmp/aprm/airline_rl_v2; mkdir -p "$MDIR"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$MDIR/orchestrator.log"; }
newest(){ ls -dt $1 2>/dev/null | head -1; }
sft_best(){ newest "$SFTR/airline_s2_${1}_heldout-*/step_best"; }

rl(){  # tag  gpu  resume_ckpt(or BASE)
  local tag=$1 gpu=$2 ckpt=$3
  if [ -n "$(newest "$CKR/${tag}-*/step_best")" ]; then log "RL $tag: skip (exists)"; return 0; fi
  local resume=()
  if [ "$ckpt" != BASE ]; then
    [ -n "$ckpt" ] && [ -f "$ckpt/adapter_model.safetensors" ] || { log "RL $tag: no ckpt ($ckpt), skip"; return 1; }
    resume=(--resume_from "$ckpt")
  fi
  log "RL $tag (GPU $gpu) ${ckpt/#*\//} ${resume:+warm-start} "
  CUDA_VISIBLE_DEVICES="$gpu" "${PY[@]}" main_pytorch.py \
    --env_config tau2bench/airline_rlvr --model_config $MODEL --lora_config r8_a16_linear \
    --generator_config hf_rlvr --trainer_config pg --replay_buffer_config default "${resume[@]}" \
    --group_size 8 --batch_size 1 --max_turns 30 --max_tokens 2048 --learning_rate 1e-4 \
    --num_batches 100 --eval_every 10 --no_initial_eval --gradient_checkpointing \
    --best_metric final_reward --early_stop_patience 3 --run_tag "$tag" --verbose > "$MDIR/${tag}.log" 2>&1 \
    && log "RL $tag: done" || log "RL $tag: FAILED (see $MDIR/${tag}.log)"
}
# a serial stream of "tag|ckpt" jobs pinned to one GPU
stream(){ local gpu=$1; shift; for j in "$@"; do IFS='|' read -r tag ckpt <<< "$j"; rl "$tag" "$gpu" "$ckpt"; done; }

log "=== airline RL v2 (RLVR) fleet: base + 4 SFT variants, gs8/bs1 mt30 lr1e-4, train=32 / eval=18, 1 run/GPU ==="
# 1 run/GPU (a single mt=30 rollout context can fill an 80GB card). Two SERIAL streams
# (3 arms on GPU0, 2 on GPU1). If any run OOMs, add --obs_max_chars to cap observations.
stream 0 "airline_rlvr_base|BASE" "airline_rlvr_thoughts_policy|$(sft_best thoughts_policy)" "airline_rlvr_thoughts_base|$(sft_best thoughts_base)" & s0=$!
stream 1 "airline_rlvr_actions_only|$(sft_best actions_only)" "airline_rlvr_expert_thoughts|$(sft_best expert_thoughts)"                          & s1=$!
wait $s0; wait $s1
log "=== airline RL v2 fleet complete ==="
