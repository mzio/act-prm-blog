#!/usr/bin/env bash
# Stage-3 airline RL fleet: warm-start from each SFT variant's step_best (+ base, no SFT)
# and RL against the tau2 airline gym (claude_agent_sdk user-sim). Trains on the 32 tasks
# present in the Act-PRM logs, evals on the 18 never-seen tau2 tasks
# (configs/environments/tau2bench/airline.yaml -> task_id_map_file). This tests whether the
# offline action-subspan lift (SFT-on-thoughts) translates to actual TASK SUCCESS.
#
# RL is LLM-gated (each turn waits on the user-sim), so runs co-locate 2/GPU. Offline for
# the policy model; proxy on for the user-sim. Resumable (skips a run whose step_best exists).
#
# Usage:  nohup ./scripts/run_airline_rl.sh > /tmp/aprm/airline_rl/run.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/usr/local/bin:$PATH"
export UV_PROJECT_ENVIRONMENT=.venv-tau2 UV_FROZEN=1
export HF_HOME="${HF_HOME:-/data/users/$USER/models/hf_cache}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1          # policy model offline
export https_proxy=http://fwdproxy:8080 http_proxy=http://fwdproxy:8080      # user-sim (claude_agent_sdk/Vertex)
export TAU2_DATA_DIR="$(pwd)/tau2-bench/data"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

MODEL=hf_qwen3_4b_instruct
SFTR=checkpoints_lora/act_prm_tau2_airline/$MODEL
CKR=checkpoints_lora/act_prm_tau2bench_airline/$MODEL
MDIR=/tmp/aprm/airline_rl; mkdir -p "$MDIR"
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
  CUDA_VISIBLE_DEVICES="$gpu" uv run --no-sync python main_pytorch.py \
    --env_config tau2bench/airline --model_config $MODEL --lora_config r8_a16_linear \
    --generator_config hf_grpo --trainer_config pg --replay_buffer_config default "${resume[@]}" \
    --group_size 4 --batch_size 2 --max_turns 20 --max_tokens 2048 \
    --num_batches 100 --eval_every 10 --no_initial_eval --gradient_checkpointing \
    --best_metric final_reward --early_stop_patience 3 --run_tag "$tag" --verbose > "$MDIR/${tag}.log" 2>&1 \
    && log "RL $tag: done" || log "RL $tag: FAILED (see $MDIR/${tag}.log)"
}
# a serial stream of "tag|ckpt" jobs pinned to one GPU
stream(){ local gpu=$1; shift; for j in "$@"; do IFS='|' read -r tag ckpt <<< "$j"; rl "$tag" "$gpu" "$ckpt"; done; }

log "=== airline RL fleet: base + 4 SFT variants (hide regime), train=32 / eval=18, 1 run/GPU ==="
# 1 run/GPU: a single RL rollout context can balloon to ~40GB (long tau2 tool chains at
# max_turns 20), so 2/GPU OOMs. Two SERIAL streams, one per GPU (3 on GPU0, 2 on GPU1).
stream 0 "airline_rl_base|BASE" "airline_rl_thoughts_policy|$(sft_best thoughts_policy)" "airline_rl_thoughts_base|$(sft_best thoughts_base)" & s0=$!
stream 1 "airline_rl_actions_only|$(sft_best actions_only)" "airline_rl_expert_thoughts|$(sft_best expert_thoughts)"                          & s1=$!
wait $s0; wait $s1
log "=== airline RL fleet complete ==="
