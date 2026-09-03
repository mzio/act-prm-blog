#!/usr/bin/env bash
# Base-model (no LoRA) rollout on the 40 held-out insurance tasks.
#
# WHY THIS IS NEEDED: retail has a base reference (14.3%) and airline has one (55.6%), but
# insurance has NONE -- the gym was only unblocked on 2026-09-02. Without it the insurance
# Stage-2 numbers are uninterpretable: thoughts_policy/step_0020 scored 52.5% (21/40) and
# we cannot say whether that is above or below an untrained Qwen3-4B. The smoke test does
# NOT serve as a reference (1 task, max_turns 10, deliberately under the expert p50 of 12).
#
# Identical to run_insurance_rollout.sh except: no --resume_from, and the tag says base.
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/usr/local/bin:$PATH"
export https_proxy="${https_proxy:-http://fwdproxy:8080}"
export http_proxy="${http_proxy:-http://fwdproxy:8080}"
export HF_TOKEN="${HF_TOKEN:-$(cat "$HOME/models/token" 2>/dev/null || true)}"
export ACT_PRM_DUMP_TRAJECTORIES=1
MODEL="${MODEL_CFG:-hf_qwen3_4b_instruct}"; export MODEL_CFG="$MODEL"
MDIR=/tmp/aprm/insurance_rollout; mkdir -p "$MDIR"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$MDIR/rollout.log"; }
newest(){ ls -dt $1 2>/dev/null | head -1; }

IDS=$(python3 -c "import json;print(' '.join(str(i) for i in json.load(open('data/splits/snorkel_insurance_uid_to_task.json'))['eval_task_ids']))")
TAG="insurance_rollout_BASE"
[ -f "$MDIR/${TAG}.done" ] && { log "$TAG done, skip"; exit 0; }
log "=== insurance BASE rollout: $(echo $IDS|wc -w) tasks (no LoRA) ==="
CLAUDECODE= UV_PROJECT_ENVIRONMENT=.venv-tau2 uv run --no-sync python main_pytorch.py \
    --env_config act_prm/snorkel_insurance_gym --model_config "$MODEL" \
    --lora_config r8_a16_linear --generator_config hf_grpo --trainer_config pg \
    --replay_buffer_config default \
    --no_train --num_batches 1 --eval_every 1 --group_size 2 --batch_size 1 \
    --max_tokens 2048 --hide_observations --run_tag "$TAG" \
    --eval_task_ids $IDS --verbose \
    > "$MDIR/${TAG}.log" 2>&1 || { log "$TAG FAILED"; exit 1; }
D=$(newest "logs/act_prm_snorkel_insurance_gym/$MODEL/${TAG}-*/")
uv run --no-project python scripts/check_rollout_valid.py "$D" && { touch "$MDIR/${TAG}.done"; log "$TAG done"; } \
  || log "$TAG INVALID; not marking done"
