#!/usr/bin/env bash
# 1-task smoke test of the Snorkel-insurance gym on the BASE model (no --resume_from).
#
# Why base and not a checkpoint: this validates the HARNESS (env constructs, tools
# respond, LLM judge grades, rollout writes a score), independently of whether any
# Stage-2 checkpoint exists yet. The gym has never produced a rollout log, so a failure
# here is a harness bug, not a policy result -- and finding it costs ~5 min instead of
# discovering it 10h later when the insurance rollouts finally start.
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/usr/local/bin:$PATH"
export https_proxy="${https_proxy:-http://fwdproxy:8080}"
export http_proxy="${http_proxy:-http://fwdproxy:8080}"
export HF_TOKEN="${HF_TOKEN:-$(cat "$HOME/models/token" 2>/dev/null || true)}"
# fwdproxy began 403-ing huggingface.co on 2026-09-03, which killed model loading in
# load_hf_model_and_tokenizer -> hf_hub list_repo_tree (all 4 expert_thoughts_all runs
# and the insurance base rollout died in ~25s). The weights are cached locally under
# HF_HOME, so go offline and never touch the Hub. Verified: AutoConfig+AutoTokenizer
# for Qwen3-4B-Instruct-2507 load fine with HF_HUB_OFFLINE=1.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_HOME="${HF_HOME:-/data/users/mzio/models/hf_cache}"
export ACT_PRM_DUMP_TRAJECTORIES=1
MODEL="${MODEL_CFG:-hf_qwen3_4b_instruct}"; export MODEL_CFG="$MODEL"
MDIR=/tmp/aprm/insurance_rollout; mkdir -p "$MDIR"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$MDIR/smoke.log"; }

ID=$(python3 -c "import json;print(json.load(open('data/splits/snorkel_insurance_uid_to_task.json'))['eval_task_ids'][0])")
TAG="insurance_smoke_base"
log "=== insurance gym SMOKE: 1 task (id=$ID), base model, max_turns 10 ==="
CLAUDECODE= UV_PROJECT_ENVIRONMENT=.venv-tau2 uv run --no-sync python main_pytorch.py \
    --env_config act_prm/snorkel_insurance_gym --model_config "$MODEL" \
    --lora_config r8_a16_linear --generator_config hf_grpo --trainer_config pg \
    --replay_buffer_config default \
    --no_train --num_batches 1 --eval_every 1 --group_size 1 --batch_size 1 \
    --max_tokens 2048 --max_turns 10 --hide_observations --run_tag "$TAG" \
    --eval_task_ids "$ID" --verbose \
    > "$MDIR/${TAG}.log" 2>&1
rc=$?
D=$(ls -dt logs/act_prm_snorkel_insurance_gym/$MODEL/${TAG}-*/ 2>/dev/null | head -1)
log "rc=$rc  logdir=${D:-NONE}"
if [ -n "$D" ] && [ -f "$D/rollouts_per_task.jsonl" ]; then
  n=$(grep -c . "$D/rollouts_per_task.jsonl")
  log "SMOKE PASS: $n scored row(s)"
  python3 -c "
import json
for l in open('$D/rollouts_per_task.jsonl'):
    d=json.loads(l); print('   ', {k:d.get(k) for k in ('split','task_id','final_reward','timesteps')})
" | tee -a "$MDIR/smoke.log"
  touch "$MDIR/SMOKE_OK"
else
  log "SMOKE FAIL: no rollouts_per_task.jsonl -- see $MDIR/${TAG}.log"
  tail -25 "$MDIR/${TAG}.log" | tee -a "$MDIR/smoke.log"
fi
