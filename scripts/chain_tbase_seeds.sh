#!/usr/bin/env bash
# 4 extra seeds on insurance thoughts_base/step_0020, to make base-vs-policy testable.
#
# POWER: insurance seed sd = 3.1pt, so detecting the observed 5pt gap needs ~6 runs/arm.
# thoughts_policy already has 4 seeds (52.5/55.0/47.5/55.0); thoughts_base has 1 (47.5).
# These 4 bring it to n=5 vs n=4 -- enough for a real test instead of "within noise".
#
# Calls main_pytorch DIRECTLY: run_insurance_rollout.sh has no EXTRA_ARGS passthrough, so
# a --seed handed to it is silently dropped and you get identical duplicates (the snorkel
# gyms are deterministic given a seed -- see CLAUDE.md).
set -uo pipefail
cd /home/mzio/projects/act-prm-blog
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/usr/local/bin:$PATH"
[ -f scripts/wandb_preflight.sh ] && . scripts/wandb_preflight.sh
export PYTHONUNBUFFERED=1 HF_HOME=/data/users/mzio/models/hf_cache HF_HUB_OFFLINE=1
export WANDB_ERROR_REPORTING=false ACT_PRM_DUMP_TRAJECTORIES=1
export https_proxy="${https_proxy:-http://fwdproxy:8080}" http_proxy="${http_proxy:-http://fwdproxy:8080}"
G=/tmp/aprm/tbase_seeds; mkdir -p "$G"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$G/chain.log"; }
reap(){ for p in $(pgrep -f 'main_pytorch\.py'); do kill -9 "$p" 2>/dev/null; done; sleep 15; }
newest(){ ls -dt $1 2>/dev/null | head -1; }
IDS=$(python3 -c "import json;print(' '.join(str(i) for i in json.load(open('data/splits/snorkel_insurance_uid_to_task.json'))['eval_task_ids']))")
CK=$(newest "checkpoints_lora/act_prm_snorkel_insurance/hf_qwen3_4b_instruct/snorkel_insurance_s2_thoughts_base_adamw30_lr1e_3_nb200_flat32sgd_heldout-*/step_0020")
[ -z "$CK" ] && { log "FATAL: no thoughts_base step_0020"; exit 1; }
for SEED in 1234 777 555 999; do
  m="$G/seed$SEED.done"; [ -f "$m" ] && continue
  TAG="insurance_rollout_thoughts_base_adamw30_seed${SEED}_step_0020"
  log "SEED $SEED  <- $CK"
  CLAUDECODE= UV_PROJECT_ENVIRONMENT=.venv-tau2 uv run --no-sync python main_pytorch.py \
      --env_config act_prm/snorkel_insurance_gym --model_config hf_qwen3_4b_instruct \
      --lora_config r8_a16_linear --generator_config hf_grpo --trainer_config pg \
      --replay_buffer_config default --resume_from "$CK" \
      --no_train --num_batches 1 --eval_every 1 --group_size 2 --batch_size 1 \
      --max_tokens 2048 --hide_observations --run_tag "$TAG" \
      --eval_task_ids $IDS --seed "$SEED" --verbose >>"$G/seed$SEED.log" 2>&1
  log "  rc=$?"; touch "$m"; reap
done
log "=== thoughts_base seed sweep complete ==="
