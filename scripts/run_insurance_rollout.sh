#!/usr/bin/env bash
# Rollout eval of Snorkel-insurance Stage-2 checkpoints on the 40 held-out underwriting
# tasks. Modelled on run_finance_rollout.sh.
#
# NEW HARNESS (2026-09-02): the insurance gym had never been exercised for a rollout --
# logs/act_prm_snorkel_insurance_gym/ was empty. cc-11.0 listed insurance rollouts as
# blocked on the rl_eval uid -> gym task-id map; that map now exists and is COMPLETE
# (data/splits/snorkel_insurance_uid_to_task.json: n_mapped=261, n_ambiguous=0,
# n_missing=0), and the gym data is present locally, so the GitHub-clone blocker is moot.
# Run scripts/smoke_insurance_gym.sh before trusting a full pass.
#
# max_turns: NOT overridden. The gym config already defaults to 28, and the 180 expert
# trajectories run p50=12, p90=18, p95=19, max=20 assistant turns -- so 28 covers 100% of
# expert demonstrations. (Contrast finance, whose env default of 12 covered only 61% and
# made the cap, not the policy, the binding constraint: every arm scored 0/10.)
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/usr/local/bin:$PATH"
# This driver calls main_pytorch.py directly, so it does not inherit any proxy setup.
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
export ACT_PRM_DUMP_TRAJECTORIES="${ACT_PRM_DUMP_TRAJECTORIES:-1}"
MODEL="${MODEL_CFG:-hf_qwen3_4b_instruct}"; export MODEL_CFG="$MODEL"
MDIR=/tmp/aprm/insurance_rollout; mkdir -p "$MDIR"
ARMS="${ARMS:-actions_only thoughts_policy_adamw30}"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$MDIR/rollout.log"; }
wait_gpu_free(){ while pgrep -f '[m]ain_pytorch.py' >/dev/null 2>&1; do sleep 60; done; sleep 10; }
newest(){ ls -dt $1 2>/dev/null | head -1; }

IDS=$(python3 -c "import json;print(' '.join(str(i) for i in json.load(open('data/splits/snorkel_insurance_uid_to_task.json'))['eval_task_ids']))")
log "=== insurance rollout: $(echo $IDS|wc -w) held-out tasks, arms='$ARMS' ==="

for arm in $ARMS; do
  CK=$(newest "checkpoints_lora/act_prm_snorkel_insurance/$MODEL/snorkel_insurance_s2_${arm}_${CKPT_PAT:?set CKPT_PAT}_heldout-*/${CKPT_STEP:-step_best}")
  [ -z "$CK" ] && { log "insurance/$arm: no checkpoint for CKPT_PAT=$CKPT_PAT, skip"; continue; }
  # TAG encodes generation AND snapshot, or a re-run collides with a prior one: same log
  # dir (metrics appended) and same .done marker (arm silently skipped).
  TAG="insurance_rollout_${arm}_${CKPT_TAG:-sgd}${CKPT_STEP:+_${CKPT_STEP}}"
  [ -f "$MDIR/${TAG}.done" ] && { log "$TAG: done, skip"; continue; }
  log "ROLLOUT $TAG <- $CK"
  wait_gpu_free
  CLAUDECODE= UV_PROJECT_ENVIRONMENT=.venv-tau2 uv run --no-sync python main_pytorch.py \
      --env_config act_prm/snorkel_insurance_gym --model_config "$MODEL" \
      --lora_config r8_a16_linear --generator_config hf_grpo --trainer_config pg \
      --replay_buffer_config default --resume_from "$CK" \
      --no_train --num_batches 1 --eval_every 1 --group_size 2 --batch_size 1 \
      --max_tokens 2048 --hide_observations --run_tag "$TAG" \
      --eval_task_ids $IDS --verbose \
      > "$MDIR/${TAG}.log" 2>&1 \
    || { log "$TAG: FAILED (see $MDIR/${TAG}.log)"; continue; }
  D=$(newest "logs/act_prm_snorkel_insurance_gym/$MODEL/${TAG}-*/")
  if uv run --no-project python scripts/check_rollout_valid.py "$D"; then
    touch "$MDIR/${TAG}.done"; log "$TAG: done"
  else
    log "$TAG: INVALID (judge/user-sim outage?); not marking done"
  fi
done
log "=== insurance rollout pass finished ==="
