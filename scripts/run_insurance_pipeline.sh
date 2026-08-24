#!/usr/bin/env bash
# Full Act-PRM pipeline on snorkel_insurance: Stage-1 EM -> Stage-2 SFT -> rollout eval.
#
# Data (scripts/build_insurance_pools.py, already run):
#   source  mzio/aprm-insurance-gpt5m_med-gs4-s0-r1-train -- GPT-5-mini rollouts,
#           910/1000 trajectories successful (91%), 261/262 questions
#   split   QUESTION-level and three-way disjoint: 180 train / 40 eval / 41 rollout.
#           The rollout questions are never seen by Stage 1 or Stage 2.
#   pools   ONE demonstration per question -> 2,273 train / 540 eval targets, IDENTICAL
#           across all four arms. Retail kept two generations per task, which gave its
#           thought arms 1043 targets against the baseline's 635 and forced a separate
#           volume-matched control; one-per-question removes that confound by construction.
#
# Arms (all matched step-for-step, no --require_thought):
#   actions_only     expert action, no reasoning            (behavioural-cloning baseline)
#   expert_thoughts  GPT-5-mini's own reasoning + action    (the oracle)
#   thoughts_policy  Act-PRM inferred thought + action, policy-scored
#   thoughts_base    Act-PRM inferred thought + action, base-scored
#
# There is no expert_thoughts_all arm here: 220/220 selected trajectories already carry at
# least one reasoning step, so the filter that arm applies is inert on this domain.
#
# Stages:
#   1  EM thought generation over the 180 train questions, twice (policy- and base-scored),
#      exported to SFT corpora.  --no_train: this is a generate-only relabel pass.
#   2  SFT per arm, hide-observations, lr 3e-3, tracking train + eval action-span PPL and
#      accuracy. Uses the SAME lr the other three domains converged at (probed: 1e-4 dead,
#      1e-3 +1.19%, 3e-3 +6.94% over 30 batches).
#   3  Rollout eval on the 41 held-out questions, per arm, in the gym.
#
# Usage: ./scripts/run_insurance_pipeline.sh          (cron drives it via sweep_guard.sh)
#        STAGES=2 ./scripts/run_insurance_pipeline.sh (only Stage 2)
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/usr/local/bin:$PATH"
# HF access: the pools are on disk, but the model/tokenizer load still touches the hub, and
# cron has no proxy -- that killed all ten finance rollouts on HF DNS. Xet CAS is not
# reachable through fwdproxy either, hence HF_HUB_DISABLE_XET.
export https_proxy="${https_proxy:-http://fwdproxy:8080}"
export http_proxy="${http_proxy:-http://fwdproxy:8080}"
export HF_TOKEN="${HF_TOKEN:-$(cat "$HOME/models/token" 2>/dev/null || true)}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

MODEL="${MODEL_CFG:-hf_qwen3_4b_instruct}"; export MODEL_CFG="$MODEL"
MDIR="${MDIR:-/tmp/aprm/insurance}"; mkdir -p "$MDIR"
_fail=0
NB="${NUM_BATCHES:-150}"
LR="${LR:-3e-3}"; LRTAG="${LR//[-.]/_}"
ENVCFG=act_prm/snorkel_insurance
STAGES="${STAGES:-1 2 3}"
ARMS="${ARMS:-actions_only expert_thoughts thoughts_policy thoughts_base}"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$MDIR/insurance.log"; }
wait_gpu_free(){ while pgrep -f '[m]ain_pytorch.py' >/dev/null 2>&1; do sleep 60; done; sleep 10; }
newest(){ ls -dt $1 2>/dev/null | head -1; }
has(){ case " $STAGES " in *" $1 "*) return 0;; *) return 1;; esac; }

# ---------------------------------------------------------------- Stage 1: EM relabel
if has 1; then
for scorer in policy base; do
  TAG="insurance_s1_${scorer}"
  CORPUS="data/sft_corpus/snorkel_insurance/${scorer}"
  if [ -s "$CORPUS/train.json" ]; then log "$TAG: corpus exists, skip"; continue; fi
  if [ -f "$MDIR/${TAG}.done" ]; then log "$TAG: done, skip"; continue; fi
  SCORE_FLAG=--score_with_base; [ "$scorer" = policy ] && SCORE_FLAG=--no-score_with_base
  log "STAGE-1 EM $TAG (180 train questions, ${scorer}-scored)"
  wait_gpu_free
  ./scripts/train.sh --env_config "$ENVCFG" --generator_config act_prm --trainer_config pg \
      --run_tag "$TAG" --no_train "$SCORE_FLAG" \
      --group_size 4 --batch_size 2 --length_penalty 0.15 \
      > "$MDIR/${TAG}.log" 2>&1 \
    || { _fail=$((_fail+1)); log "$TAG: FAILED (see $MDIR/${TAG}.log)"; continue; }
  GEN=$(newest "logs/act_prm_snorkel_insurance/$MODEL/${TAG}-*/generations.jsonl")
  [ -z "$GEN" ] && { _fail=$((_fail+1)); log "$TAG: no generations.jsonl"; continue; }
  uv run --no-project python scripts/export_sft_corpus.py \
      --generations "$GEN" --source-pools data/snorkel_insurance_split \
      --out "$CORPUS" --advantage_mode best >> "$MDIR/${TAG}.log" 2>&1 \
    || { _fail=$((_fail+1)); log "$TAG: corpus export FAILED"; continue; }
  touch "$MDIR/${TAG}.done"; log "$TAG: done -> $CORPUS"
done
fi

# ---------------------------------------------------------------- Stage 2: SFT per arm
if has 2; then
for arm in $ARMS; do
  TAG="insurance_s2_${arm}_lr${LRTAG}_nb${NB}_heldout"
  [ -f "$MDIR/${TAG}.done" ] && { log "$TAG: done, skip"; continue; }
  DS=()
  case "$arm" in
    expert_thoughts) DS=(--dataset_path data/snorkel_insurance_split_expert_thoughts) ;;
    thoughts_policy) DS=(--dataset_path data/sft_corpus/snorkel_insurance/policy) ;;
    thoughts_base)   DS=(--dataset_path data/sft_corpus/snorkel_insurance/base) ;;
  esac
  if [ ${#DS[@]} -gt 0 ] && [ ! -s "${DS[1]}/train.json" ]; then
    log "$TAG: missing corpus ${DS[1]} (Stage 1 not done?) -- skip"; _fail=$((_fail+1)); continue
  fi
  log "STAGE-2 SFT $TAG (hide-obs, lr=$LR, $NB batches)"
  wait_gpu_free
  ./scripts/train_sft.sh "$ENVCFG" "$arm" \
      --run_tag "$TAG" --best_metric eval_action_ppl \
      --learning_rate "$LR" --num_batches "$NB" --early_stop_patience 3 "${DS[@]}" \
      > "$MDIR/${TAG}.log" 2>&1 \
    && { touch "$MDIR/${TAG}.done"; log "$TAG: done"; } \
    || { _fail=$((_fail+1)); log "$TAG: FAILED (see $MDIR/${TAG}.log)"; }
done
fi

# ---------------------------------------------------------------- Stage 3: rollout eval
if has 3; then
IDS=$(uv run --no-project python -c "import json;print(' '.join(str(i) for i in json.load(open('data/splits/snorkel_insurance_uid_to_task.json'))['rollout_task_ids']))")
TID=$(uv run --no-project python -c "import json;print(json.load(open('data/splits/snorkel_insurance_uid_to_task.json'))['train_task_ids'][0])")
log "rollout task set: $(echo $IDS | wc -w) held-out questions"
for arm in $ARMS; do
  SFT="insurance_s2_${arm}_lr${LRTAG}_nb${NB}_heldout"
  RTAG="insurance_rollout_${arm}_lr${LRTAG}"
  [ -f "$MDIR/${RTAG}.done" ] && { log "$RTAG: done, skip"; continue; }
  CK=$(newest "checkpoints_lora/act_prm_snorkel_insurance/$MODEL/${SFT}-*/step_best")
  [ -z "$CK" ] && { log "$RTAG: no checkpoint, skip"; continue; }
  log "STAGE-3 ROLLOUT $RTAG ($(echo $IDS|wc -w) questions) <- $CK"
  wait_gpu_free
  CLAUDECODE= UV_PROJECT_ENVIRONMENT=.venv-tau2 uv run --no-sync python main_pytorch.py \
      --env_config act_prm/snorkel_insurance_gym --model_config "$MODEL" \
      --lora_config r8_a16_linear --generator_config hf_grpo --trainer_config pg \
      --replay_buffer_config default --resume_from "$CK" \
      --no_train --num_batches 1 --eval_every 1 --group_size 2 --batch_size 1 \
      --max_tokens 2048 --hide_observations --run_tag "$RTAG" \
      --train_task_ids "$TID" --eval_task_ids $IDS --verbose \
      > "$MDIR/${RTAG}.log" 2>&1 \
    || { _fail=$((_fail+1)); log "$RTAG: FAILED (see $MDIR/${RTAG}.log)"; continue; }
  D=$(newest "logs/act_prm_snorkel_insurance_gym/$MODEL/${RTAG}-*/")
  uv run --no-project python scripts/check_rollout_valid.py "$D" \
    && { touch "$MDIR/${RTAG}.done"; log "$RTAG: done"; } \
    || { _fail=$((_fail+1)); log "$RTAG: INVALID (user-sim/judge outage?); not marking done"; }
done
fi

if [ "${_fail:-0}" -eq 0 ]; then
  touch "$MDIR/ALLDONE"; log "=== insurance pipeline complete ==="
else
  log "=== insurance pipeline INCOMPLETE: $_fail failure(s); not marking ALLDONE ==="
fi
