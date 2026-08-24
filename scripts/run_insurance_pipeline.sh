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
#   1  EM thought generation over the 180 train questions, twice (policy- and base-scored):
#      1a EM training -> step_best, then 1b a relabel pass from that checkpoint that commits
#      the best-scored thought per logged step, exported to an SFT corpus.
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

# ---------------------------------------------------------------- Stage 1: EM + relabel
# TWO passes per scorer, matching what retail/airline/finance did:
#   1a EM TRAINING   (nb=25 bs=4 gs=4, save_generations) -> step_best checkpoint
#   1b RELABEL       (--no_train --resume_from step_best --advantage_mode best) -> commits
#      thoughts[best] per logged step -> generations.jsonl -> export_sft_corpus.py
# A single no-train pass from the BASE model is NOT the same experiment: the thoughts would
# come from an untrained policy, so thoughts_policy would no longer be policy-scored.
#
# Relabel coverage: nb*bs must reach every train trajectory. 180 train / bs 4 = 45 batches
# (the other domains used nb=20 because retail had 49 train trajectories and finance 116).
CKROOT="checkpoints_lora/act_prm_snorkel_insurance/$MODEL"
LOGROOT="logs/act_prm_snorkel_insurance/$MODEL"
EM_NB="${EM_NB:-25}"
RELABEL_NB="${RELABEL_NB:-45}"
n_rows(){ uv run --no-project python -c "import json;print(len(json.load(open('$1'))))" 2>/dev/null || echo 0; }

if has 1; then
for scorer in policy base; do
  CORPUS="data/sft_corpus/snorkel_insurance/${scorer}"
  if [ -s "$CORPUS/train.json" ] && [ "$(n_rows "$CORPUS/train.json")" -gt 0 ]; then
    log "stage1/$scorer: corpus exists ($(n_rows "$CORPUS/train.json") train / $(n_rows "$CORPUS/eval.json") eval), skip"
    continue
  fi
  SWB=--no-score_with_base; SWBV=0
  [ "$scorer" = base ] && { SWB=--score_with_base; SWBV=1; }

  # -- 1a. EM training
  EMTAG="insurance_s1em_${scorer}"
  if [ ! -f "$MDIR/${EMTAG}.done" ]; then
    # TIMING CHECK. This is the largest EM pass attempted: 2,273 train steps x group_size 4,
    # against retail's 635. Cost was never calibrated for this scale, so measure it and
    # report a real ETA instead of guessing (and so a pathological run is visible early).
    log "STAGE-1a EM training $EMTAG (${scorer}-scored, nb=$EM_NB bs=4 gs=4)"
    log "  [timing] EM pass starting; will report measured s/batch on completion"
    _t0=$(date +%s)
    wait_gpu_free
    ./scripts/train.sh --env_config "$ENVCFG" --generator_config act_prm --trainer_config pg \
        --model_config "$MODEL" --lora_config r8_a16_linear --replay_buffer_config default \
        $SWB --run_tag "$EMTAG" --group_size 4 --batch_size 4 --num_batches "$EM_NB" \
        --length_penalty 0.15 --save_generations --verbose \
        > "$MDIR/${EMTAG}.log" 2>&1 \
      || { _fail=$((_fail+1)); log "$EMTAG: FAILED (see $MDIR/${EMTAG}.log)"; continue; }
    _el=$(( $(date +%s) - _t0 ))
    log "  [timing] $EMTAG took ${_el}s for $EM_NB batches = $((_el / (EM_NB>0?EM_NB:1)))s/batch"
    log "  [timing] -> relabel ($RELABEL_NB batches) projects to ~$(( _el / (EM_NB>0?EM_NB:1) * RELABEL_NB / 60 ))min"
    touch "$MDIR/${EMTAG}.done"; log "$EMTAG: done"
  fi

  # -- 1b. relabel from the EM checkpoint, then export the corpus
  CK=$(ls -dt "$CKROOT"/*swb=${SWBV}-*/step_best 2>/dev/null | head -1)
  [ -z "$CK" ] && { _fail=$((_fail+1)); log "stage1/$scorer: no EM step_best under $CKROOT, skip"; continue; }
  RTAG="insurance_s1relabel_${scorer}"
  log "STAGE-1b relabel $RTAG from $CK (nb=$RELABEL_NB bs=4 -> $((RELABEL_NB*4)) trajectories)"
  wait_gpu_free
  ./scripts/train.sh --env_config "$ENVCFG" --generator_config act_prm --trainer_config pg \
      --model_config "$MODEL" --lora_config r8_a16_linear --replay_buffer_config default \
      $SWB --no_train --resume_from "$CK" --advantage_mode best \
      --group_size 4 --batch_size 4 --num_batches "$RELABEL_NB" --no_initial_eval \
      --length_penalty 0.15 --save_generations --run_tag "$RTAG" --verbose \
      > "$MDIR/${RTAG}.log" 2>&1 \
    || { _fail=$((_fail+1)); log "$RTAG: FAILED (see $MDIR/${RTAG}.log)"; continue; }
  GEN=$(newest "$LOGROOT/${RTAG}-*/generations.jsonl")
  [ -z "$GEN" ] && { _fail=$((_fail+1)); log "$RTAG: no generations.jsonl (--save_generations?)"; continue; }
  # No --advantage_mode here: that is a main_pytorch flag, not an export one. The exporter
  # already commits thoughts[best] (the candidate index the EM generator marked) per step.
  uv run --no-project python scripts/export_sft_corpus.py \
      --generations "$GEN" --source-pools data/snorkel_insurance_split \
      --out "$CORPUS" >> "$MDIR/${RTAG}.log" 2>&1 \
    || { _fail=$((_fail+1)); log "$RTAG: corpus export FAILED"; continue; }
  NT=$(n_rows "$CORPUS/train.json"); NE=$(n_rows "$CORPUS/eval.json")
  # An empty or short corpus is the failure that silently produced train:0 corpora before.
  if [ "$NT" -lt 90 ]; then
    _fail=$((_fail+1)); log "$RTAG: corpus only $NT/180 train trajectories -- raise RELABEL_NB; not marking done"
  else
    log "$RTAG: done -> $CORPUS ($NT train / $NE eval of 180/40)"
  fi
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
  # JUDGE PARSE-RATE CHECK. The finance rollouts silently graded 67% of answers by DEFAULT
  # ("no") because the judge replied in a shape the parser did not match. The fall-through
  # parser should fix that, but "should" is not a measurement -- verify per run, and refuse
  # to bank one whose scores are mostly unread defaults (they all bias toward incorrect).
  _JOK=1
  uv run --no-project python scripts/check_judge_parse_rate.py "$MDIR/${RTAG}.log" \
      2>&1 | tee -a "$MDIR/insurance.log" | grep -q "FAIL:" && _JOK=0
  if ! uv run --no-project python scripts/check_rollout_valid.py "$D"; then
    _fail=$((_fail+1)); log "$RTAG: INVALID (episodes died early -- user-sim/judge outage?); not marking done"
  elif [ "$_JOK" = 0 ]; then
    _fail=$((_fail+1)); log "$RTAG: judge parse rate too low; not marking done"
  else
    touch "$MDIR/${RTAG}.done"; log "$RTAG: done"
  fi
done
fi

if [ "${_fail:-0}" -eq 0 ]; then
  touch "$MDIR/ALLDONE"; log "=== insurance pipeline complete ==="
else
  log "=== insurance pipeline INCOMPLETE: $_fail failure(s); not marking ALLDONE ==="
fi
