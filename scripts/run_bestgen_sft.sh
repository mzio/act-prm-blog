#!/usr/bin/env bash
# Act-PRM Stage-2 with ONE generation per task -- the best one -- for every domain.
#
# The Stage-1 relabel emitted several expert generations per task and the export kept them
# all, so the thought arms trained on more supervised steps than the arms they were compared
# against (retail 1043 vs 635 = 64% more; finance 1409 vs 1347). That is an alternative
# explanation for their advantage which has nothing to do with thought quality.
#
# make_bestgen_corpus.py keeps the argmax-per-task by mean EM likelihood of the committed
# thoughts (joined to generations.jsonl on thought text; 100% of uids scored, no fallbacks).
# Volumes now match the actions_only baseline: retail 627 vs 635, airline 221 vs 221,
# finance 1347 vs 1347.
#
# hide-obs, lr 3e-3, 150 batches -- identical to every other Stage-2 arm. Tracks the same
# train/eval action-span PPL + accuracy curves, then rollout-evals the best checkpoint per
# environment.
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/usr/local/bin:$PATH"
MODEL="${MODEL_CFG:-hf_qwen3_4b_instruct}"; export MODEL_CFG="$MODEL"
MDIR=/tmp/aprm/bestgen; mkdir -p "$MDIR"
NB="${NUM_BATCHES:-150}"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$MDIR/bestgen.log"; }
wait_gpu_free(){ while pgrep -f '[m]ain_pytorch.py' >/dev/null 2>&1; do sleep 60; done; sleep 10; }
newest(){ ls -dt $1 2>/dev/null | head -1; }

# env : dom : arm : corpus
SPECS=(
  "act_prm/tau2_retail:retail:thoughts_policy:data/sft_corpus/tau2_retail/policy_bestgen"
  "act_prm/tau2_retail:retail:thoughts_base:data/sft_corpus/tau2_retail/base_bestgen"
  "act_prm/tau2_airline:airline:thoughts_policy:data/sft_corpus/tau2_airline/policy_bestgen"
  "act_prm/tau2_airline:airline:thoughts_base:data/sft_corpus/tau2_airline/base_bestgen"
  "act_prm/snorkel_finance_split:snorkel_finance_split:thoughts_policy:data/sft_corpus/snorkel_finance_split/policy_v3_bestgen"
  "act_prm/snorkel_finance_split:snorkel_finance_split:thoughts_base:data/sft_corpus/snorkel_finance_split/base_v3_bestgen"
)

# --- 1. SFT ---
for spec in "${SPECS[@]}"; do
  IFS=":" read -r env dom arm pool <<< "$spec"
  TAG="${dom}_s2_${arm}_bestgen_lr3e_3_nb${NB}_heldout"
  [ -f "$MDIR/${TAG}.done" ] && { log "$TAG: done, skip"; continue; }
  [ -s "$pool/train.json" ] || { log "$TAG: corpus $pool missing, skip"; continue; }
  log "SFT $TAG  <- $pool"
  wait_gpu_free
  ./scripts/train_sft.sh "$env" "$arm" --dataset_path "$pool" \
      --run_tag "$TAG" --best_metric eval_action_ppl \
      --learning_rate 3e-3 --num_batches "$NB" --early_stop_patience 3 \
      > "$MDIR/${TAG}.log" 2>&1 \
    && { touch "$MDIR/${TAG}.done"; log "$TAG: done"; } || log "$TAG: FAILED (see $MDIR/${TAG}.log)"
done

# --- 2. rollout eval of the best checkpoint, per environment ---
RETAIL_IDS=$(python3 -c "import json;print(' '.join(json.load(open('data/splits/tau2_retail_uid_to_tau2id.json'))['never_in_logs']['ids']))")
RETAIL_TID=$(python3 -c "import json;print(json.load(open('data/splits/tau2_retail_uid_to_tau2id.json'))['coverage']['covered_tau2_ids'][0])")
AIR_IDS=$(python3 -c "import json;print(' '.join(str(i) for i in json.load(open('data/splits/tau2_airline_taskmap.json'))['unseen_tau2_ids']))")
AIR_TID=$(python3 -c "import json;print(json.load(open('data/splits/tau2_airline_taskmap.json'))['covered_tau2_ids'][0])")
FIN_FAIR=$(python3 -c "import json;print(' '.join(json.load(open('data/splits/snorkel_finance_v3.json'))['eval_questions']))")
FIN_HARD=$(python3 -c "import json;print(' '.join(json.load(open('data/splits/snorkel_finance_v3.json'))['hard_rollout_questions']))")

for spec in "${SPECS[@]}"; do
  IFS=":" read -r env dom arm pool <<< "$spec"
  TAG="${dom}_s2_${arm}_bestgen_lr3e_3_nb${NB}_heldout"
  CK=$(newest "checkpoints_lora/${env//\//_}/$MODEL/${TAG}-*/step_best")
  [ -z "$CK" ] && { log "rollout $dom/$arm: no checkpoint, skip"; continue; }
  if [ "$dom" = "snorkel_finance_split" ]; then
    for setname in fair hard; do
      IDS=$FIN_FAIR; [ "$setname" = hard ] && IDS=$FIN_HARD
      RT="finance_rollout_${arm}_bestgen_${setname}"
      [ -f "$MDIR/${RT}.done" ] && continue
      log "ROLLOUT $RT ($(echo $IDS|wc -w) questions)"; wait_gpu_free
      CLAUDECODE= UV_PROJECT_ENVIRONMENT=.venv-tau2 uv run --no-sync python main_pytorch.py \
          --env_config act_prm/snorkel_finance_gym --model_config "$MODEL" \
          --lora_config r8_a16_linear --generator_config hf_grpo --trainer_config pg \
          --replay_buffer_config default --resume_from "$CK" \
          --no_train --num_batches 1 --eval_every 1 --group_size 2 --batch_size 1 \
          --max_tokens 2048 --hide_observations --run_tag "$RT" --eval_query_ids $IDS --verbose \
          > "$MDIR/${RT}.log" 2>&1 || { log "$RT: FAILED"; continue; }
      D=$(newest "logs/act_prm_snorkel_finance_gym/$MODEL/${RT}-*/")
      uv run --no-project python scripts/check_rollout_valid.py "$D" \
        && { touch "$MDIR/${RT}.done"; log "$RT: done"; } || log "$RT: INVALID; not marking done"
    done
  else
    IDS=$RETAIL_IDS; TID=$RETAIL_TID
    [ "$dom" = airline ] && { IDS=$AIR_IDS; TID=$AIR_TID; }
    RT="${dom}_rollout_${arm}_bestgen"
    [ -f "$MDIR/${RT}.done" ] && continue
    log "ROLLOUT $RT ($(echo $IDS|wc -w) tasks)"; wait_gpu_free
    ./scripts/train_rl_from_sft.sh "$dom" "$CK" --run_tag "$RT" \
        --generator_config hf_rlvr --env_config "tau2bench/${dom}_rlvr" \
        --no_train --num_batches 1 --eval_every 1 --group_size 2 --batch_size 1 \
        --max_turns 20 --max_tokens 2048 --discount_factor 1.0 --hide_observations \
        --train_task_ids "$TID" --eval_task_ids $IDS > "$MDIR/${RT}.log" 2>&1 \
      || { log "$RT: FAILED"; continue; }
    D=$(newest "logs/tau2bench_${dom}_rlvr/$MODEL/${RT}-*/")
    uv run --no-project python scripts/check_rollout_valid.py "$D" \
      && { touch "$MDIR/${RT}.done"; log "$RT: done"; } || log "$RT: INVALID; not marking done"
  fi
done
touch "$MDIR/ALLDONE"; log "=== bestgen SFT + rollout complete ==="
