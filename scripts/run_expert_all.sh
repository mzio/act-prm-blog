#!/usr/bin/env bash
# expert_thoughts_all: SFT on the expert reasoning+action targets, but ONLY on the turns
# that actually carry reasoning.
#
# Why: ~50% of GPT-5-mini's logged retail actions are a bare <tool_call> with no reasoning
# (46% airline, 71% finance). The plain expert_thoughts arm was therefore taught "usually
# don't think", and at rollout it reasons before 0-8% of its tool calls while burning up to
# 2x the baseline's tool calls for worse completion. This arm tests whether expert thoughts
# help once the arm is actually trained to produce them.
#
# --require_thought filters TARGETS only; every turn stays in the context, so trajectories
# stay coherent (state is still messages[:idx]).
#
# CAVEAT: this halves the supervised steps (retail 635 -> 317, airline 221 -> 119, finance
# 1370 -> 402) relative to every arm it is compared against. NUM_BATCHES=300 would match
# total target steps instead of batch count, at ~2x the epochs.
#
# hide-obs throughout, matching every other Stage-2 arm and every rollout eval.
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/usr/local/bin:$PATH"
MODEL="${MODEL_CFG:-hf_qwen3_4b_instruct}"; export MODEL_CFG="$MODEL"
MDIR=/tmp/aprm/expert_all; mkdir -p "$MDIR"
NUM_BATCHES="${NUM_BATCHES:-150}"
ENVS="${ENVS:-act_prm/tau2_retail act_prm/tau2_airline act_prm/snorkel_finance_split}"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$MDIR/expert_all.log"; }
wait_gpu_free(){ while pgrep -f '[m]ain_pytorch.py' >/dev/null 2>&1; do sleep 60; done; sleep 10; }
newest(){ ls -dt $1 2>/dev/null | head -1; }

# --- 1. SFT per domain ---
for env in $ENVS; do
  ENVNAME="${env##*/}"; DOM="${ENVNAME#tau2_}"
  TAG="${DOM}_s2_expert_thoughts_all_lr3e_3_nb${NUM_BATCHES}_heldout"
  if [ -f "$MDIR/${TAG}.done" ]; then log "$TAG: done, skip"; continue; fi
  log "SFT $TAG (hide-obs, lr=3e-3, ${NUM_BATCHES} batches)"
  wait_gpu_free
  ./scripts/train_sft.sh "$env" expert_thoughts_all \
      --run_tag "$TAG" --best_metric eval_action_ppl \
      --learning_rate 3e-3 --num_batches "$NUM_BATCHES" --early_stop_patience 3 \
      > "$MDIR/${TAG}.log" 2>&1 \
    && { touch "$MDIR/${TAG}.done"; log "$TAG: done"; } || log "$TAG: FAILED (see $MDIR/${TAG}.log)"
done

# --- 2. rollout eval (retail + airline only; finance has no gym data on this box) ---
for spec in "retail:act_prm_tau2_retail" "airline:act_prm_tau2_airline"; do
  DOM="${spec%%:*}"; ENVDIR="${spec##*:}"
  TAG="${DOM}_s2_expert_thoughts_all_lr3e_3_nb${NUM_BATCHES}_heldout"
  RTAG="${DOM}_rollout_expert_thoughts_all_lr3e_3"
  [ -f "$MDIR/${RTAG}.done" ] && { log "$RTAG: done, skip"; continue; }
  CK=$(newest "checkpoints_lora/$ENVDIR/$MODEL/${TAG}-*/step_best")
  [ -z "$CK" ] && { log "$RTAG: no checkpoint, skip"; continue; }
  if [ "$DOM" = retail ]; then
    IDS=$(python3 -c "import json;print(' '.join(json.load(open('data/splits/tau2_retail_uid_to_tau2id.json'))['never_in_logs']['ids']))")
    TID=$(python3 -c "import json;print(json.load(open('data/splits/tau2_retail_uid_to_tau2id.json'))['coverage']['covered_tau2_ids'][0])")
  else
    IDS=$(python3 -c "import json;print(' '.join(str(i) for i in json.load(open('data/splits/tau2_airline_taskmap.json'))['unseen_tau2_ids']))")
    TID=$(python3 -c "import json;print(json.load(open('data/splits/tau2_airline_taskmap.json'))['covered_tau2_ids'][0])")
  fi
  log "ROLLOUT $RTAG ($(echo $IDS | wc -w) tasks)"
  wait_gpu_free
  ./scripts/train_rl_from_sft.sh "$DOM" "$CK" --run_tag "$RTAG" \
      --generator_config hf_rlvr --env_config "tau2bench/${DOM}_rlvr" \
      --no_train --num_batches 1 --eval_every 1 --group_size 2 --batch_size 1 \
      --max_turns 20 --max_tokens 2048 --discount_factor 1.0 --hide_observations \
      --train_task_ids "$TID" --eval_task_ids $IDS \
      > "$MDIR/${RTAG}.log" 2>&1 \
    && { touch "$MDIR/${RTAG}.done"; log "$RTAG: done"; } || log "$RTAG: FAILED"
done
touch "$MDIR/ALLDONE"; log "=== expert_thoughts_all complete ==="
