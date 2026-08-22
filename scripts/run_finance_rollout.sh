#!/usr/bin/env bash
# Rollout eval of the finance v3 Stage-2 checkpoints, on TWO deliberately different sets.
#
#   fair  : the 10 v3 eval questions. The expert solved these (they have reward>0
#           trajectories), they are the same questions the SFT curves are scored on, and
#           they never appear in v3 train. Directly comparable to retail/airline rollouts.
#   hard  : the 29 questions with NO successful expert trajectory. All 123 of their
#           rollouts are reward=0 -- these are the questions GPT-5-mini FAILED. Expect a
#           near-floor score; the value is in whether an arm degrades or holds up, NOT in
#           comparing the number against the fair set or against other domains.
#
# Reported separately and never pooled: the hard set is selected for difficulty, so mixing
# them would produce a number that means nothing.
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/usr/local/bin:$PATH"
MODEL="${MODEL_CFG:-hf_qwen3_4b_instruct}"; export MODEL_CFG="$MODEL"
MDIR=/tmp/aprm/finance_rollout; mkdir -p "$MDIR"
NB="${NUM_BATCHES:-150}"
ARMS="${ARMS:-actions_only expert_thoughts expert_thoughts_all thoughts_policy thoughts_base}"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$MDIR/rollout.log"; }
wait_gpu_free(){ while pgrep -f '[m]ain_pytorch.py' >/dev/null 2>&1; do sleep 60; done; sleep 10; }
newest(){ ls -dt $1 2>/dev/null | head -1; }

FAIR=$(python3 -c "import json;print(' '.join(json.load(open('data/splits/snorkel_finance_v3.json'))['eval_questions']))")
HARD=$(python3 -c "import json;print(' '.join(json.load(open('data/splits/snorkel_finance_v3.json'))['hard_rollout_questions']))")
log "=== finance rollout: fair=$(echo $FAIR|wc -w) questions, hard=$(echo $HARD|wc -w) questions ==="

for arm in $ARMS; do
  CK=$(newest "checkpoints_lora/act_prm_snorkel_finance_split/$MODEL/snorkel_finance_split_s2_${arm}_v3_lr3e_3_nb${NB}_heldout-*/step_best")
  [ -z "$CK" ] && { log "finance/$arm: no v3 checkpoint, skip"; continue; }
  for setname in fair hard; do
    IDS=$FAIR; [ "$setname" = hard ] && IDS=$HARD
    TAG="finance_rollout_${arm}_v3_${setname}"
    [ -f "$MDIR/${TAG}.done" ] && { log "$TAG: done, skip"; continue; }
    log "ROLLOUT $TAG ($(echo $IDS|wc -w) questions)"
    wait_gpu_free
    CLAUDECODE= UV_PROJECT_ENVIRONMENT=.venv-tau2 uv run --no-sync python main_pytorch.py \
        --env_config act_prm/snorkel_finance_gym --model_config "$MODEL" \
        --lora_config r8_a16_linear --generator_config hf_grpo --trainer_config pg \
        --replay_buffer_config default --resume_from "$CK" \
        --no_train --num_batches 1 --eval_every 1 --group_size 2 --batch_size 1 \
        --max_tokens 2048 --hide_observations --run_tag "$TAG" \
        --eval_query_ids $IDS --verbose \
        > "$MDIR/${TAG}.log" 2>&1 \
      || { log "$TAG: FAILED (see $MDIR/${TAG}.log)"; continue; }
    D=$(newest "logs/act_prm_snorkel_finance_gym/$MODEL/${TAG}-*/")
    if uv run --no-project python scripts/check_rollout_valid.py "$D"; then
      touch "$MDIR/${TAG}.done"; log "$TAG: done"
    else
      log "$TAG: INVALID (user-sim outage?); not marking done"
    fi
  done
done
touch "$MDIR/ALLDONE"; log "=== finance rollout complete ==="
