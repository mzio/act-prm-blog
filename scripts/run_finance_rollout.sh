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
#
# MAX_TURNS=24, not the env default of 12. Measured on the 116 v3 expert trajectories, the
# assistant-turn count is p50=11, p90=19, max=24 -- a cap of 12 covers only 61% of the
# EXPERT's own demonstrations, so it makes the cap, not the policy, the binding constraint.
# At 12 every arm scored 0/10 with timesteps=0 and a final message of "Sorry, you have
# reached the maximum number of steps": the policy never got to call respond_user, so no
# answer was ever graded. 24 covers 100% of expert trajectories.
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/usr/local/bin:$PATH"
# HuggingFace access. This driver calls main_pytorch.py DIRECTLY rather than going through
# train_rl_from_sft.sh, so it does not inherit that script's proxy setup -- and cron gives
# us an environment with none of it. huggingface.co is not directly resolvable from this
# box, so load_llm died with `httpx.ConnectError: [Errno -2] Name or service not known`
# AFTER loading the cached weights, taking all 10 arms down in ~18s each on 08-23 07:55.
export https_proxy="${https_proxy:-http://fwdproxy:8080}"
export http_proxy="${http_proxy:-http://fwdproxy:8080}"
export HF_TOKEN="${HF_TOKEN:-$(cat "$HOME/models/token" 2>/dev/null || true)}"
<<<<<<< HEAD
# fwdproxy began 403-ing huggingface.co on 2026-09-03, which killed model loading in
# load_hf_model_and_tokenizer -> hf_hub list_repo_tree (all 4 expert_thoughts_all runs
# and the insurance base rollout died in ~25s). The weights are cached locally under
# HF_HOME, so go offline and never touch the Hub. Verified: AutoConfig+AutoTokenizer
# for Qwen3-4B-Instruct-2507 load fine with HF_HUB_OFFLINE=1.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_HOME="${HF_HOME:-/data/users/mzio/models/hf_cache}"
=======
>>>>>>> a588bf289485252b715d699604b5a28688f50be9
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
<<<<<<< HEAD
  # CKPT_PAT / CKPT_STEP select WHICH Stage-2 generation and snapshot to roll out. The
  # pattern was hardcoded to the SGD-era v3_lr3e_3 checkpoints; the 2026-09-01 lr 1e-4 runs
  # are lr1e_4_adamw_nb200_flat32, and step_best is NOT the checkpoint we want (on retail
  # it scored 0/42 while step_0020 scored 5/42).
  CK=$(newest "checkpoints_lora/act_prm_snorkel_finance_split/$MODEL/snorkel_finance_split_s2_${arm}_${CKPT_PAT:-v3_lr3e_3_nb${NB}}_heldout-*/${CKPT_STEP:-step_best}")
  [ -z "$CK" ] && { log "finance/$arm: no v3 checkpoint, skip"; continue; }
  # SETS restricts which question sets run. Default keeps both (historical behaviour);
  # SETS=fair is the cross-domain-comparable one and halves the cost of a sweep.
  for setname in ${SETS:-fair hard}; do
    IDS=$FAIR; [ "$setname" = hard ] && IDS=$HARD
    # TAG must encode the checkpoint generation AND snapshot, or a new rollout collides
    # with the SGD-era one: same log dir (metrics appended) and same .done marker (the arm
    # gets silently skipped). That exact collision hid the actions_only baseline on
    # 2026-08-31 until the tag was fixed.
    TAG="finance_rollout_${arm}_${CKPT_TAG:-v3}${CKPT_STEP:+_${CKPT_STEP}}_${setname}"
=======
  CK=$(newest "checkpoints_lora/act_prm_snorkel_finance_split/$MODEL/snorkel_finance_split_s2_${arm}_v3_lr3e_3_nb${NB}_heldout-*/step_best")
  [ -z "$CK" ] && { log "finance/$arm: no v3 checkpoint, skip"; continue; }
  for setname in fair hard; do
    IDS=$FAIR; [ "$setname" = hard ] && IDS=$HARD
    TAG="finance_rollout_${arm}_v3_${setname}"
>>>>>>> a588bf289485252b715d699604b5a28688f50be9
    [ -f "$MDIR/${TAG}.done" ] && { log "$TAG: done, skip"; continue; }
    log "ROLLOUT $TAG ($(echo $IDS|wc -w) questions)"
    wait_gpu_free
    CLAUDECODE= UV_PROJECT_ENVIRONMENT=.venv-tau2 uv run --no-sync python main_pytorch.py \
        --env_config act_prm/snorkel_finance_gym --model_config "$MODEL" \
        --lora_config r8_a16_linear --generator_config hf_grpo --trainer_config pg \
        --replay_buffer_config default --resume_from "$CK" \
        --no_train --num_batches 1 --eval_every 1 --group_size 2 --batch_size 1 \
        --max_tokens 2048 --max_turns "${MAX_TURNS:-24}" --hide_observations --run_tag "$TAG" \
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
# Only claim completion if every arm/set actually banked a .done. Touching ALLDONE
# unconditionally is how 10 straight failures got recorded as "finance rollout complete"
# and the guard advanced past them.
_missing=0
for arm in $ARMS; do
<<<<<<< HEAD
  # Must mirror the SETS filter above, or the gate demands .done markers for sets that
  # were deliberately not run and never reports completion.
  for setname in ${SETS:-fair hard}; do
    [ -f "$MDIR/finance_rollout_${arm}_${CKPT_TAG:-v3}${CKPT_STEP:+_${CKPT_STEP}}_${setname}.done" ] || _missing=$((_missing+1))
=======
  for setname in fair hard; do
    [ -f "$MDIR/finance_rollout_${arm}_v3_${setname}.done" ] || _missing=$((_missing+1))
>>>>>>> a588bf289485252b715d699604b5a28688f50be9
  done
done
if [ "$_missing" -eq 0 ]; then
  touch "$MDIR/ALLDONE"; log "=== finance rollout complete ==="
else
  log "=== finance rollout INCOMPLETE: $_missing run(s) missing; not marking ALLDONE ==="
fi
