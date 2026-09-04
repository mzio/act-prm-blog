#!/usr/bin/env bash
# How many thoughts per action should Stage-2 distil? top-1 vs top-2 vs top-4, retail.
#
# DESIGN. One relabel at --group_size 8 (vs the usual 4) gives a richer candidate pool, so
# top-1/2/4 are genuinely SELECTIVE (best 1 of 8, best 2 of 8, best 4 of 8) rather than
# "keep everything". `scripts/export_sft_corpus.py --top_k K` then emits K copies of each
# trajectory, copy k using each step's k-th best thought ranked by the E-step reward.
# Task coverage is IDENTICAL across K (verified: 49 unique tasks at K=1,2,4) -- only the
# number of distilled thoughts per action changes, which is the variable under test.
#
# Everything else matches the policy_adamw30 lineage exactly: same Stage-1 checkpoint,
# r32_a32_linear, advantage_mode best, length_penalty 0, policy-scored.
#
# TWO CAVEATS, recorded so the result is not over-read:
#  1. num_batches x steps_per_batch is FIXED, so a top-4 corpus does 1/4 the epochs over 4x
#     the data. K trades epochs for thought diversity; that is the honest reading of
#     "distil more thoughts", not a confound to be removed.
#  2. The E-step commits GREEDILY: thought t is generated conditioned on the length-
#     penalised argmax of steps 1..t-1 (generator/act_prm/base.py:616, prompts.py:57).
#     Scoring, however, uses the raw logged prefix. So a rank-2 thought was never generated
#     in a world where rank-2 was committed -- top-K gives K alternative thoughts PER
#     ACTION, not K coherent alternative thought-chains.
set -uo pipefail
cd /home/mzio/projects/act-prm-blog
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/usr/local/bin:$PATH"
[ -f scripts/wandb_preflight.sh ] && . scripts/wandb_preflight.sh
export PYTHONUNBUFFERED=1
export HF_HOME=/data/users/mzio/models/hf_cache
export HF_HUB_OFFLINE=1
export WANDB_ERROR_REPORTING=false
export ACT_PRM_DUMP_TRAJECTORIES=1
G=/tmp/aprm/topk; mkdir -p "$G"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$G/chain.log"; }
reap(){ for p in $(pgrep -f 'main_pytorch\.py'); do kill -9 "$p" 2>/dev/null; done; sleep 15; }
newest(){ ls -dt $1 2>/dev/null | head -1; }

CKROOT=checkpoints_lora/act_prm_tau2_retail/hf_qwen3_4b_instruct
LOGROOT=logs/act_prm_tau2_retail/hf_qwen3_4b_instruct
S1=$(newest "$CKROOT/retail_s1em_policy_adamw30-*/step_best")
[ -z "$S1" ] && { log "FATAL: no retail_s1em_policy_adamw30 step_best"; exit 1; }


# ---- 0. SEED-VARIED insurance repeats (prepended 2026-09-03)
# The phase-2 "repeats" measured NOTHING for insurance: per-task agreement between run 1
# and rep2 was 40/40 = 100% with identical seed=42. The snorkel gyms are deterministic
# given a seed -- unlike tau2, which calls an EXTERNAL Claude user simulator we do not seed
# (that is where retail's 14pt swing between two numerically identical checkpoints came
# from). To measure insurance's real rollout variance the SEED has to change.
# Targeted at the two configs carrying the headline 52.5% -> 15.0% result.
# Calls main_pytorch DIRECTLY: run_insurance_rollout.sh has no EXTRA_ARGS passthrough, so
# a --seed handed to it would be SILENTLY DROPPED and we would just produce more duplicates.
INS_IDS=$(python3 -c "import json;print(' '.join(str(i) for i in json.load(open('data/splits/snorkel_insurance_uid_to_task.json'))['eval_task_ids']))")
for SEED in 1234 777; do
  for step in step_0020 step_best; do
    m="$G/seedvar.$SEED.$step.done"; [ -f "$m" ] && continue
    CK=$(newest "checkpoints_lora/act_prm_snorkel_insurance/hf_qwen3_4b_instruct/snorkel_insurance_s2_thoughts_policy_adamw30_lr1e_3_nb200_flat32sgd_heldout-*/$step")
    [ -z "$CK" ] && { log "SEED-VAR $step: no checkpoint, skip"; touch "$m"; continue; }
    TAG="insurance_rollout_thoughts_policy_adamw30_seed${SEED}_${step}"
    log "SEED-VAR insurance thoughts_policy/$step seed=$SEED"
    CLAUDECODE= UV_PROJECT_ENVIRONMENT=.venv-tau2 uv run --no-sync python main_pytorch.py \
        --env_config act_prm/snorkel_insurance_gym --model_config hf_qwen3_4b_instruct \
        --lora_config r8_a16_linear --generator_config hf_grpo --trainer_config pg \
        --replay_buffer_config default --resume_from "$CK" \
        --no_train --num_batches 1 --eval_every 1 --group_size 2 --batch_size 1 \
        --max_tokens 2048 --hide_observations --run_tag "$TAG" \
        --eval_task_ids $INS_IDS --seed "$SEED" --verbose >>"$G/seedvar.log" 2>&1
    log "  rc=$?"; touch "$m"; reap
  done
done

# ---- 1. relabel at group_size 8 (52 retail train trajectories / batch 4 = 13 batches)
if [ ! -f "$G/relabel.done" ]; then
  log "RELABEL retail @ group_size 8  <- $S1"
  ./scripts/train.sh --env_config act_prm/tau2_retail --generator_config act_prm \
    --trainer_config pg --model_config hf_qwen3_4b_instruct --lora_config r32_a32_linear \
    --replay_buffer_config default --no-score_with_base \
    --no_train --resume_from "$S1" --advantage_mode best --group_size 8 --batch_size 4 \
    --num_batches 13 --no_initial_eval --length_penalty 0 --save_generations \
    --run_tag retail_s1relabel_policy_g8 --verbose >>"$G/relabel.log" 2>&1
  log "  rc=$?"
  gen=$(newest "$LOGROOT/retail_s1relabel_policy_g8-*/generations.jsonl")
  [ -z "$gen" ] && { log "FATAL: no generations.jsonl"; exit 1; }
  # Confirm we really got 8 candidates per step before exporting top-4.
  n=$(python3 -c "import json;print(len(json.loads(open('$gen').readline()).get('thoughts',[])))")
  log "  candidates/step = $n (want 8)"
  touch "$G/relabel.done"; reap
fi
GEN=$(newest "$LOGROOT/retail_s1relabel_policy_g8-*/generations.jsonl")

# ---- 2. export top-1 / top-2 / top-4
for K in 1 2 4; do
  OUT=data/sft_corpus/tau2_retail/policy_g8_top$K
  [ -s "$OUT/train.json" ] && { log "export top$K exists, skip"; continue; }
  rm -rf "$OUT"
  uv run --no-project python scripts/export_sft_corpus.py --generations "$GEN" \
    --source-pools data/tau2_retail --out "$OUT" --top_k $K >>"$G/export.log" 2>&1
  log "export top$K -> $(python3 -c "import json;print(len(json.load(open('$OUT/train.json'))))" 2>/dev/null || echo FAILED) train trajectories"
done

# ---- 3. SFT each corpus (identical Stage-2 recipe to every other SGD 1e-3 arm)
for K in 1 2 4; do
  TAG=retail_s2_thoughts_policy_g8top${K}_lr1e_3_nb200_flat32sgd_heldout
  if [ ! -f "$G/sft.$K.done" ]; then
    log "SFT top$K"
    TRAINER_CFG=sft_flat ./scripts/train_sft.sh act_prm/tau2_retail thoughts_policy \
      --run_tag "$TAG" --best_metric eval_actiononly_ppl \
      --learning_rate 1e-3 --optimizer sgd --num_batches 200 --eval_every 5 \
      --steps_per_batch 32 --early_stop_patience 6 --hide_observations \
      --dataset_path "data/sft_corpus/tau2_retail/policy_g8_top$K" >>"$G/sft.$K.log" 2>&1
    log "  rc=$?  snapshots=$(ls -d $CKROOT/${TAG}-*/step_* 2>/dev/null | wc -l)"
    touch "$G/sft.$K.done"; reap
  fi
done

# ---- 4. rollouts
for K in 1 2 4; do
  for step in step_0020 step_best; do
    m="$G/roll.$K.$step.done"; [ -f "$m" ] && continue
    log "ROLLOUT top$K/$step"
    VARIANTS="thoughts_policy_g8top${K}" CKPT_PAT="lr1e_3_nb200_flat32sgd" \
    CKPT_STEP="$step" CKPT_TAG="g8top${K}" ./scripts/run_sft_rollout_eval.sh >>"$G/chain.log" 2>&1
    log "  rc=$?"; touch "$m"; reap
  done
done
log "=== top-K chain complete ==="
