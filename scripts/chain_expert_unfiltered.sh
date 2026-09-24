#!/usr/bin/env bash
# Stage-2 CONTROL ARM: expert_thoughts (unfiltered) at SGD 1e-3 / nb200, then step_0020
# rollouts. One GPU, strictly sequential.
#
# WHAT THIS ISOLATES
#   expert_thoughts_all differs from the EM arms in THREE ways at once, so its
#   "best PPL, no rollout gain" result is unattributable:
#     1. thought source  -- GPT-5-class expert vs the policy's own samples
#     2. thought length  -- 2-5x longer median, tails to 3-4k chars vs a ~550-685 ceiling
#     3. turn coverage   -- --require_thought DROPS the thoughtless turns from training
#                           (sft_flat.py:114, train split only), leaving 64.1% of turns on
#                           insurance, 49.9% retail, 53.8% airline vs 100% for every other
#                           arm -- and a BIASED subset, the turns the expert found hard
#                           (confirmed: eval_actiononly_ppl_ownthoughtsub > overall PPL).
#   Dropping --require_thought moves variable 3 alone: same pool, same thoughts, same
#   optimizer/lr/nb/trainer. The resolved command differs from the expert_thoughts_all run
#   by exactly that one flag plus the run tag (verified with SFT_DRY_RUN=1).
#
# KNOWN PRIOR: train_sft.sh:84-86 records that the unfiltered arm "taught it not to think:
# at rollout it reasons before 0-8% of its tool calls" -- which is WHY --require_thought
# was added. So a collapse here is the expected outcome, not a surprise. It is still worth
# the GPU time because that prior was measured on the OLD AdamW / SGD-3e-3 generations at
# step_best only, never at step_0020 on the SGD 1e-3 / nb200 generation the headline
# numbers come from, and because scripts/measure_thought_emission.py now shows EVERY arm
# (including ones trained at 100% coverage) emits a thought before only 4-17% of its tool
# calls -- so "trained coverage" may simply not control "rollout emission" at all.
#
# ORDER: insurance first. It is the only domain whose noise floor (seed sd ~3.0pt, n=40)
# resolves the ~5pt effects at issue; retail sd is 9.0pt, airline 22pt, finance n=10 has no
# information. If the GPU has to be reclaimed, the informative half is already banked.
#
# Budget (measured medians): SFT 2.6/1.9/1.75/1.9h + rollouts 5x51 / 3x73 / 3x34 / 1x23 min
# ~= 18h wall-clock.
set -uo pipefail
cd /home/mzio/projects/act-prm-blog
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/usr/local/bin:$PATH"
[ -f scripts/wandb_preflight.sh ] && . scripts/wandb_preflight.sh
# HF_HUB_OFFLINE + pinned HF_HOME: fwdproxy started 403-ing huggingface.co once and killed
# five runs in 25s. Everything needed is already in the cache.
export PYTHONUNBUFFERED=1 HF_HOME=/data/users/mzio/models/hf_cache HF_HUB_OFFLINE=1
export WANDB_ERROR_REPORTING=false ACT_PRM_DUMP_TRAJECTORIES=1

G=/tmp/aprm/expert_unfiltered; mkdir -p "$G"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$G/chain.log"; }
reap(){ for p in $(pgrep -f 'main_pytorch\.py'); do kill -9 "$p" 2>/dev/null; done; sleep 15; }
newest(){ ls -dt $1 2>/dev/null | head -1; }

ARM=expert_thoughts
CKPAT="lr1e_3_nb200_flat32sgd"
SFT_FLAGS=(--best_metric eval_actiononly_ppl --learning_rate 1e-3 --optimizer sgd
           --num_batches 200 --eval_every 5 --early_stop_patience 6 --steps_per_batch 32)

# Pools: reuse the EXACT dataset_path each domain's expert_thoughts_all run used. The
# derived default (data/<envname>_expert_thoughts) resolves to a different pool for retail
# and finance, whose eval set differs from the base -- which silently makes the arm
# non-comparable (train_sft.sh:88-91).
sft(){  # sft <env_config> <envdir> <run_tag_prefix> <pool>
  local env="$1" envdir="$2" prefix="$3" pool="$4"
  local tag="${prefix}_s2_${ARM}_${CKPAT}_heldout"
  local m="$G/sft.$prefix.done"; [ -f "$m" ] && { log "SFT $prefix: done, skip"; return; }
  if [ -n "$(newest "checkpoints_lora/$envdir/hf_qwen3_4b_instruct/${tag}-*/step_0020")" ]; then
    log "SFT $prefix: step_0020 already exists, skip"; touch "$m"; return
  fi
  log "SFT $prefix ($ARM, unfiltered, pool=$pool)"
  TRAINER_CFG=sft_flat EXPERT_POOL="$pool" ./scripts/train_sft.sh "$env" "$ARM" \
      --run_tag "$tag" "${SFT_FLAGS[@]}" >>"$G/sft.$prefix.log" 2>&1
  log "  rc=$?"; touch "$m"; reap
}

# ---------------------------------------------------------------- 1. insurance (n=40, 5 seeds)
sft act_prm/snorkel_insurance act_prm_snorkel_insurance snorkel_insurance \
    data/snorkel_insurance_expert_thoughts

INS_IDS=$(python3 -c "import json;print(' '.join(str(i) for i in json.load(open('data/splits/snorkel_insurance_uid_to_task.json'))['eval_task_ids']))")
# The snorkel gyms have no simulated user, so a rollout is DETERMINISTIC given a seed --
# repeats at the default 42 are byte-identical (measured 40/40). Samples must vary --seed,
# and run_insurance_rollout.sh has no EXTRA_ARGS passthrough, so call main_pytorch directly.
for SEED in 1234 777 555 999 42; do
  m="$G/ins.$SEED.done"; [ -f "$m" ] && continue
  CK=$(newest "checkpoints_lora/act_prm_snorkel_insurance/hf_qwen3_4b_instruct/snorkel_insurance_s2_${ARM}_${CKPAT}_heldout-*/step_0020")
  [ -z "$CK" ] && { log "ins: no step_0020, skip seed $SEED"; touch "$m"; continue; }
  log "INSURANCE rollout $ARM seed=$SEED step_0020"
  CLAUDECODE= UV_PROJECT_ENVIRONMENT=.venv-tau2 uv run --no-sync python main_pytorch.py \
      --env_config act_prm/snorkel_insurance_gym --model_config hf_qwen3_4b_instruct \
      --lora_config r8_a16_linear --generator_config hf_grpo --trainer_config pg \
      --replay_buffer_config default --resume_from "$CK" \
      --no_train --num_batches 1 --eval_every 1 --group_size 2 --batch_size 1 \
      --max_tokens 2048 --hide_observations \
      --run_tag "insurance_rollout_${ARM}_seed${SEED}_step_0020" \
      --eval_task_ids $INS_IDS --seed "$SEED" --verbose >>"$G/ins.$SEED.log" 2>&1
  log "  rc=$?"; touch "$m"; reap
done

# ---------------------------------------------------------------- 2. retail + airline
sft act_prm/tau2_retail  act_prm_tau2_retail  retail  data/tau2_retail_expert_thoughts_matched
sft act_prm/tau2_airline act_prm_tau2_airline airline data/tau2_airline_expert_thoughts

# tau2 calls an EXTERNAL Claude user simulator we do not seed, so plain repeats are already
# independent; they only need a distinct CKPT_TAG so .done markers and log dirs do not
# collide. One driver call covers retail AND airline.
for rep in rep1 rep2 rep3; do
  m="$G/tau2.$rep.done"; [ -f "$m" ] && continue
  log "TAU2 (retail+airline) rollout $ARM $rep step_0020"
  VARIANTS="$ARM" CKPT_PAT="$CKPAT" CKPT_STEP="step_0020" CKPT_TAG="sgdlr1e_3_$rep" \
      ./scripts/run_sft_rollout_eval.sh >>"$G/chain.log" 2>&1
  log "  rc=$?"; touch "$m"; reap
done

# ---------------------------------------------------------------- 3. finance (n=10, no power)
sft act_prm/snorkel_finance_split act_prm_snorkel_finance_split snorkel_finance_split \
    data/snorkel_finance_split_expert_thoughts_all_v3

if [ ! -f "$G/fin.done" ]; then
  log "FINANCE rollout $ARM step_0020 (fair set only)"
  ARMS="$ARM" SETS=fair CKPT_PAT="$CKPAT" CKPT_STEP="step_0020" CKPT_TAG="sgdlr1e_3" \
      ./scripts/run_finance_rollout.sh >>"$G/chain.log" 2>&1
  log "  rc=$?"; touch "$G/fin.done"; reap
fi

# ---------------------------------------------------------------- 4. re-export
log "re-exporting CSVs"
uv run --no-sync python scripts/export_training_results.py   >>"$G/chain.log" 2>&1
uv run --no-sync python scripts/export_rollout_results.py    >>"$G/chain.log" 2>&1
uv run --no-sync python scripts/export_trajectories.py       >>"$G/chain.log" 2>&1
uv run --no-sync python scripts/measure_thought_emission.py  >>"$G/chain.log" 2>&1
log "=== expert_thoughts (unfiltered) control complete ==="
