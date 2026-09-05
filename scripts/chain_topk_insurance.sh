#!/usr/bin/env bash
# top-K thought distillation on INSURANCE: K = 1, 2, 4, 8 from a group_size-8 relabel.
#
# WHY INSURANCE AND NOT RETAIL. The retail top-K sweep (chain_topk.sh) came back with NO
# detectable difference -- PPL spread 0.016 across K, rollouts 16.7-28.6% -- but retail
# cannot resolve anything under ~18 points (two numerically identical checkpoints scored
# 31.0% and 16.7% there). Insurance is the only domain with real power: same-seed noise
# ~2.5pt, seed-varied sd ~3.1pt, n=40, and a measured base of 30.0%. If K matters at all,
# insurance is where it will show.
#
# K = 1, 2, 4, 8 spans fully selective (best 1-of-8) to no selection at all (all 8), so the
# sweep separates "better-chosen thoughts" from "more thoughts".
#
# RECIPE: identical to the insurance policy_adamw30 lineage except --group_size 8.
#   relabel: r32_a32_linear, advantage_mode best, length_penalty 0, batch 4, nb 45
#            (180 insurance train trajectories / 4), policy-scored.
#   SFT:     sft_flat, SGD 1e-3, nb 200, spb 32, eval_every 5, patience 6, hide-obs.
#
# CAVEATS (same as retail's, restated because they bite here too):
#  1. nb x spb is FIXED, so top-8 does 1/8 the epochs over 8x the data. K trades epochs for
#     thought diversity -- that IS the question, not a confound to remove.
#  2. The E-step commits greedily (generator/act_prm/base.py:616): thought t is generated
#     conditioned on the length-penalised argmax of 1..t-1, while SCORING uses the raw
#     logged prefix. So rank-k thoughts are alternatives PER ACTION, not coherent
#     alternative thought-chains.
#  3. Rollout repeats on insurance MUST vary --seed (snorkel gyms are near-deterministic
#     given a seed; see CLAUDE.md). The rollouts here are single-seed -- treat them as one
#     sample each, and seed-sweep whichever K looks interesting.
set -uo pipefail
cd /home/mzio/projects/act-prm-blog
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/usr/local/bin:$PATH"
[ -f scripts/wandb_preflight.sh ] && . scripts/wandb_preflight.sh
export PYTHONUNBUFFERED=1
export HF_HOME=/data/users/mzio/models/hf_cache
export HF_HUB_OFFLINE=1
export WANDB_ERROR_REPORTING=false
export ACT_PRM_DUMP_TRAJECTORIES=1
G=/tmp/aprm/topk_ins; mkdir -p "$G"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$G/chain.log"; }
reap(){ for p in $(pgrep -f 'main_pytorch\.py'); do kill -9 "$p" 2>/dev/null; done; sleep 15; }
newest(){ ls -dt $1 2>/dev/null | head -1; }

CKROOT=checkpoints_lora/act_prm_snorkel_insurance/hf_qwen3_4b_instruct
LOGROOT=logs/act_prm_snorkel_insurance/hf_qwen3_4b_instruct
S1=$(newest "$CKROOT/insurance_s1em_policy_adamw30-*/step_best")
[ -z "$S1" ] && { log "FATAL: no insurance_s1em_policy_adamw30 step_best"; exit 1; }

# ---- 1. relabel @ group_size 8
# Fallback to batch_size 2 / nb 90 on failure: g8 x bs4 on insurance's long contexts may
# OOM, and the same 180 trajectories are covered either way.
if [ ! -f "$G/relabel.done" ]; then
  for attempt in "4 45" "2 90"; do
    set -- $attempt; BS=$1; NB=$2
    log "RELABEL insurance @ group_size 8, batch_size $BS, nb $NB  <- $S1"
    ./scripts/train.sh --env_config act_prm/snorkel_insurance --generator_config act_prm \
      --trainer_config pg --model_config hf_qwen3_4b_instruct --lora_config r32_a32_linear \
      --replay_buffer_config default --no-score_with_base \
      --no_train --resume_from "$S1" --advantage_mode best --group_size 8 --batch_size "$BS" \
      --num_batches "$NB" --no_initial_eval --length_penalty 0 --save_generations \
      --run_tag insurance_s1relabel_policy_g8 --verbose >>"$G/relabel.log" 2>&1
    rc=$?; log "  rc=$rc (bs=$BS)"
    gen=$(newest "$LOGROOT/insurance_s1relabel_policy_g8-*/generations.jsonl")
    if [ -n "$gen" ] && [ -s "$gen" ]; then
      n=$(python3 -c "import json;print(len(json.loads(open('$gen').readline()).get('thoughts',[])))")
      log "  candidates/step = $n (want 8), rows=$(wc -l < "$gen")"
      [ "$n" = 8 ] && { touch "$G/relabel.done"; break; }
      log "  WRONG candidate count -- not marking done"
    fi
    log "  attempt with bs=$BS failed; trying next"
    reap
  done
  [ -f "$G/relabel.done" ] || { log "FATAL: relabel failed both attempts"; exit 1; }
  reap
fi
GEN=$(newest "$LOGROOT/insurance_s1relabel_policy_g8-*/generations.jsonl")

# ---- 2. export top-1 / 2 / 4 / 8
for K in 1 2 4 8; do
  OUT=data/sft_corpus/snorkel_insurance/policy_g8_top$K
  [ -s "$OUT/train.json" ] && { log "export top$K exists, skip"; continue; }
  rm -rf "$OUT"
  uv run --no-project python scripts/export_sft_corpus.py --generations "$GEN" \
    --source-pools data/snorkel_insurance_split --out "$OUT" --top_k $K >>"$G/export.log" 2>&1
  log "export top$K -> $(python3 -c "import json;print(len(json.load(open('$OUT/train.json'))))" 2>/dev/null || echo FAILED) train trajectories"
done

# ---- 3. SFT each corpus
for K in 1 2 4 8; do
  TAG=snorkel_insurance_s2_thoughts_policy_g8top${K}_lr1e_3_nb200_flat32sgd_heldout
  if [ ! -f "$G/sft.$K.done" ]; then
    log "SFT insurance top$K"
    TRAINER_CFG=sft_flat ./scripts/train_sft.sh act_prm/snorkel_insurance thoughts_policy \
      --run_tag "$TAG" --best_metric eval_actiononly_ppl \
      --learning_rate 1e-3 --optimizer sgd --num_batches 200 --eval_every 5 \
      --steps_per_batch 32 --early_stop_patience 6 --hide_observations \
      --dataset_path "data/sft_corpus/snorkel_insurance/policy_g8_top$K" >>"$G/sft.$K.log" 2>&1
    log "  rc=$?  snapshots=$(ls -d $CKROOT/${TAG}-*/step_* 2>/dev/null | wc -l)"
    touch "$G/sft.$K.done"; reap
  fi
done

# ---- 4. rollouts (40 held-out insurance tasks each)
for K in 1 2 4 8; do
  for step in step_0020 step_best; do
    m="$G/roll.$K.$step.done"; [ -f "$m" ] && continue
    log "ROLLOUT insurance top$K/$step"
    ARMS="thoughts_policy_g8top${K}" CKPT_PAT="lr1e_3_nb200_flat32sgd" \
    CKPT_STEP="$step" CKPT_TAG="g8top${K}" ./scripts/run_insurance_rollout.sh >>"$G/chain.log" 2>&1
    log "  rc=$?"; touch "$m"; reap
  done
done
log "=== insurance top-K chain complete ==="
