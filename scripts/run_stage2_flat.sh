#!/usr/bin/env bash
# Stage-2 SFT via the FLAT trainer (corpus-wide step-level sampling), all four domains.
#
# vs scripts/run_stage2_adamw.sh (the 12 completed arms), three deliberate changes:
#   trainer  sft -> sft_flat   : batches drawn from the whole corpus, so one update sees
#                                ~32 steps from ~32 different trajectories instead of every
#                                step of 4 trajectories. Also drops the per-step generator
#                                forward (measured 19.0s -> 3.4s per training step).
#   batches  100 -> 200        : steps_per_batch 32 vs sft's implicit ~64-70 means half the
#                                data per update, so 200 batches holds total step-visits
#                                (~6400) constant. NOTE it is 2x the optimizer steps.
#   eval_every 10 -> 5         : eval is cheaper now; patience 6 keeps the SAME
#                                "30 batches without improvement" stopping rule.
#
# Correctness gates run before each domain:
#   * check_arm_pools.py -- every arm must share the base eval set. finance needs
#     EXPERT_POOL=..._v3 or its expert arm is scored on a 3/25-overlapping eval set.
set -uo pipefail
cd /home/mzio/projects/act-prm-blog
[ -f scripts/wandb_preflight.sh ] && . scripts/wandb_preflight.sh
export PYTHONUNBUFFERED=1
export HF_HOME=/data/users/mzio/models/hf_cache

MDIR=/tmp/aprm/stage2_flat; mkdir -p "$MDIR"
L="$MDIR/stage2_flat.log"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$L"; }

DOMAINS="${ONLY_DOMAINS:-retail airline finance insurance}"
declare -A ENVOF=( [retail]=act_prm/tau2_retail [airline]=act_prm/tau2_airline
                   [finance]=act_prm/snorkel_finance_split [insurance]=act_prm/snorkel_insurance )
declare -A BASEPOOL=( [retail]=data/tau2_retail [airline]=data/tau2_airline
                      [finance]=data/snorkel_finance_split_v3 [insurance]=data/snorkel_insurance_split )
# finance's derived expert pool is the pre-v3 one (eval overlap 3/25) -- override it.
# retail's stock expert pool has 10 eval trajectories vs the base pool's 8, so that arm
# was scored on 115 steps against the others' 92 -- use the matched (filtered) pool.
declare -A EXPERTPOOL=( [retail]=data/tau2_retail_expert_thoughts_matched
                        [airline]=data/tau2_airline_expert_thoughts
                        [finance]=data/snorkel_finance_split_expert_thoughts_v3
                        [insurance]=data/snorkel_insurance_expert_thoughts )

LR="${LR:-1e-3}"; OPTIMIZER="${OPTIMIZER:-adamw}"
NUM_BATCHES="${NUM_BATCHES:-200}"; EVAL_EVERY="${EVAL_EVERY:-5}"
SPB="${STEPS_PER_BATCH:-32}"; PATIENCE="${PATIENCE:-6}"
TAG="${TAGSFX:-_flat${SPB}}"

log "=== Stage-2 FLAT  opt=$OPTIMIZER lr=$LR nb=$NUM_BATCHES spb=$SPB eval_every=$EVAL_EVERY patience=$PATIENCE tag=$TAG ==="
for dom in $DOMAINS; do
  env="${ENVOF[$dom]:-}"; [ -z "$env" ] && { log "$dom: unknown"; continue; }
  envname="${env##*/}"
  corpus="data/sft_corpus/$envname/policy_adamw30"
  [ -s "$corpus/train.json" ] || { log "$dom: no policy_adamw30 corpus -- SKIP"; continue; }
  [ -f "$MDIR/$dom.done" ] && { log "$dom: done, skip"; continue; }

  # GATE: all arms must share the base eval set, else the comparison is meaningless.
  if ! uv run --no-project python scripts/check_arm_pools.py \
        "${BASEPOOL[$dom]}" "${EXPERTPOOL[$dom]}" "$corpus" >> "$MDIR/$dom.poolcheck.log" 2>&1; then
    log "$dom: ARM POOL MISMATCH -- SKIPPING (see $MDIR/$dom.poolcheck.log)"; continue
  fi
  log "$dom: pool check OK"

  log "SFT-flat $dom ($env)"
  TRAINER_CFG=sft_flat EXPERT_POOL="${EXPERTPOOL[$dom]}" \
  LR="$LR" OPTIMIZER="$OPTIMIZER" NUM_BATCHES="$NUM_BATCHES" EVAL_EVERY="$EVAL_EVERY" \
  STEPS_PER_BATCH="$SPB" PATIENCE="$PATIENCE" TAGSFX="$TAG" \
  REGIMES=hide CORPUS_VARIANTS="_adamw30" \
    ./scripts/run_sft_sweep.sh "$env" >> "$MDIR/$dom.log" 2>&1 \
    && { touch "$MDIR/$dom.done"; log "$dom: done"; } \
    || log "$dom: FAILED (rc=$?)"
done
log "=== Stage-2 FLAT sweep complete ==="
