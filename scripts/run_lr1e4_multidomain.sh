#!/usr/bin/env bash
# lr 1e-4 Stage-2 (sft_flat, AdamW) + rollouts for airline / finance / insurance.
# Design, rationale and results: notes/cc-11.0-lr1e4-multidomain-plan.md
#
# Arms: actions_only, expert_thoughts_all, thoughts_policy_adamw30.
# Plain expert_thoughts is DROPPED -- ~50% of its targets are bare <tool_call>, which
# teaches the model not to think (5.6% of retail rollout turns carried reasoning vs 19.7%
# for thoughts_policy). expert_thoughts_all filters to reasoning-bearing targets.
set -uo pipefail
cd /home/mzio/projects/act-prm-blog
[ -f scripts/wandb_preflight.sh ] && . scripts/wandb_preflight.sh
export PYTHONUNBUFFERED=1 HF_HOME=/data/users/mzio/models/hf_cache
MDIR=/tmp/aprm/lr1e4_multi; mkdir -p "$MDIR"
L="$MDIR/driver.log"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$L"; }

declare -A ENVOF=( [airline]=act_prm/tau2_airline
                   [finance]=act_prm/snorkel_finance_split
                   [insurance]=act_prm/snorkel_insurance )
declare -A BASEPOOL=( [airline]=data/tau2_airline
                      [finance]=data/snorkel_finance_split_v3
                      [insurance]=data/snorkel_insurance_split )
# finance's DERIVED expert pool is the pre-v3 one (3/25 eval overlap) -- must override.
declare -A EXPERTPOOL=( [airline]=data/tau2_airline_expert_thoughts
                        [finance]=data/snorkel_finance_split_expert_thoughts_v3
                        [insurance]=data/snorkel_insurance_expert_thoughts )

DOMAINS="${ONLY_DOMAINS:-airline finance insurance}"
log "=== lr 1e-4 multidomain: domains='$DOMAINS' arms={actions_only,expert_thoughts_all,thoughts_policy_adamw30} ==="

for dom in $DOMAINS; do
  env="${ENVOF[$dom]}"; envname="${env##*/}"
  corpus="data/sft_corpus/$envname/policy_adamw30"
  [ -s "$corpus/train.json" ] || { log "$dom: no policy_adamw30 corpus -- SKIP"; continue; }
  [ -f "$MDIR/$dom.train.done" ] && { log "$dom: training done, skip"; continue; }

  # GATE: every arm must score the SAME eval set, else the comparison is meaningless.
  if ! uv run --no-project python scripts/check_arm_pools.py \
        "${BASEPOOL[$dom]}" "${EXPERTPOOL[$dom]}" "$corpus" >> "$MDIR/$dom.poolcheck.log" 2>&1; then
    log "$dom: ARM POOL MISMATCH -- SKIPPING (see $MDIR/$dom.poolcheck.log)"; continue
  fi
  log "$dom: pool check OK"

  log "TRAIN $dom (3 arms, lr 1e-4)"
  TRAINER_CFG=sft_flat EXPERT_POOL="${EXPERTPOOL[$dom]}" \
  LR=1e-4 OPTIMIZER=adamw NUM_BATCHES=200 EVAL_EVERY=5 STEPS_PER_BATCH=32 PATIENCE=6 \
  VARIANTS="actions_only expert_thoughts_all thoughts_policy_adamw30" \
  REGIMES=hide CORPUS_VARIANTS="_adamw30" TAGSFX=_flat32 \
    ./scripts/run_sft_sweep.sh "$env" >> "$MDIR/$dom.train.log" 2>&1 \
    && { touch "$MDIR/$dom.train.done"; log "$dom: training done"; } \
    || log "$dom: training FAILED (rc=$?)"

  # per-arm grid (PPL / accuracy / max|B@A| / median|B@A| per snapshot)
  for arm in actions_only expert_thoughts_all thoughts_policy_adamw30; do
    {
      echo "======== $dom / $arm"
      uv run --no-project python scripts/report_checkpoint_grid.py \
        --env "act_prm_${envname}" \
        --run "${dom}_s2_${arm}_lr1e_4_adamw_nb200_flat32_heldout-*" 2>&1
    } >> "$MDIR/grids.txt"
  done
done
log "=== training phase complete ==="
