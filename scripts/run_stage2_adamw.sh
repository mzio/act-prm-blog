#!/usr/bin/env bash
# Stage-2 SFT over the AdamW/lp0 Stage-1 corpora, all four domains, sequentially.
#
# Only three things change from the prior Stage-2 generation:
#   optimizer sgd -> adamw   (main_pytorch never passed a name, so optim.get_optimizer's
#                             name="sgd" default silently applied to EVERY earlier run)
#   lr 3e-3 -> 1e-3          (adamw's update is ~lr*sign(grad); 3e-3 is an SGD-era value)
#   corpus  -> policy_adamw30 (via CORPUS_VARIANTS; without it the sweep silently picks
#                             $CORPUS/policy, the OLD corpus, confounding optimizer with data)
# Everything else is held at the prior settings so the comparison is like-for-like:
#   num_batches 150, eval_every 10, batch_size 4, hide_observations, r8_a16_linear,
#   ACTION_ONLY=0 (whole-span loss; eval_actiononly_ppl/accuracy still reported).
set -uo pipefail
cd /home/mzio/projects/act-prm-blog
MDIR=/tmp/aprm/stage2_adamw; mkdir -p "$MDIR"
L="$MDIR/stage2_adamw.log"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$L"; }

DOMAINS="${ONLY_DOMAINS:-retail airline finance insurance}"
declare -A ENVOF=(
  [retail]=act_prm/tau2_retail
  [airline]=act_prm/tau2_airline
  [finance]=act_prm/snorkel_finance_split
  [insurance]=act_prm/snorkel_insurance
)
LR="${LR:-1e-3}"; OPTIMIZER="${OPTIMIZER:-adamw}"
NUM_BATCHES="${NUM_BATCHES:-150}"; EVAL_EVERY="${EVAL_EVERY:-10}"

log "=== Stage-2 $OPTIMIZER lr=$LR nb=$NUM_BATCHES corpus=policy_adamw30 domains='$DOMAINS' ==="
for dom in $DOMAINS; do
  env="${ENVOF[$dom]:-}"
  [ -z "$env" ] && { log "$dom: unknown domain, skipping"; continue; }
  envname="${env##*/}"
  if [ ! -s "data/sft_corpus/$envname/policy_adamw30/train.json" ]; then
    log "$dom: no policy_adamw30 corpus -- SKIPPING"; continue
  fi
  [ -f "$MDIR/$dom.done" ] && { log "$dom: done, skip"; continue; }
  log "SFT $dom ($env)"
  LR="$LR" OPTIMIZER="$OPTIMIZER" NUM_BATCHES="$NUM_BATCHES" EVAL_EVERY="$EVAL_EVERY" \
    REGIMES=hide CORPUS_VARIANTS="_adamw30" \
    ./scripts/run_sft_sweep.sh "$env" >> "$MDIR/$dom.log" 2>&1 \
    && { touch "$MDIR/$dom.done"; log "$dom: done"; } \
    || log "$dom: FAILED (rc=$?)"
done
log "=== Stage-2 adamw sweep complete ==="
