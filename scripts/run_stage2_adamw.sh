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

# W&B: api.wandb.ai is NOT reachable through fwdproxy. Without this, main_pytorch inits
# W&B online, the run hangs, and -- worse -- it hangs in atexit teardown
# (service_connection.teardown -> service_process.join -> subprocess.wait) waiting on the
# wandb service to flush, so the process looks ALIVE while doing nothing: 0% CPU, 4 MiB of
# GPU, no stdout (block-buffered to the log and never flushed). run_stage1_r32.sh:52 sources
# this; omitting it here cost a 21-minute no-op on 2026-08-29.
[ -f scripts/wandb_preflight.sh ] && . scripts/wandb_preflight.sh
# Unbuffered so a stall is visible in the log instead of sitting in a 4 KB stdio buffer.
export PYTHONUNBUFFERED=1
# Pin the model cache: an inherited HF_HOME (e.g. /home/mzio/models from an interactive
# shell) points away from the populated cache the configs expect and triggers a re-download.
export HF_HOME=/data/users/mzio/models/hf_cache
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
