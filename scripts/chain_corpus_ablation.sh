#!/usr/bin/env bash
# Single-variable test: is the adamw30 Stage-1 relabel WORSE than the old one for
# downstream agentic behaviour?
#
# Motivation. Old-vs-new SGD 3e-3 on retail moved in OPPOSITE directions by arm:
#   actions_only     11.9% -> 16.7%  (Fisher p=0.756)
#   thoughts_policy  21.4% ->  9.5%  (Fisher p=0.227)
# Neither is significant, but the asymmetry is informative: actions_only is corpus-
# INDEPENDENT (it trains on expert actions), so for that arm old-vs-new isolates the
# trainer change (sft/nb150 -> sft_flat/nb200) and it went UP. thoughts_policy changed
# BOTH the trainer AND the Stage-1 corpus (tau2_retail/policy -> policy_adamw30). So the
# thought-arm drop may be a Stage-1 effect that the trainer comparison cannot see.
#
# This run holds the current Stage-2 setup fixed (sft_flat, SGD 1e-3, nb200, spb32) and
# swaps ONLY the corpus back to the old data/sft_corpus/tau2_retail/policy. Compare its
# rollouts against retail thoughts_policy_adamw30 SGD 1e-3 (step_0020 31.0%, step_best
# 23.8%) -- same trainer, same optimizer, same LR, same tasks, different Stage-1 corpus.
#
# Queued LAST so it cannot delay the cc-13.0 multidomain sweep or the nb1000 retry.
set -uo pipefail
cd /home/mzio/projects/act-prm-blog
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/usr/local/bin:$PATH"
[ -f scripts/wandb_preflight.sh ] && . scripts/wandb_preflight.sh
export PYTHONUNBUFFERED=1 HF_HOME=/data/users/mzio/models/hf_cache
export ACT_PRM_DUMP_TRAJECTORIES=1

WAIT_PID="${WAIT_PID:?set WAIT_PID}"
G=/tmp/aprm/corpus_abl; mkdir -p "$G"
LOG="$G/chain.log"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
reap(){ for p in $(pgrep -f 'main_pytorch\.py'); do kill -9 "$p" 2>/dev/null; done; sleep 20; }

log "waiting on pid=$WAIT_PID"
while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
log "pid $WAIT_PID exited"; reap

# CORPUS_VARIANTS="" -> label "thoughts_policy", dataset_path data/sft_corpus/<env>/policy
# (the OLD relabel). VARIANTS must match that LABEL exactly -- passing the wrong label is
# what made an earlier 8.5h run silently no-op in 20s.
if [ ! -f "$G/train.done" ]; then
  log "TRAIN retail thoughts_policy on the OLD corpus (sgd 1e-3, sft_flat, nb200) -- ~1.6h"
  TRAINER_CFG=sft_flat LR=1e-3 OPTIMIZER=sgd NUM_BATCHES=200 EVAL_EVERY=5 \
  STEPS_PER_BATCH=32 PATIENCE=6 VARIANTS="thoughts_policy" \
  REGIMES=hide CORPUS_VARIANTS="" TAGSFX=_flat32sgdOLDC \
    ./scripts/run_sft_sweep.sh act_prm/tau2_retail >>"$G/train.log" 2>&1
  log "  rc=$?"
  n=$(ls -d checkpoints_lora/act_prm_tau2_retail/hf_qwen3_4b_instruct/retail_s2_thoughts_policy_lr1e_3_nb200_flat32sgdOLDC_heldout-*/step_* 2>/dev/null | wc -l)
  log "  produced $n snapshot(s)"
  [ "$n" -gt 0 ] && touch "$G/train.done" || { log "NO SNAPSHOTS -- aborting"; exit 1; }
  reap
fi

for step in step_0020 step_best; do
  m="$G/roll.$step.done"; [ -f "$m" ] && { log "roll $step done, skip"; continue; }
  log "ROLLOUT oldcorpus/$step"
  VARIANTS="thoughts_policy" CKPT_PAT="lr1e_3_nb200_flat32sgdOLDC" \
  CKPT_STEP="$step" CKPT_TAG="sgd1e_3_oldcorpus" ./scripts/run_sft_rollout_eval.sh >>"$LOG" 2>&1
  log "  rc=$?"; touch "$m"; reap
done
log "=== corpus ablation complete ==="
