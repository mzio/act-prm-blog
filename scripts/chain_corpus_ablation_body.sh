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

G=/tmp/aprm/corpus_abl; mkdir -p "$G"
LOG="$G/chain.log"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
reap(){ for p in $(pgrep -f 'main_pytorch\.py'); do kill -9 "$p" 2>/dev/null; done; sleep 20; }


# CORPUS_VARIANTS="" -> label "thoughts_policy", dataset_path data/sft_corpus/<env>/policy
# (the OLD relabel). VARIANTS must match that LABEL exactly -- passing the wrong label is
# what made an earlier 8.5h run silently no-op in 20s.
# BYPASSES run_sft_sweep.sh ON PURPOSE. The sweep reaches thoughts_policy only inside
# `for k in $CORPUS_VARIANTS`, and CORPUS_VARIANTS="" does NOT give one iteration with an
# empty suffix: `${CORPUS_VARIANTS:-"" _last}` treats empty as UNSET and substitutes the
# default, so the loop ran only k=_last -> label "thoughts_policy_last", which never
# matched VARIANTS="thoughts_policy". Zero arms, sweep exited 0 in 60s, and the chain
# aborted with "produced 0 snapshot(s)". Calling train_sft.sh directly sidesteps the loop.
TAG=retail_s2_thoughts_policy_lr1e_3_nb200_flat32sgdOLDC_heldout
if [ ! -f "$G/train.done" ]; then
  log "TRAIN retail thoughts_policy on the OLD corpus (sgd 1e-3, sft_flat, nb200) -- ~1.6h"
  TRAINER_CFG=sft_flat ./scripts/train_sft.sh act_prm/tau2_retail thoughts_policy \
      --run_tag "$TAG" --best_metric eval_actiononly_ppl \
      --learning_rate 1e-3 --optimizer sgd --num_batches 200 --eval_every 5 \
      --steps_per_batch 32 --early_stop_patience 6 --hide_observations \
      --dataset_path data/sft_corpus/tau2_retail/policy >>"$G/train.log" 2>&1
  log "  rc=$?"
  n=$(ls -d checkpoints_lora/act_prm_tau2_retail/hf_qwen3_4b_instruct/${TAG}-*/step_* 2>/dev/null | wc -l)
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
