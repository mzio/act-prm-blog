#!/usr/bin/env bash
# Does the OLD (SGD-era) Stage-2 recipe still produce a good agent when fed the NEW
# AdamW-derived Stage-1 corpus?  Fills the missing cell of:
#
#                       Stage-1 corpus: SGD-era     Stage-1 corpus: AdamW (policy_adamw30)
#   Stage-2 SGD 3e-3         20.5% (known good)          <-- THIS RUN
#   Stage-2 AdamW            (never run)                 0.0%-9.1% (collapsed)
#
# Recipe held EXACTLY at the known-good config so the corpus is the only variable:
#   trainer sft (not sft_flat) | optimizer sgd | lr 3e-3 | nb 150 | batch_size 4
#   best_metric eval_action_ppl | hide_observations
#
# TWO arms:
#   control : old `policy` corpus  -> must reproduce ~20.5%. If it does not, something in
#             the code has drifted since 08-21 and the comparison is void. This control is
#             the point -- a lot has changed (optimizer plumbing, best_metric, arm pools,
#             run-dir rotation), and without it a low number is uninterpretable.
#   test    : new `policy_adamw30` corpus -> the question.
#
# Known confound, recorded not hidden: the old corpus has 80 train entries vs the new
# one's 52 (the old carries epoch-wrap duplicate trajectories), so the test arm also sees
# ~35% less data. A drop cannot be attributed to corpus QUALITY alone.
set -uo pipefail
cd /home/mzio/projects/act-prm-blog
[ -f scripts/wandb_preflight.sh ] && . scripts/wandb_preflight.sh
export PYTHONUNBUFFERED=1 HF_HOME=/data/users/mzio/models/hf_cache
MDIR=/tmp/aprm/sgd_newcorpus; mkdir -p "$MDIR"
L="$MDIR/driver.log"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$L"; }

log "=== SGD 3e-3 / sft / nb150 on old vs new Stage-1 corpus ==="
for spec in "control:data/sft_corpus/tau2_retail/policy" \
            "test:data/sft_corpus/tau2_retail/policy_adamw30"; do
  which="${spec%%:*}"; corpus="${spec##*:}"
  tag="sgdrepro_${which}"
  [ -f "$MDIR/$tag.done" ] && { log "$tag: done, skip"; continue; }
  log "TRAIN $tag  corpus=$corpus"
  ./scripts/train_sft.sh act_prm/tau2_retail thoughts_policy \
      --dataset_path "$corpus" --run_tag "retail_s2_${tag}_sgd3e_3_nb150_heldout" \
      --learning_rate 3e-3 --optimizer sgd --num_batches 150 --eval_every 10 \
      --best_metric eval_action_ppl --hide_observations \
      >> "$MDIR/$tag.log" 2>&1 \
    && { touch "$MDIR/$tag.done"; log "$tag: done"; } || log "$tag: FAILED (rc=$?)"
done

log "=== rollouts ==="
for which in control test; do
  tag="sgdrepro_${which}"
  [ -f "$MDIR/$tag.rollout.done" ] && { log "$tag rollout done, skip"; continue; }
  ck=$(ls -dt checkpoints_lora/act_prm_tau2_retail/hf_qwen3_4b_instruct/retail_s2_${tag}_sgd3e_3_nb150_heldout-*/step_best 2>/dev/null | head -1)
  [ -z "$ck" ] && { log "$tag: no checkpoint, skip rollout"; continue; }
  log "ROLLOUT $tag <- $ck"
  VARIANTS="${tag}" CKPT_PAT="sgd3e_3_nb150" CKPT_TAG="sgdrepro" \
    ./scripts/run_sft_rollout_eval.sh >> "$L" 2>&1 || log "  $tag rollout FAILED"
  touch "$MDIR/$tag.rollout.done"
  for p in $(ps -eo pid,args | awk '$2=="bash" && $3 ~ /run_sft_rollout_eval\.sh$/ {print $1}'); do kill -9 "$p" 2>/dev/null; done
  for p in $(pgrep -f 'main_pytorch\.py'); do kill -9 "$p" 2>/dev/null; done
  sleep 8
done
log "=== complete ==="
