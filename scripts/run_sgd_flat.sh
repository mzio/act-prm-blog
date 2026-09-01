#!/usr/bin/env bash
# SGD Stage-2 on the CURRENT pipeline. Everything about the ongoing lr 1e-4 runs is
# preserved -- sft_flat (corpus-wide step shuffling), nb200, steps_per_batch 32,
# eval_every 5, patience 6, save_every 10, hide-obs, best_metric eval_actiononly_ppl,
# and the same policy_adamw30 corpus. The ONLY change is the optimizer.
#
# Why: AdamW collapsed agentic behaviour at every LR tried (1e-3 -> 0.0%, 1e-4 -> 9.1%,
# both below the untrained base model's 14.3%), while the SGD-era checkpoints reached
# 20.5%. Median weight drift was ruled out as the mechanism (the lr 1e-4 ladder inverted
# the predicted ordering), so the remaining suspect is the optimizer's UPDATE CHARACTER:
# SGD moves a few coordinates a long way (median 4.7e-05 against max 4.5e-02), AdamW moves
# everything (median 2.2e-03) because its step is per-coordinate scale-invariant.
#
# TWO learning rates, because they test different things:
# LR 1e-3: between the SGD LR that produced the good checkpoints (3e-3) and the AdamW LR
# whose collapse we are chasing (1e-3). NOTE SGD and AdamW are NOT comparable at equal LR:
# AdamW's step is ~lr*sign(grad) while SGD's is lr*grad, and with LoRA grads ~1e-3 that is
# a ~1000x difference in effective step size. SGD 1e-3 is therefore ~3x below the working
# SGD setting, not "the same as" the AdamW 1e-3 run. Measured anchors: SGD 4e-5 was a total
# no-op (max|B@A| 6.7e-07); SGD 3e-3 gave max 4.5e-02 / median 4.7e-05 and 20.5% rollout.
set -uo pipefail
cd /home/mzio/projects/act-prm-blog
[ -f scripts/wandb_preflight.sh ] && . scripts/wandb_preflight.sh
export PYTHONUNBUFFERED=1 HF_HOME=/data/users/mzio/models/hf_cache
MDIR=/tmp/aprm/sgd_flat; mkdir -p "$MDIR"
L="$MDIR/driver.log"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$L"; }

ARMS="${ARMS:-thoughts_policy_adamw30 actions_only}"
LRS="${LRS:-1e-3}"
log "=== SGD on sft_flat: arms='$ARMS' lrs='$LRS' (all else identical to the lr 1e-4 runs) ==="

for lr in $LRS; do
  lrtag="lr${lr//[-.]/_}"
  [ -f "$MDIR/train_$lrtag.done" ] && { log "train $lrtag done, skip"; continue; }
  log "TRAIN sgd $lr  (sft_flat, nb200, spb32)"
  TRAINER_CFG=sft_flat EXPERT_POOL=data/tau2_retail_expert_thoughts_matched \
  LR="$lr" OPTIMIZER=sgd NUM_BATCHES=200 EVAL_EVERY=5 STEPS_PER_BATCH=32 PATIENCE=6 \
  VARIANTS="$ARMS" REGIMES=hide CORPUS_VARIANTS="_adamw30" TAGSFX=_flat32sgd \
    ./scripts/run_sft_sweep.sh act_prm/tau2_retail >> "$MDIR/train_$lrtag.log" 2>&1 \
    && { touch "$MDIR/train_$lrtag.done"; log "train $lrtag done"; } || log "train $lrtag FAILED (rc=$?)"
  for arm in $ARMS; do
    { echo "======== sgd $lr / $arm"
      uv run --no-project python scripts/report_checkpoint_grid.py \
        --run "retail_s2_${arm}_${lrtag}_sgd_nb200_flat32sgd_heldout-*" 2>&1
    } >> "$MDIR/grids.txt"
  done
done

log "=== rollouts ==="
for lr in $LRS; do
  lrtag="lr${lr//[-.]/_}"
  for arm in $ARMS; do
    for step in step_0020 step_best; do
      key="$lrtag.$arm.$step"
      [ -f "$MDIR/roll_$key.done" ] && { log "rollout $key done, skip"; continue; }
      ck=$(ls -d checkpoints_lora/act_prm_tau2_retail/hf_qwen3_4b_instruct/retail_s2_${arm}_${lrtag}_sgd_nb200_flat32sgd_heldout-*/${step} 2>/dev/null | head -1)
      [ -z "$ck" ] && { log "$key: no checkpoint, skip"; continue; }
      log "ROLLOUT $key"
      VARIANTS="$arm" CKPT_PAT="${lrtag}_sgd_nb200_flat32sgd" CKPT_TAG="sgd${lrtag}" CKPT_STEP="$step" \
        ./scripts/run_sft_rollout_eval.sh >> "$L" 2>&1 || log "  $key FAILED"
      touch "$MDIR/roll_$key.done"
      for p in $(ps -eo pid,args | awk '$2=="bash" && $3 ~ /run_sft_rollout_eval\.sh$/ {print $1}'); do kill -9 "$p" 2>/dev/null; done
      for p in $(pgrep -f 'main_pytorch\.py'); do kill -9 "$p" 2>/dev/null; done
      sleep 8
    done
  done
done
log "=== complete ==="
