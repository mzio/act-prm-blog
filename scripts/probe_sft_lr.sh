#!/usr/bin/env bash
# LR probe for Stage-2 SFT: does the LoRA adapter actually move?
#
# The whole PyTorch path was run at lr=4e-5 (configs/trainer/{sft,pg}.yaml, never
# tuned since the initial commit) and every stage produced a near-null adapter:
# max|B@A| ~ 1e-6..4e-5 against base weights of order 1e-2, with lora_A still pinned
# at its seeded init. Held-out PPL consequently moved 0.08-0.26% over 60 batches.
# Before spending ~3.5 GPU-days on a full 72-run LR sweep, this runs one cheap arm
# per LR and reports how far the adapter travelled.
#
# Deliberately tagged `<dom>_s2probe_*` so these never match the `<dom>_s2_*` glob
# that analyze_sft.py / plot_sft_curves.py / the RL drivers use.
#
# Usage: CUDA_VISIBLE_DEVICES=0 nohup ./scripts/probe_sft_lr.sh > /tmp/aprm/lrprobe.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/usr/local/bin:$PATH"

ENVCFG="${ENVCFG:-act_prm/tau2_retail}"
DOM="${ENVCFG##*/}"; DOM="${DOM#tau2_}"
VARIANT="${VARIANT:-actions_only}"     # actions_only needs no corpus -> cheapest arm
BATCHES="${BATCHES:-8}"
LRS="${LRS:-4e-5 1e-4 1e-3}"           # 4e-5 is the control (current default)
# Thought variants SFT on a fixed Stage-1.5 corpus; without --dataset_path they would
# regenerate thoughts on the fly, which is a different (and much slower) experiment.
# thoughts_policy -> policy corpus, thoughts_base -> base corpus.
ENVNAME="${ENVCFG##*/}"
DS_ARGS=()
case "$VARIANT" in
  thoughts_policy) DS_ARGS=(--dataset_path "data/sft_corpus/$ENVNAME/policy") ;;
  thoughts_base)   DS_ARGS=(--dataset_path "data/sft_corpus/$ENVNAME/base") ;;
esac
[ ${#DS_ARGS[@]} -gt 0 ] && { [ -s "${DS_ARGS[1]}/train.json" ] || { echo "FATAL: missing corpus ${DS_ARGS[1]}"; exit 1; }; }
MDIR=/tmp/aprm/lrprobe; mkdir -p "$MDIR"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$MDIR/probe.log"; }
wait_gpu_free(){ while pgrep -f '[m]ain_pytorch.py' >/dev/null 2>&1; do sleep 30; done; sleep 5; }

log "=== SFT LR probe: env=$ENVCFG variant=$VARIANT batches=$BATCHES lrs='$LRS' ==="
for lr in $LRS; do
  tag="${DOM}_s2probe_${VARIANT}_lr${lr}"
  log "--- $tag"
  wait_gpu_free
  ./scripts/train_sft.sh "$ENVCFG" "$VARIANT" \
      --run_tag "$tag" --best_metric eval_action_ppl \
      --learning_rate "$lr" --num_batches "$BATCHES" --eval_every "${EVAL_EVERY:-10}" \
      "${DS_ARGS[@]}" \
      > "$MDIR/${tag}.log" 2>&1 \
    && log "$tag: done" || log "$tag: FAILED (see $MDIR/${tag}.log)"
done
log "=== probe runs done; measuring adapters ==="
uv run --no-project python scripts/report_lr_probe.py 2>&1 | tee -a "$MDIR/probe.log"
