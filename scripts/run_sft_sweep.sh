#!/usr/bin/env bash
# Stage-2 SFT sweep for ONE env: 4 variants x {hide-obs, full-context} = 8 runs.
# Resumable (skips any run whose step_best exists), serial by default (one GPU).
# Requires the Stage-1.5 corpora to already exist at data/sft_corpus/<envname>/{policy,base}
# (built by relabel + scripts/export_sft_corpus.py). Env-agnostic: works for retail,
# airline, finance — the run_tags match run_retail_matrix.sh so the hide-obs runs it
# already produced are detected and skipped.
#
# Usage:  CUDA_VISIBLE_DEVICES=0 nohup ./scripts/run_sft_sweep.sh act_prm/tau2_airline \
#           > /tmp/aprm/sft_sweep.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/usr/local/bin:$PATH"

ENVCFG="${1:?env config, e.g. act_prm/tau2_retail}"
ENVNAME="${ENVCFG##*/}"                     # tau2_retail
DOM="${ENVNAME#tau2_}"                      # retail / airline (matches existing run_tags)
MODEL=hf_qwen3_4b_instruct
CKROOT="checkpoints_lora/${ENVCFG//\//_}/$MODEL"
CORPUS="data/sft_corpus/$ENVNAME"
MDIR="/tmp/aprm/sft_sweep_$ENVNAME"; mkdir -p "$MDIR"

log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$MDIR/sweep.log"; }
wait_gpu_free(){ while pgrep -f '[m]ain_pytorch.py' >/dev/null 2>&1; do sleep 60; done; sleep 10; }
newest(){ ls -dt $1 2>/dev/null | head -1; }

run_one(){  # $1=variant  $2=regime(hide|full)  $3..=extra flags (e.g. --dataset_path ...)
  local variant=$1 regime=$2; shift 2
  local tag="${DOM}_s2_${variant}_heldout"; [ "$regime" = full ] && tag="${tag}_fullctx"
  if [ -n "$(newest "$CKROOT/${tag}-*/step_best")" ]; then log "$tag: done, skip"; return 0; fi
  local fc=(); [ "$regime" = full ] && fc=(SFT_FULLCTX=1)
  log "$tag: SFT ($regime-obs) $*"
  wait_gpu_free
  env "${fc[@]}" ./scripts/train_sft.sh "$ENVCFG" "$variant" \
      --run_tag "$tag" --best_metric eval_action_ppl "$@" > "$MDIR/${tag}.log" 2>&1 \
    && log "$tag: done" || log "$tag: FAILED (see $MDIR/${tag}.log)"
}

[ -f "$CORPUS/policy/train.json" ] || log "WARN: $CORPUS/policy missing — thoughts_policy will fail (run Stage 1.5 first)"
[ -f "$CORPUS/base/train.json" ]   || log "WARN: $CORPUS/base missing — thoughts_base will fail (run Stage 1.5 first)"

log "=== SFT sweep $ENVCFG : 4 variants x {hide-obs, full-context} ==="
for regime in hide full; do
  run_one actions_only    "$regime"
  run_one expert_thoughts "$regime"
  run_one thoughts_policy "$regime" --dataset_path "$CORPUS/policy"
  run_one thoughts_base   "$regime" --dataset_path "$CORPUS/base"
done
log "=== SFT sweep $ENVCFG done ==="
