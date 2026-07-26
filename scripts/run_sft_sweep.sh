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

run_one(){  # $1=sft_variant  $2=label(run_tag)  $3=regime(hide|full)  $4..=extra flags
  local variant=$1 label=$2 regime=$3; shift 3
  local tag="${DOM}_s2_${label}_heldout"; [ "$regime" = full ] && tag="${tag}_fullctx"
  if [ -n "$(newest "$CKROOT/${tag}-*/step_best")" ]; then log "$tag: done, skip"; return 0; fi
  local fc=(); [ "$regime" = full ] && fc=(SFT_FULLCTX=1)
  log "$tag: SFT ($regime-obs) $*"
  wait_gpu_free
  env "${fc[@]}" ./scripts/train_sft.sh "$ENVCFG" "$variant" \
      --run_tag "$tag" --best_metric eval_action_ppl "$@" > "$MDIR/${tag}.log" 2>&1 \
    && log "$tag: done" || log "$tag: FAILED (see $MDIR/${tag}.log)"
}

# Thought corpora come from two EM relabel checkpoints: "" = step_best (early-peaked),
# _last = step_last (fully trained). Each is an SFT arm. best-corpus is required; _last
# is skipped if its relabel/export hasn't produced it.
for k in "" _last; do
  [ -f "$CORPUS/policy$k/train.json" ] || log "WARN: $CORPUS/policy$k missing (thoughts_policy$k skipped)"
  [ -f "$CORPUS/base$k/train.json" ]   || log "WARN: $CORPUS/base$k missing (thoughts_base$k skipped)"
done

log "=== SFT sweep $ENVCFG : {actions_only, expert_thoughts, thoughts_{policy,base}x{best,last}} x {hide,full} ==="
for regime in hide full; do
  run_one actions_only    actions_only    "$regime"
  run_one expert_thoughts expert_thoughts "$regime"
  for k in "" _last; do   # "" = step_best corpus, _last = step_last corpus
    [ -f "$CORPUS/policy$k/train.json" ] && run_one thoughts_policy "thoughts_policy$k" "$regime" --dataset_path "$CORPUS/policy$k"
    [ -f "$CORPUS/base$k/train.json" ]   && run_one thoughts_base   "thoughts_base$k"   "$regime" --dataset_path "$CORPUS/base$k"
  done
done
log "=== SFT sweep $ENVCFG done ==="
