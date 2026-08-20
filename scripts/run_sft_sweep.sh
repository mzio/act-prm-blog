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
# LR sweep: unset (default) reproduces the original lr=4e-5 runs with their original
# run_tags. Set LR=1e-4 to append --learning_rate and tag the runs
# <dom>_s2_<variant>_lr1e-4_heldout[_fullctx], so they neither collide with nor are
# mistaken for the 4e-5 runs -- and so analyze_sft.py still parses them (the glob is
# <dom>_s2_*, the regime is still the _fullctx suffix, and the lr rides in the
# variant column). NB the 4e-5 runs left an adapter that barely moved off its
# init (max|B@A| ~ 4e-5 vs base weights ~1e-2), which is why this knob exists.
LR="${LR:-}"
LR_ARGS=(); LRTAG=""
if [ -n "$LR" ]; then LR_ARGS=(--learning_rate "$LR"); LRTAG="_lr${LR}"; fi
# Short-probe knobs: cheap validation that the adapter actually moves before
# committing the full matrix (NUM_BATCHES=8 EVAL_EVERY=2 VARIANTS=actions_only).
NUM_BATCHES="${NUM_BATCHES:-}"
EVAL_EVERY="${EVAL_EVERY:-}"
# Early-stop on held-out action PPL. Never fired at lr=4e-5 because the curve was
# flat to 0.1%; at a working LR the run should bend then overfit (49 train tasks at
# bs 4 => 60 batches ~= 5 epochs), so this is what keeps the sweep affordable.
PATIENCE="${PATIENCE:-3}"
EXTRA=(); [ -n "$NUM_BATCHES" ] && EXTRA+=(--num_batches "$NUM_BATCHES")
[ -n "$EVAL_EVERY" ] && EXTRA+=(--eval_every "$EVAL_EVERY")
[ "$PATIENCE" != "0" ] && EXTRA+=(--early_stop_patience "$PATIENCE")
ONLY="${VARIANTS:-}"   # space-separated variant labels to restrict to (default: all)
# ACTION_ONLY=1: supervise only the <tool_call>/Final Answer: tokens, masking the
# reasoning prefix out of the loss (the thought stays in the input and still
# conditions the prediction). Makes the TRAINED span identical to the span
# eval_actiononly_ppl/accuracy score, so the variants are compared like-for-like.
# Tagged _ao so these never collide with the whole-span runs.
ACTION_ONLY="${ACTION_ONLY:-0}"
AOTAG=""
if [ "$ACTION_ONLY" = 1 ]; then EXTRA+=(--train_action_only); AOTAG="_ao"; fi
# Model is parametrized: MODEL_CFG selects both the <MODEL> path dir AND (exported for the
# train_sft.sh children) their --model_config. Default keeps 4B behavior.
MODEL="${MODEL_CFG:-hf_qwen3_4b_instruct}"; export MODEL_CFG="$MODEL"
CKROOT="checkpoints_lora/${ENVCFG//\//_}/$MODEL"
CORPUS="data/sft_corpus/$ENVNAME"
MDIR="/tmp/aprm/sft_sweep_$ENVNAME"; mkdir -p "$MDIR"

log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$MDIR/sweep.log"; }
wait_gpu_free(){ while pgrep -f '[m]ain_pytorch.py' >/dev/null 2>&1; do sleep 60; done; sleep 10; }
newest(){ ls -dt $1 2>/dev/null | head -1; }
# a corpus is usable only if train.json is NON-EMPTY (guards the old train:0 relabel bug)
corpus_ok(){ [ -s "$1/train.json" ] && [ "$(python3 -c "import json;print(len(json.load(open('$1/train.json'))))" 2>/dev/null || echo 0)" -gt 0 ]; }

run_one(){  # $1=sft_variant  $2=label(run_tag)  $3=regime(hide|full)  $4..=extra flags
  local variant=$1 label=$2 regime=$3; shift 3
  if [ -n "$ONLY" ] && [[ " $ONLY " != *" $label "* ]]; then return 0; fi
  local tag="${DOM}_s2_${label}${AOTAG}${LRTAG}_heldout"; [ "$regime" = full ] && tag="${tag}_fullctx"
  if [ -n "$(newest "$CKROOT/${tag}-*/step_best")" ]; then log "$tag: done, skip"; return 0; fi
  local fc=(); [ "$regime" = full ] && fc=(SFT_FULLCTX=1)
  log "$tag: SFT ($regime-obs, lr=${LR:-default}) $*"
  wait_gpu_free
  env "${fc[@]}" ./scripts/train_sft.sh "$ENVCFG" "$variant" \
      --run_tag "$tag" --best_metric eval_action_ppl \
      "${LR_ARGS[@]}" "${EXTRA[@]}" "$@" > "$MDIR/${tag}.log" 2>&1 \
    && log "$tag: done" || log "$tag: FAILED (see $MDIR/${tag}.log)"
}

# Thought corpora come from two EM relabel checkpoints: "" = step_best (early-peaked),
# _last = step_last (fully trained). Each is an SFT arm. best-corpus is required; _last
# is skipped if its relabel/export hasn't produced it.
for k in "" _last; do
  corpus_ok "$CORPUS/policy$k" || log "WARN: $CORPUS/policy$k missing/empty (thoughts_policy$k skipped)"
  corpus_ok "$CORPUS/base$k"   || log "WARN: $CORPUS/base$k missing/empty (thoughts_base$k skipped)"
done

log "=== SFT sweep $ENVCFG : {actions_only, expert_thoughts, thoughts_{policy,base}x{best,last}} x {hide,full} ==="
for regime in hide full; do
  run_one actions_only    actions_only    "$regime"
  run_one expert_thoughts expert_thoughts "$regime"
  for k in "" _last; do   # "" = step_best corpus, _last = step_last corpus
    corpus_ok "$CORPUS/policy$k" && run_one thoughts_policy "thoughts_policy$k" "$regime" --dataset_path "$CORPUS/policy$k"
    corpus_ok "$CORPUS/base$k"   && run_one thoughts_base   "thoughts_base$k"   "$regime" --dataset_path "$CORPUS/base$k"
  done
done
log "=== SFT sweep $ENVCFG done ==="
