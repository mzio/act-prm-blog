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
# Optimizer. Every Stage-2 result before 08-27 used plain SGD, because optim.get_optimizer
# defaults to name="sgd" and main_pytorch never passed one -- so the 3e-3 those runs used was
# "the LR that makes SGD limp", not a tuned value. AdamW needs ~1e-3 (MZ has run this before);
# 3e-3 under AdamW would be an enormous step since the update is roughly lr*sign(grad).
OPTIMIZER="${OPTIMIZER:-sgd}"
LR_ARGS=(); LRTAG=""
# NB sanitize the LR for the tag: main_pytorch's run-name builder rewrites "-" and "."
# to "_", so a tag of _lr1e-4 lands on disk as _lr1e_4 and the step_best skip-glob below
# would never match -> the sweep would re-run finished work forever. Pre-sanitize so the
# tag we search for is the tag that exists.
if [ -n "$LR" ]; then LR_ARGS=(--learning_rate "$LR"); LRTAG="_lr${LR//[-.]/_}"; fi
OPT_ARGS=(--optimizer "$OPTIMIZER")
[ "$OPTIMIZER" != "sgd" ] && LRTAG="${LRTAG}_${OPTIMIZER}"   # keep AdamW runs in distinct dirs
# Short-probe knobs: cheap validation that the adapter actually moves before
# committing the full matrix (NUM_BATCHES=8 EVAL_EVERY=2 VARIANTS=actions_only).
NUM_BATCHES="${NUM_BATCHES:-}"
EVAL_EVERY="${EVAL_EVERY:-}"
# Early-stop on held-out action PPL. Never fired at lr=4e-5 because the curve was
# flat to 0.1%; at a working LR the run should bend then overfit (49 train tasks at
# bs 4 => 60 batches ~= 5 epochs), so this is what keeps the sweep affordable.
PATIENCE="${PATIENCE:-3}"
NBTAG=""; [ -n "$NUM_BATCHES" ] && NBTAG="_nb${NUM_BATCHES}"
EXTRA=(); [ -n "$NUM_BATCHES" ] && EXTRA+=(--num_batches "$NUM_BATCHES")
[ -n "$EVAL_EVERY" ] && EXTRA+=(--eval_every "$EVAL_EVERY")
[ "$PATIENCE" != "0" ] && EXTRA+=(--early_stop_patience "$PATIENCE")
ONLY="${VARIANTS:-}"   # space-separated variant labels to restrict to (default: all)
# Which context regimes this invocation covers. Default both, but the matrix driver
# passes one at a time so it can run regime-major: ALL hide arms across every dataset
# before any full-context arm.
REGIMES="${REGIMES:-hide full}"
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
  # TAGSFX distinguishes otherwise-identical configs that differ by a CODE change
  # (e.g. _rshuffle for the per-epoch-reshuffle fix), so the runs land in separate dirs
  # and separate .done markers instead of colliding with the previous generation.
  local tag="${DOM}_s2_${label}${AOTAG}${LRTAG}${NBTAG}${TAGSFX:-}_heldout"; [ "$regime" = full ] && tag="${tag}_fullctx"
  # Completion is marked by an explicit .done file, NOT by step_best: step_best is
  # written at the FIRST eval (batch 10), so an interrupted run would otherwise look
  # finished and be skipped forever, silently leaving a half-trained arm in the matrix.
  if [ -f "$MDIR/${tag}.done" ]; then log "$tag: done, skip"; return 0; fi
  local fc=(); [ "$regime" = full ] && fc=(SFT_FULLCTX=1)
  log "$tag: SFT ($regime-obs, lr=${LR:-default}) $*"
  wait_gpu_free
  # RE-CHECK after waiting. wait_gpu_free can block for hours behind another run, and in
  # that window a different driver may have completed this very arm -- which is exactly
  # what happened on 08-20: a sweep queued at 14:14 waited 3.5h for a trainer that was
  # finishing THIS arm, then launched a duplicate into the same log dir and began
  # overwriting the finished curve.
  if [ -f "$MDIR/${tag}.done" ]; then log "$tag: completed by another driver while waiting, skip"; return 0; fi
  env "${fc[@]}" ./scripts/train_sft.sh "$ENVCFG" "$variant" \
      --run_tag "$tag" --best_metric eval_actiononly_ppl \
      "${LR_ARGS[@]}" "${OPT_ARGS[@]}" "${EXTRA[@]}" "$@" > "$MDIR/${tag}.log" 2>&1 \
    && { touch "$MDIR/${tag}.done"; log "$tag: done"; } || log "$tag: FAILED (see $MDIR/${tag}.log)"
}

# Thought corpora come from two EM relabel checkpoints: "" = step_best (early-peaked),
# _last = step_last (fully trained). Each is an SFT arm. best-corpus is required; _last
# is skipped if its relabel/export hasn't produced it.
# CORPUS_VARIANTS selects WHICH relabel corpora to use. Default "" _last is the historical
# pair (step_best / step_last of the SGD-era relabel). Set it explicitly to consume a
# regenerated corpus, e.g. CORPUS_VARIANTS="_adamw30" -> data/sft_corpus/<env>/policy_adamw30.
# Without this the sweep silently picks $CORPUS/policy -- the OLD corpus -- which would
# confound the optimizer change with a data change and quietly invalidate the comparison.
CORPUS_VARIANTS="${CORPUS_VARIANTS:-"" _last}"
for k in $CORPUS_VARIANTS; do
  corpus_ok "$CORPUS/policy$k" || log "WARN: $CORPUS/policy$k missing/empty (thoughts_policy$k skipped)"
  corpus_ok "$CORPUS/base$k"   || log "WARN: $CORPUS/base$k missing/empty (thoughts_base$k skipped)"
done

log "=== SFT sweep $ENVCFG : {actions_only, expert_thoughts, thoughts_{policy,base}x{best,last}} x {hide,full} ==="
for regime in $REGIMES; do
  run_one actions_only    actions_only    "$regime"
  run_one expert_thoughts expert_thoughts "$regime"
  for k in $CORPUS_VARIANTS; do
    corpus_ok "$CORPUS/policy$k" && run_one thoughts_policy "thoughts_policy$k" "$regime" --dataset_path "$CORPUS/policy$k"
    corpus_ok "$CORPUS/base$k"   && run_one thoughts_base   "thoughts_base$k"   "$regime" --dataset_path "$CORPUS/base$k"
  done
done
log "=== SFT sweep $ENVCFG done ==="
