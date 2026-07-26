#!/usr/bin/env bash
# Full retail experiment matrix, BOTH splits, serial on one GPU. Resumable: each
# step is skipped if its output already exists, so you can Ctrl-C / crash / re-run.
#
#   Stage 1     EM thought-gen           : {policy, base} x {heldout(A), all(B)}
#   Stage 1.5   relabel(best ckpt)+export: -> data/sft_corpus/tau2_retail{,_all}/{policy,base}
#   Stage 2     SFT (fresh base)         : {actions_only, thoughts_policy, thoughts_base,
#                                           expert_thoughts} x {A, B}
#   Stage 3     env-RL                   : guarded — needs .venv-tau2 + ./tau2-bench + the
#                                          rl_eval->tau2-index map (skipped w/ a note if absent)
#
# Split A stage-1 runs are ALREADY in flight (untagged); this script DETECTS their
# step_best rather than relaunching, then tags everything downstream.
#
# Usage:  CUDA_VISIBLE_DEVICES=0 nohup ./scripts/run_retail_matrix.sh > /tmp/aprm/matrix/run.log 2>&1 &
# Watch:  tail -f /tmp/aprm/matrix/orchestrator.log
set -uo pipefail                      # NOT -e: we log+continue/skip on per-step failure
cd "$(dirname "$0")/.."

export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/usr/local/bin:$PATH"
export https_proxy="${https_proxy:-http://fwdproxy:8080}"
export http_proxy="${http_proxy:-http://fwdproxy:8080}"
export HF_TOKEN="${HF_TOKEN:-$(cat "$HOME/models/token" 2>/dev/null || true)}"
export HF_HOME="${HF_HOME:-/data/users/$USER/models/hf_cache}"

MODEL=hf_qwen3_4b_instruct
CKROOT=checkpoints_lora/act_prm_tau2_retail
CKROOT_ALL=checkpoints_lora/act_prm_tau2_retail_all
LOGROOT=logs/act_prm_tau2_retail
LOGROOT_ALL=logs/act_prm_tau2_retail_all
MDIR=/tmp/aprm/matrix; mkdir -p "$MDIR"
CORPUS=data/sft_corpus

# stderr (+file) so $(stage1 ...) / $(export_corpus ...) capture ONLY the path they echo.
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$MDIR/orchestrator.log" >&2; }

# Block until no training process holds the GPU (so steps run one at a time).
wait_gpu_free(){
  while pgrep -f '[m]ain_pytorch.py' >/dev/null 2>&1; do sleep 60; done
  sleep 10
}

# Newest matching dir (glob expanded), else empty.
newest(){ ls -dt $1 2>/dev/null | head -1; }

# --- per-split params: envcfg | ck-root | log-root | split-tag ------------------
# A = heldout (configs/environments/act_prm/tau2_retail.yaml)
# B = all     (configs/environments/act_prm/tau2_retail_all.yaml)
splits=(A B)
env_A=act_prm/tau2_retail;      env_B=act_prm/tau2_retail_all
ck_A=$CKROOT;                   ck_B=$CKROOT_ALL
tag_A=heldout;                  tag_B=all

# swb flag per scorer
swb_policy="--no-score_with_base"; swbstr_policy="swb=0"
swb_base="--score_with_base";      swbstr_base="swb=1"

# ============================ STAGE 1 : EM thought-gen =========================
# Split A is already running untagged; only launch B here. For A we just locate the
# existing step_best (uncapped: prefer the dir WITHOUT omc=).
stage1(){
  local S=$1 scorer=$2
  local envc ckr stag swb swbstr
  eval "envc=\$env_$S; ckr=\$ck_$S; stag=\$tag_$S"
  eval "swb=\$swb_$scorer; swbstr=\$swbstr_$scorer"
  local tag="retail_s1_${scorer}_${stag}"

  # Already have a step_best for this cell? (tagged OR pre-existing untagged for A)
  local best
  best=$(newest "$ckr/$MODEL/${tag}-*/step_best")
  [ -z "$best" ] && best=$(newest "$ckr/$MODEL/*${swbstr}-*/step_best")
  if [ -n "$best" ] && [ -f "$best/adapter_model.safetensors" ]; then
    log "STAGE1 $S/$scorer: reuse $best"; echo "$best"; return 0
  fi

  log "STAGE1 $S/$scorer: launching EM ($envc $swb, tag=$tag)"
  wait_gpu_free
  # Re-check after waiting: the chained base run (or a prior invocation) may have
  # produced step_best while we waited — reuse it instead of launching a duplicate.
  best=$(newest "$ckr/$MODEL/${tag}-*/step_best")
  [ -z "$best" ] && best=$(newest "$ckr/$MODEL/*${swbstr}-*/step_best")
  if [ -n "$best" ] && [ -f "$best/adapter_model.safetensors" ]; then
    log "STAGE1 $S/$scorer: reuse (post-wait) $best"; echo "$best"; return 0
  fi
  ./scripts/train.sh --env_config "$envc" --generator_config act_prm --trainer_config pg \
    --model_config $MODEL --lora_config r8_a16_linear --replay_buffer_config default \
    $swb --group_size 4 --batch_size 4 --num_batches 25 --eval_every 5 --no_initial_eval \
    --length_penalty 0.15 --save_generations --run_tag "$tag" --verbose \
    > "$MDIR/s1_${scorer}_${stag}.log" 2>&1
  best=$(newest "$ckr/$MODEL/${tag}-*/step_best")
  [ -n "$best" ] && [ -f "$best/adapter_model.safetensors" ] \
    && { log "STAGE1 $S/$scorer: done -> $best"; echo "$best"; } \
    || log "STAGE1 $S/$scorer: FAILED (no step_best; see $MDIR/s1_${scorer}_${stag}.log)"
}

# ================= STAGE 1.5 : relabel with best ckpt + export corpus ==========
export_corpus(){
  local S=$1 scorer=$2 best=$3
  local envc stag out
  eval "envc=\$env_$S; stag=\$tag_$S"
  local envname=${envc##*/}
  out="$CORPUS/${envname}/${scorer}"
  if [ -f "$out/train.json" ]; then log "EXPORT $S/$scorer: reuse $out"; echo "$out"; return 0; fi
  [ -z "$best" ] && { log "EXPORT $S/$scorer: no ckpt, skip"; return 1; }

  local tag="retail_s1relabel_${scorer}_${stag}"
  local logroot=$LOGROOT; [ "$S" = B ] && logroot=$LOGROOT_ALL

  log "RELABEL $S/$scorer: best-of-G thoughts from $best"
  wait_gpu_free
  ./scripts/train.sh --env_config "$envc" --generator_config act_prm --trainer_config pg \
    --model_config $MODEL --lora_config r8_a16_linear --replay_buffer_config default \
    $([ "$scorer" = base ] && echo --score_with_base || echo --no-score_with_base) \
    --no_train --resume_from "$best" --advantage_mode best --group_size 4 --batch_size 4 \
    --num_batches 40 --no_initial_eval --length_penalty 0.15 --save_generations \
    --run_tag "$tag" --verbose > "$MDIR/s1relabel_${scorer}_${stag}.log" 2>&1

  local gen; gen=$(newest "$logroot/$MODEL/${tag}-*/generations.jsonl")
  [ -z "$gen" ] && { log "EXPORT $S/$scorer: no generations.jsonl, skip"; return 1; }
  local srcpools; eval "srcpools=data/${envname}"    # env dataset_path (source state)
  log "EXPORT $S/$scorer: $gen -> $out"
  uv run python scripts/export_sft_corpus.py --generations "$gen" \
    --source-pools "$srcpools" --out "$out" >> "$MDIR/s1relabel_${scorer}_${stag}.log" 2>&1 \
    && { log "EXPORT $S/$scorer: OK"; echo "$out"; } \
    || log "EXPORT $S/$scorer: export FAILED (see log)"
}

# ============================ STAGE 2 : SFT (fresh base) =======================
stage2(){
  local S=$1 variant=$2; shift 2
  local envc stag; eval "envc=\$env_$S; stag=\$tag_$S"
  local tag="retail_s2_${variant}_${stag}"
  local ckr; eval "ckr=\$ck_$S"
  # heldout has an eval set -> early-stop on eval_action_ppl; 'all' has none.
  local best_metric=(); [ "$S" = A ] && best_metric=(--best_metric eval_action_ppl)
  if [ -n "$(newest "$ckr/$MODEL/${tag}-*/step_best")" ]; then
    log "STAGE2 $S/$variant: already done, skip"; return 0; fi
  log "STAGE2 $S/$variant: SFT (tag=$tag) $*"
  wait_gpu_free
  ./scripts/train_sft.sh "$envc" "$variant" --run_tag "$tag" "${best_metric[@]}" "$@" \
    > "$MDIR/s2_${variant}_${stag}.log" 2>&1 \
    && log "STAGE2 $S/$variant: done" \
    || log "STAGE2 $S/$variant: FAILED (see $MDIR/s2_${variant}_${stag}.log)"
}

# ================================== DRIVE =====================================
log "=== retail matrix start (GPU ${CUDA_VISIBLE_DEVICES:-?}) ==="
# Let the in-flight untagged split-A stage-1 runs (policy + chained base) finish first,
# so reuse-detection picks their (newest) step_best instead of a stale earlier run.
log "waiting for any in-flight main_pytorch runs to finish before detection..."
wait_gpu_free
declare -A BEST CORP
for S in "${splits[@]}"; do
  for scorer in policy base; do
    BEST[$S:$scorer]=$(stage1 "$S" "$scorer")
    CORP[$S:$scorer]=$(export_corpus "$S" "$scorer" "${BEST[$S:$scorer]}")
  done
done

for S in "${splits[@]}"; do
  stage2 "$S" actions_only
  stage2 "$S" expert_thoughts
  [ -n "${CORP[$S:policy]:-}" ] && stage2 "$S" thoughts_policy --dataset_path "${CORP[$S:policy]}"
  [ -n "${CORP[$S:base]:-}" ]   && stage2 "$S" thoughts_base   --dataset_path "${CORP[$S:base]}"
done

# ============================ STAGE 3 : env-RL (guarded) ======================
if [ -x .venv-tau2/bin/python ] && [ -d tau2-bench ]; then
  log "STAGE3: tau2-gym present — RL-from-SFT is available; wire rl_eval->tau2 index map first."
  log "  e.g. ./scripts/train_rl_from_sft.sh retail <stage2_step_best> --run_tag retail_s3_<variant>_<split>"
else
  log "STAGE3: SKIPPED — needs .venv-tau2 + ./tau2-bench clone (github-blocked here) + rl_eval->tau2-index map."
fi
log "=== retail matrix: stages 1-2 pass complete ==="
