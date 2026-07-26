#!/usr/bin/env bash
# Stage-1 Act-PRM EM thought-generation for MODEL_CFG, BOTH scorers:
#   policy  (--no-score_with_base, swb=0)  — current LoRA scores p(x|s,z)
#   base    (--score_with_base,    swb=1)  — frozen base model scores p(x|s,z)
# Resumable: a scorer is skipped if its step_best already exists. GPU-serialized
# (wait_gpu_free before each). Model is parametrized via MODEL_CFG (default 4B); for an
# 8B model the H100 won't fit the uncapped E-step, so --obs_max_chars 3000 is appended.
#
# Usage:  CUDA_VISIBLE_DEVICES=0 MODEL_CFG=hf_qwen3_8b ./scripts/run_em.sh act_prm/tau2_retail
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/usr/local/bin:$PATH"
export https_proxy="${https_proxy:-http://fwdproxy:8080}"
export http_proxy="${http_proxy:-http://fwdproxy:8080}"
export HF_TOKEN="${HF_TOKEN:-$(cat "$HOME/models/token" 2>/dev/null || true)}"

ENVCFG="${1:-act_prm/tau2_retail}"; ENVNAME="${ENVCFG##*/}"; DOM="${ENVNAME#tau2_}"
# Model is parametrized: MODEL_CFG selects both --model_config AND the <MODEL> path dir.
MODEL="${MODEL_CFG:-hf_qwen3_4b_instruct}"; export MODEL_CFG="$MODEL"
CKROOT="checkpoints_lora/${ENVCFG//\//_}/$MODEL"
MDIR="/tmp/aprm/em_${ENVNAME}_${MODEL}"; mkdir -p "$MDIR"

log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$MDIR/em.log"; }
wait_gpu_free(){ while pgrep -f '[m]ain_pytorch.py' >/dev/null 2>&1; do sleep 60; done; sleep 10; }
newest(){ ls -dt $1 2>/dev/null | head -1; }

# VRAM: an 8B uncapped E-step won't fit the H100 (~95 GiB) — cap the observation chars.
OBSCAP=()
case "$MODEL" in *8b*|*8B*) OBSCAP=(--obs_max_chars 3000);; esac

run_em(){  # $1=scorer(policy|base)  $2=swb(0|1)
  local scorer=$1 swb=$2
  local best; best=$(newest "$CKROOT/*swb=${swb}*/step_best")
  if [ -n "$best" ]; then log "$scorer (swb=$swb): step_best exists ($best), skip"; return 0; fi
  local sflag; sflag=$([ "$scorer" = base ] && echo --score_with_base || echo --no-score_with_base)
  local tag="${DOM}_s1em_${scorer}"
  local cmd=(./scripts/train.sh --env_config "$ENVCFG" --generator_config act_prm --trainer_config pg
    --model_config "$MODEL" --lora_config r8_a16_linear --replay_buffer_config default $sflag
    --group_size 4 --batch_size 4 --num_batches 25 --eval_every 5 --no_initial_eval
    --length_penalty 0.15 --save_generations "${OBSCAP[@]}"
    --run_tag "$tag" --verbose)
  # EM_DRY_RUN=1 -> print the resolved command (per scorer) and do NOT touch the GPU.
  if [ "${EM_DRY_RUN:-0}" = 1 ]; then printf '%s ' "${cmd[@]}"; echo; return 0; fi
  log "EM $scorer (swb=$swb) model=$MODEL ${OBSCAP[*]} -> tag=$tag"
  wait_gpu_free
  # Re-check after waiting (a concurrent/chained run may have produced it).
  best=$(newest "$CKROOT/*swb=${swb}*/step_best")
  if [ -n "$best" ]; then log "$scorer (swb=$swb): step_best appeared ($best), skip"; return 0; fi
  "${cmd[@]}" > "$MDIR/em_${scorer}.log" 2>&1 \
    && log "$scorer: done" || log "$scorer: FAILED (see $MDIR/em_${scorer}.log)"
}

log "=== Stage-1 EM for $ENVCFG (model=$MODEL) ==="
run_em policy 0
run_em base   1
log "=== EM done for $ENVCFG (model=$MODEL) ==="
