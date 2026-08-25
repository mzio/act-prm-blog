#!/usr/bin/env bash
# Does a higher-rank LoRA make the Stage-1 EM adapter actually MOVE at lr=4e-5?
#
# Background. Every Stage-1 EM run in every domain has used lr=4e-5 with r8_a16_linear, and
# the resulting adapter is a numerical no-op: on insurance, max|B@A| = 2.8e-06 against base
# weights of order 1e-2. The EM reward curve is consequently flat in all domains.
#
# The hypothesis under test (MZ, 08-25): keep lr=4e-5 but raise the LoRA rank to 32.
#
# My prior is that rank alone will NOT fix it, because the failure looks like a zero-init
# cold start -- lora_B starts at 0, so dL/dA is proportional to B^T and stays ~0, while B
# only grows via dL/dB proportional to A. More rank adds more zero-init B columns rather
# than escaping that loop. The counter-argument is real though: more random A directions
# means a larger dL/dB, so ||B@A|| could grow faster with rank. Cheap to measure, so measure.
#
# NOTE on alpha: configs/lora/r32_a64_linear.yaml uses alpha=64 = 2r. PEFT scales by alpha/r,
# and the repo's other configs both hold that ratio at 2.0; r=32 with alpha=16 would give
# scaling 0.5, a 4x SMALLER update than the r8 baseline, inverting the experiment.
#
# DESIGN: short probes (default 5 batches, ~50 min each) then measure the adapter, rather
# than a 4.4h full EM run per arm. The question is "does it move", which 5 batches answers.
# Arms, all on insurance (its pool is already built and its no-op is measured):
#   r8/a16  @ 4e-5  -- reproduce the known no-op as a control        (scaling 2.0)
#   r32/a64 @ 4e-5  -- the hypothesis, scaling held at the baseline   (scaling 2.0)
#   r32/a32 @ 4e-5  -- same rank, HALF the scaling                    (scaling 1.0)
#   r8/a16  @ 3e-3  -- the LR fix, reference for what "moving" means  (scaling 2.0)
# The two r32 arms separate rank from scaling, which are otherwise confounded: comparing
# r8/a16 to r32/a64 varies rank at fixed scaling, and r32/a64 to r32/a32 varies scaling at
# fixed rank.
#
# Usage: ./scripts/probe_stage1_lora_rank.sh
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/usr/local/bin:$PATH"
export https_proxy="${https_proxy:-http://fwdproxy:8080}"
export http_proxy="${http_proxy:-http://fwdproxy:8080}"
export HF_TOKEN="${HF_TOKEN:-$(cat "$HOME/models/token" 2>/dev/null || true)}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
MODEL="${MODEL_CFG:-hf_qwen3_4b_instruct}"; export MODEL_CFG="$MODEL"
MDIR="${MDIR:-/tmp/aprm/lorarank}"; mkdir -p "$MDIR"
_fail=0
NB="${NB:-5}"
ENVCFG=act_prm/snorkel_insurance
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$MDIR/lorarank.log"; }
wait_gpu_free(){ while pgrep -f '[m]ain_pytorch.py' >/dev/null 2>&1; do sleep 60; done; sleep 10; }

# name : lora_config : lr        (scaling = alpha/r shown for reference)
ARMS=("r8_lr4e5:r8_a16_linear:4e-5"      # scaling 2.0 -- control, the known no-op
      "r32a64_lr4e5:r32_a64_linear:4e-5" # scaling 2.0 -- rank varies, scaling held
      "r32a32_lr4e5:r32_a32_linear:4e-5" # scaling 1.0 -- scaling varies, rank held
      "r8_lr3e3:r8_a16_linear:3e-3")     # scaling 2.0 -- the LR fix, reference for "moved"

log "=== Stage-1 LoRA-rank probe: $NB batches per arm, insurance ==="
for spec in "${ARMS[@]}"; do
  IFS=":" read -r name lc lr <<< "$spec"
  TAG="s1probe_${name}"
  [ -f "$MDIR/${TAG}.done" ] && { log "$TAG: done, skip"; continue; }
  log "PROBE $TAG (lora=$lc lr=$lr nb=$NB)"
  wait_gpu_free
  ./scripts/train.sh --env_config "$ENVCFG" --generator_config act_prm --trainer_config pg \
      --model_config "$MODEL" --lora_config "$lc" --replay_buffer_config default \
      --no-score_with_base --run_tag "$TAG" --group_size 4 --batch_size 4 \
      --num_batches "$NB" --length_penalty 0.15 --learning_rate "$lr" \
      --no_initial_eval --eval_every "$NB" --gradient_checkpointing --verbose \
      > "$MDIR/${TAG}.log" 2>&1 \
    && { touch "$MDIR/${TAG}.done"; log "$TAG: done"; } \
    || { _fail=$((_fail+1)); log "$TAG: FAILED (see $MDIR/${TAG}.log)"; }
done

log "--- measuring adapter movement (the actual question) ---"
uv run --no-project python scripts/report_lora_movement.py 2>&1 | tee -a "$MDIR/lorarank.log"
if [ "${_fail:-0}" -eq 0 ]; then touch "$MDIR/ALLDONE"; log "=== rank probe complete ==="
else log "=== rank probe INCOMPLETE: $_fail failure(s) ==="; fi
