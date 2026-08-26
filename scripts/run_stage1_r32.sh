#!/usr/bin/env bash
# Stage-1 EM at rank 32 (r32_a32_linear) across all domains, regenerating the Act-PRM SFT
# corpora from the resulting checkpoints.
#
# WHY: every Stage-1 EM run to date used lr=4e-5 with r8_a16_linear and the adapter is a
# measured no-op -- max|B@A| of 6.7e-07 (retail), 4.1e-07 (finance), 2.8e-06 (insurance)
# against base weights of order 1e-2. So the "EM-trained" generator has always been the base
# model, and Act-PRM's measured benefit comes from the E-step (best-of-G by length-penalised
# likelihood) alone. MZ's hypothesis: keep lr=4e-5 but raise the LoRA rank to 32.
#
# BUG FOUND 08-25: the abort check below sits AFTER the EM block, so it only fires once the
# full 25-batch run finishes -- ~4h, not the ~50 min intended. The --save_every checkpoint IS
# written on schedule, so the check was run manually against it instead. To make it genuinely
# early the check must poll for the step_last checkpoint while the run is in flight.
#
# RESULT 08-25: ABORTED. r32_a32 @ lr=4e-5 gave max|B@A| = 8.6e-07 at batch 5 on airline --
# the same order as every r8 run (retail 6.7e-07, finance 4.1e-07, insurance 2.8e-06) and two
# orders below the ~1e-4 that would count as marginal movement. Rank does not escape the
# zero-init cold start; the learning rate is the only lever.
#
# ABORT DISABLED (MZ, 08-25): ABORT_AT now defaults to 0, so the sweep runs to completion
# regardless of what the adapter measurement says. The measurement is still worth taking --
# it is what told us r8/em, r32/em and r32/action_probs are all no-ops at lr 4e-5 -- but it
# no longer gates anything. Set ABORT_AT=5 to re-enable.
#
# (historical) EARLY ABORT. The full sweep is ~35h for one scorer and ~70h for both, and it is worthless
# if rank does not escape the zero-init cold start (lora_B starts at 0, so dL/dA ~ B^T ~ 0).
# So the FIRST domain writes a checkpoint at batch ABORT_AT (--save_every) and the adapter is
# measured there. If it is still a no-op, the sweep stops and nothing further is spent. This
# costs ~50 min of a run we wanted anyway rather than a separate probe.
#
# CAVEAT on the config: r8/a16 -> r32/a32 changes rank (8->32) AND scaling alpha/r
# (2.0->1.0) together. Scaling moves the WRONG way, so a positive result is strong evidence
# for rank (it overcame 2x damping) while a null is ambiguous. r32/a64 would isolate rank.
#
# ORDER: cheapest domain first (airline 21 train traj, then retail 49, finance 116,
# insurance 180) so a full end-to-end cycle -- EM, relabel, corpus -- lands as early as
# possible and any pipeline bug surfaces on the cheap domain.
#
# Usage: ./scripts/run_stage1_r32.sh              (cron drives it via sweep_guard.sh)
#        SCORERS="policy base" ./scripts/run_stage1_r32.sh
#        ABORT_AT=0 ./scripts/run_stage1_r32.sh   (disable the early-abort check)
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/usr/local/bin:$PATH"
export https_proxy="${https_proxy:-http://fwdproxy:8080}"
export http_proxy="${http_proxy:-http://fwdproxy:8080}"
export HF_TOKEN="${HF_TOKEN:-$(cat "$HOME/models/token" 2>/dev/null || true)}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
MODEL="${MODEL_CFG:-hf_qwen3_4b_instruct}"; export MODEL_CFG="$MODEL"
MDIR="${MDIR:-/tmp/aprm/stage1_r32}"; mkdir -p "$MDIR"
_fail=0
LORA="${LORA:-r32_a32_linear}"
LR="${LR:-4e-5}"
EM_NB="${EM_NB:-25}"
ABORT_AT="${ABORT_AT:-0}"        # measure the adapter after this many batches; 0 disables
SCORERS="${SCORERS:-policy}"     # add "base" for the thoughts_base arm (doubles the cost)
TAGSFX="${TAGSFX:-_ap32}"
# Reward/advantage formulation. Default reproduces the TINKER reference as actually run
# (configs/generator/aprm_qwen3_ap.yaml -> reward_method "action_probs"), which the
# documented commands in act_prm_sft_rl.py / act_prm_joint.py use:
#   * advantage = RAW length-normalised p(x|s,z), no group normalisation
#   * length handled by normalising the logprob sum by token count (already in
#     `likelihoods`), NOT by our extra subtractive penalty -> LENGTH_PENALTY=0
#   * lora_rank 32 (Tinker's trainer configs all set lora_rank: 32)
ADV_MODE="${ADV_MODE:-action_probs}"
LENGTH_PENALTY="${LENGTH_PENALTY:-0}"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$MDIR/stage1_r32.log"; }
wait_gpu_free(){ while pgrep -f '[m]ain_pytorch.py' >/dev/null 2>&1; do sleep 60; done; sleep 10; }
newest(){ ls -dt $1 2>/dev/null | head -1; }
n_rows(){ uv run --no-project python -c "import json;print(len(json.load(open('$1'))))" 2>/dev/null || echo 0; }

# domain : env_config : source pool : relabel batches (ceil(train_traj/4)) : corpus dir
DOMAINS=(
  "airline:act_prm/tau2_airline:data/tau2_airline:6:data/sft_corpus/tau2_airline"
  "retail:act_prm/tau2_retail:data/tau2_retail:13:data/sft_corpus/tau2_retail"
  "finance:act_prm/snorkel_finance_split:data/snorkel_finance_split_v3:29:data/sft_corpus/snorkel_finance_split"
  "insurance:act_prm/snorkel_insurance:data/snorkel_insurance_split:45:data/sft_corpus/snorkel_insurance"
)

log "=== Stage-1 @ $LORA lr=$LR adv=$ADV_MODE lp=$LENGTH_PENALTY, scorers='$SCORERS', abort-check at batch $ABORT_AT ==="
for spec in "${DOMAINS[@]}"; do
  IFS=":" read -r dom env pool rnb corpus <<< "$spec"
  ENVDIR="${env//\//_}"
  for scorer in $SCORERS; do
    SWB=--no-score_with_base; SWBV=0
    [ "$scorer" = base ] && { SWB=--score_with_base; SWBV=1; }
    OUT="${corpus}/${scorer}${TAGSFX}"
    if [ -s "$OUT/train.json" ] && [ "$(n_rows "$OUT/train.json")" -gt 0 ]; then
      log "$dom/$scorer: corpus exists ($(n_rows "$OUT/train.json") train), skip"; continue
    fi

    # ---- Stage 1a: EM training at rank 32
    EMTAG="${dom}_s1em_${scorer}${TAGSFX}"
    if [ ! -f "$MDIR/${EMTAG}.done" ]; then
      log "EM $EMTAG (lora=$LORA lr=$LR nb=$EM_NB)"
      wait_gpu_free
      SAVE=(); [ "$ABORT_AT" != 0 ] && SAVE=(--save_every "$ABORT_AT")
      ./scripts/train.sh --env_config "$env" --generator_config act_prm --trainer_config pg \
          --model_config "$MODEL" --lora_config "$LORA" --replay_buffer_config default \
          $SWB --run_tag "$EMTAG" --group_size 4 --batch_size 4 --num_batches "$EM_NB" \
          --learning_rate "$LR" --length_penalty "$LENGTH_PENALTY" \
          --advantage_mode "$ADV_MODE" --save_generations \
          --no_initial_eval --eval_every "$EM_NB" --gradient_checkpointing "${SAVE[@]}" --verbose \
          > "$MDIR/${EMTAG}.log" 2>&1 \
        || { _fail=$((_fail+1)); log "$EMTAG: FAILED"; continue; }
      touch "$MDIR/${EMTAG}.done"; log "$EMTAG: done"
    fi

    # ---- EARLY ABORT: did the adapter actually move?
    if [ "$ABORT_AT" != 0 ] && [ ! -f "$MDIR/MOVED" ]; then
      CK=$(newest "checkpoints_lora/$ENVDIR/$MODEL/${EMTAG}-*/step_*")
      log "adapter check on $CK"
      uv run --no-project python scripts/report_lora_movement.py --glob "$CK" \
        2>&1 | tee -a "$MDIR/stage1_r32.log" | grep -q "NO-OP" && {
          log "ABORT: $LORA @ lr=$LR is STILL a no-op -- rank does not escape the zero-init"
          log "       cold start. Stopping the sweep; nothing further spent. The remaining"
          log "       lever is the learning rate (3e-3), not the rank."
          touch "$MDIR/ABORTED"; exit 0; }
      touch "$MDIR/MOVED"; log "adapter MOVED -- continuing the full sweep"
    fi

    # ---- Stage 1b: relabel from step_best, then export the corpus
    CK=$(newest "checkpoints_lora/$ENVDIR/$MODEL/${EMTAG}-*/step_best")
    [ -z "$CK" ] && { _fail=$((_fail+1)); log "$dom/$scorer: no step_best"; continue; }
    RTAG="${dom}_s1relabel_${scorer}${TAGSFX}"
    log "RELABEL $RTAG (nb=$rnb -> $((rnb*4)) trajectories) from $CK"
    wait_gpu_free
    ./scripts/train.sh --env_config "$env" --generator_config act_prm --trainer_config pg \
        --model_config "$MODEL" --lora_config "$LORA" --replay_buffer_config default \
        $SWB --no_train --resume_from "$CK" --advantage_mode best \
        --group_size 4 --batch_size 4 --num_batches "$rnb" --no_initial_eval \
        --length_penalty "$LENGTH_PENALTY" --save_generations --run_tag "$RTAG" --verbose \
        > "$MDIR/${RTAG}.log" 2>&1 \
      || { _fail=$((_fail+1)); log "$RTAG: FAILED"; continue; }
    GEN=$(newest "logs/$ENVDIR/$MODEL/${RTAG}-*/generations.jsonl")
    [ -z "$GEN" ] && { _fail=$((_fail+1)); log "$RTAG: no generations.jsonl"; continue; }
    uv run --no-project python scripts/export_sft_corpus.py \
        --generations "$GEN" --source-pools "$pool" --out "$OUT" \
        >> "$MDIR/${RTAG}.log" 2>&1 \
      || { _fail=$((_fail+1)); log "$RTAG: export FAILED"; continue; }
    log "$dom/$scorer: corpus -> $OUT ($(n_rows "$OUT/train.json") train / $(n_rows "$OUT/eval.json") eval)"
  done
done
if [ "${_fail:-0}" -eq 0 ]; then touch "$MDIR/ALLDONE"; log "=== stage1 r32 sweep complete ==="
else log "=== stage1 r32 sweep INCOMPLETE: $_fail failure(s) ==="; fi
