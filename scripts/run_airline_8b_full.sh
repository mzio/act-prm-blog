#!/usr/bin/env bash
# FULL Qwen3-8B airline pipeline, queued AFTER the 4B sweep (waits for GPUs to free):
#   Stage 1    EM thought-gen (policy+base)        -> checkpoints .../hf_qwen3_8b/airline_s1_8b_{policy,base}
#   Stage 1.5  relabel(best+last)+export           -> data/sft_corpus/tau2_airline_8b/{policy,base,policy_last,base_last}
#   Stage 2    SFT (fresh 8B) x12                   -> {6 variants} x {hide,full}, early-stop eval_action_ppl
# Offline, uncapped, --gradient_checkpointing. Resumable (skips existing step_best/corpora).
# Mirrors scripts/run_airline_sweep.sh (4B) but for MODEL=hf_qwen3_8b with 8b-suffixed
# corpora + airline_8b_* run_tags so nothing collides with the 4B artifacts.
#
# Usage:  nohup ./scripts/run_airline_8b_full.sh > /tmp/aprm/airline_8b/run.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/usr/local/bin:$PATH"
export HF_HOME="${HF_HOME:-/data/users/$USER/models/hf_cache}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1 UV_FROZEN=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

MODEL=hf_qwen3_8b; MSHORT=8b
ENVC=act_prm/tau2_airline; ENVNAME=tau2_airline
CKR=checkpoints_lora/act_prm_tau2_airline/$MODEL
LOGROOT=logs/act_prm_tau2_airline/$MODEL
CORPUS=data/sft_corpus/${ENVNAME}_${MSHORT}
MDIR=/tmp/aprm/airline_8b; mkdir -p "$MDIR"
G0=0; G1=1
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$MDIR/orchestrator.log"; }
newest(){ ls -dt $1 2>/dev/null | head -1; }
wait_gpu_free(){ while pgrep -f '[m]ain_pytorch.py' >/dev/null 2>&1; do sleep 60; done; sleep 10; }

# ---- Stage 1 EM (one scorer on one GPU) -------------------------------------------
em(){  # scorer gpu
  local scorer=$1 gpu=$2 swb tag="airline_s1_8b_$1"
  [ "$scorer" = base ] && swb=--score_with_base || swb=--no-score_with_base
  [ -n "$(newest "$CKR/${tag}-*/step_best")" ] && { log "EM $scorer: step_best exists, skip"; return 0; }
  log "EM $scorer (GPU $gpu)"
  CUDA_VISIBLE_DEVICES="$gpu" ./scripts/train.sh --env_config "$ENVC" --generator_config act_prm \
    --trainer_config pg --model_config $MODEL --lora_config r8_a16_linear --replay_buffer_config default \
    $swb --group_size 4 --batch_size 4 --num_batches 25 --eval_every 5 --no_initial_eval \
    --length_penalty 0.15 --gradient_checkpointing --save_generations --run_tag "$tag" --verbose \
    > "$MDIR/s1_${scorer}.log" 2>&1 && log "EM $scorer: done" || log "EM $scorer: FAILED (see $MDIR/s1_${scorer}.log)"
}

# ---- Stage 1.5 relabel(ckpt)+export -----------------------------------------------
relabel_export(){  # scorer ckpt tag out gpu
  local scorer=$1 ckpt=$2 tag=$3 out=$4 gpu=$5 swb
  [ -f "$out/train.json" ] && { log "export $out: exists, skip"; return 0; }
  { [ -z "$ckpt" ] || [ ! -f "$ckpt/adapter_model.safetensors" ]; } && { log "relabel $tag: no ckpt, skip"; return 1; }
  [ "$scorer" = base ] && swb=--score_with_base || swb=--no-score_with_base
  local gen; gen=$(newest "$LOGROOT/${tag}-*/generations.jsonl")
  if [ -z "$gen" ]; then
    log "relabel $tag (GPU $gpu) from $ckpt"
    CUDA_VISIBLE_DEVICES="$gpu" ./scripts/train.sh --env_config "$ENVC" --generator_config act_prm \
      --trainer_config pg --model_config $MODEL --lora_config r8_a16_linear --replay_buffer_config default \
      $swb --no_train --resume_from "$ckpt" --advantage_mode best --group_size 4 --batch_size 4 \
      --num_batches 40 --no_initial_eval --length_penalty 0.15 --save_generations \
      --run_tag "$tag" --verbose > "$MDIR/${tag}.log" 2>&1
    gen=$(newest "$LOGROOT/${tag}-*/generations.jsonl")
  fi
  [ -z "$gen" ] && { log "relabel $tag: no generations (see $MDIR/${tag}.log)"; return 1; }
  uv run python scripts/export_sft_corpus.py --generations "$gen" --source-pools "data/$ENVNAME" \
    --out "$out" >> "$MDIR/${tag}.log" 2>&1 && log "export $out: OK" || log "export $out: FAILED"
}

# ---- Stage 2 SFT ------------------------------------------------------------------
sft(){  # variant regime tag gpu [extra...]
  local variant=$1 regime=$2 tag=$3 gpu=$4; shift 4
  [ -n "$(newest "$CKR/${tag}-*/step_best")" ] && { log "SFT $tag: done, skip"; return 0; }
  local fc=(); [ "$regime" = full ] && fc=(SFT_FULLCTX=1)
  log "SFT $tag (GPU $gpu, $regime) $*"
  CUDA_VISIBLE_DEVICES="$gpu" env "${fc[@]}" ./scripts/train_sft.sh "$ENVC" "$variant" \
    --model_config $MODEL --run_tag "$tag" --best_metric eval_action_ppl --gradient_checkpointing "$@" \
    > "$MDIR/${tag}.log" 2>&1 && log "SFT $tag: done" || log "SFT $tag: FAILED (see $MDIR/${tag}.log)"
}
run_stream(){ local gpu=$1; shift; for spec in "$@"; do IFS='|' read -r v r t e <<< "$spec"; sft "$v" "$r" "$t" "$gpu" $e; done; }

# ================================== DRIVE =====================================
log "=== 8B FULL pipeline: waiting for GPUs (4B sweep to finish) ==="
wait_gpu_free
# Stage 1 EM (policy GPU0, base GPU1)
em policy $G0 & p0=$!; sleep 60; em base $G1 & p1=$!; wait $p0; wait $p1
POL_BEST=$(newest "$CKR/airline_s1_8b_policy-*/step_best"); BAS_BEST=$(newest "$CKR/airline_s1_8b_base-*/step_best")
POL_LAST=$(newest "$CKR/airline_s1_8b_policy-*/step_last"); BAS_LAST=$(newest "$CKR/airline_s1_8b_base-*/step_last")
log "8B ckpts: best(pol=$POL_BEST bas=$BAS_BEST) last(pol=$POL_LAST bas=$BAS_LAST)"
# Stage 1.5 relabel best (parallel) then last (parallel)
( relabel_export policy "$POL_BEST" airline_s1relabel_8b_policy "$CORPUS/policy" $G0 ) & p0=$!
( relabel_export base   "$BAS_BEST" airline_s1relabel_8b_base   "$CORPUS/base"   $G1 ) & p1=$!; wait $p0; wait $p1
( relabel_export policy "$POL_LAST" airline_s1relabel_8b_policy_last "$CORPUS/policy_last" $G0 ) & p0=$!
( relabel_export base   "$BAS_LAST" airline_s1relabel_8b_base_last   "$CORPUS/base_last"   $G1 ) & p1=$!; wait $p0; wait $p1
for c in policy base policy_last base_last; do
  n=$(python3 -c "import json;print(len(json.load(open('$CORPUS/$c/train.json'))))" 2>/dev/null||echo 0)
  log "corpus 8b/$c: $n train"; [ "${n:-0}" -lt 1 ] && log "WARN: 8b/$c EMPTY"
done
# Stage 2 SFT (6/GPU, balanced hide/full)
JOBS_G0=(
  "actions_only|hide|airline_8b_s2_actions_only_heldout|"
  "thoughts_policy|hide|airline_8b_s2_thoughts_policy_heldout|--dataset_path $CORPUS/policy"
  "thoughts_policy|hide|airline_8b_s2_thoughts_policy_last_heldout|--dataset_path $CORPUS/policy_last"
  "expert_thoughts|full|airline_8b_s2_expert_thoughts_heldout_fullctx|"
  "thoughts_base|full|airline_8b_s2_thoughts_base_heldout_fullctx|--dataset_path $CORPUS/base"
  "thoughts_base|full|airline_8b_s2_thoughts_base_last_heldout_fullctx|--dataset_path $CORPUS/base_last"
)
JOBS_G1=(
  "expert_thoughts|hide|airline_8b_s2_expert_thoughts_heldout|"
  "thoughts_base|hide|airline_8b_s2_thoughts_base_heldout|--dataset_path $CORPUS/base"
  "thoughts_base|hide|airline_8b_s2_thoughts_base_last_heldout|--dataset_path $CORPUS/base_last"
  "actions_only|full|airline_8b_s2_actions_only_heldout_fullctx|"
  "thoughts_policy|full|airline_8b_s2_thoughts_policy_heldout_fullctx|--dataset_path $CORPUS/policy"
  "thoughts_policy|full|airline_8b_s2_thoughts_policy_last_heldout_fullctx|--dataset_path $CORPUS/policy_last"
)
log "=== 8B Stage 2: 12 SFT runs (6/GPU) ==="
run_stream $G0 "${JOBS_G0[@]}" & s0=$!
run_stream $G1 "${JOBS_G1[@]}" & s1=$!
wait $s0; wait $s1
log "=== 8B FULL pipeline complete ==="
