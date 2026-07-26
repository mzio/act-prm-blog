#!/usr/bin/env bash
# Airline Stage 1.5 (relabel+export, step_best AND step_last) + Stage 2 SFT matrix,
# OFFLINE, across BOTH GPUs. Resumable: every step is skipped if its output exists.
#
#   Stage 1     EM thought-gen        : DONE — reuse existing step_best + step_last
#   Stage 1.5   relabel(ckpt)+export  : {policy,base} x {best,last}
#                 -> data/sft_corpus/tau2_airline/{policy,base,policy_last,base_last}
#   Stage 2     SFT (fresh base) x12  : {actions_only, expert_thoughts,
#                 thoughts_policy(best), thoughts_base(best),
#                 thoughts_policy(last), thoughts_base(last)} x {hide-obs, full-context}
#                 fresh Qwen3-4B+LoRA, uncapped + --gradient_checkpointing,
#                 early-stop eval_action_ppl. Split across GPU0/GPU1 (serial per GPU).
#
# Usage:  nohup ./scripts/run_airline_sweep.sh > /tmp/aprm/airline_sweep/run.log 2>&1 &
# Watch:  tail -f /tmp/aprm/airline_sweep/orchestrator.log
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/usr/local/bin:$PATH"
export HF_HOME="${HF_HOME:-/data/users/$USER/models/hf_cache}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1 UV_FROZEN=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

MODEL=hf_qwen3_4b_instruct
ENVC=act_prm/tau2_airline; ENVNAME=tau2_airline
CKR=checkpoints_lora/act_prm_tau2_airline/$MODEL
LOGROOT=logs/act_prm_tau2_airline/$MODEL
CORPUS=data/sft_corpus/$ENVNAME
MDIR=/tmp/aprm/airline_sweep; mkdir -p "$MDIR"
G0=0; G1=1

log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$MDIR/orchestrator.log"; }
newest(){ ls -dt $1 2>/dev/null | head -1; }
wait_gpu_free(){ while pgrep -f '[m]ain_pytorch.py' >/dev/null 2>&1; do sleep 30; done; sleep 5; }

# ---- Stage 1.5 relabel one (scorer,ckpt) on a GPU, then export -> corpus dir -------
relabel_export(){  # scorer ckpt_path tag out_dir gpu   (runs in foreground)
  local scorer=$1 ckpt=$2 tag=$3 out=$4 gpu=$5 swb
  [ -f "$out/train.json" ] && { log "export $out: exists, skip"; return 0; }
  [ -z "$ckpt" ] || [ ! -f "$ckpt/adapter_model.safetensors" ] && { log "relabel $tag: no ckpt ($ckpt), skip"; return 1; }
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
  [ -z "$gen" ] && { log "relabel $tag: no generations.jsonl (see $MDIR/${tag}.log)"; return 1; }
  uv run python scripts/export_sft_corpus.py --generations "$gen" \
    --source-pools "data/$ENVNAME" --out "$out" >> "$MDIR/${tag}.log" 2>&1 \
    && log "export $out: OK ($(wc -l <"$out/train.json" 2>/dev/null) tasks)" || log "export $out: FAILED"
}

# ---- Stage 2: one SFT run (fresh base) --------------------------------------------
sft(){  # variant regime tag gpu [extra flags...]
  local variant=$1 regime=$2 tag=$3 gpu=$4; shift 4
  if [ -n "$(newest "$CKR/${tag}-*/step_best")" ]; then log "SFT $tag: done, skip"; return 0; fi
  local fc=(); [ "$regime" = full ] && fc=(SFT_FULLCTX=1)
  log "SFT $tag (GPU $gpu, $regime-obs) $*"
  CUDA_VISIBLE_DEVICES="$gpu" env "${fc[@]}" ./scripts/train_sft.sh "$ENVC" "$variant" \
    --run_tag "$tag" --best_metric eval_action_ppl --gradient_checkpointing "$@" \
    > "$MDIR/${tag}.log" 2>&1 && log "SFT $tag: done" || log "SFT $tag: FAILED (see $MDIR/${tag}.log)"
}

# run a list of "variant|regime|tag|extra" jobs serially on one GPU
run_stream(){
  local gpu=$1; shift
  for spec in "$@"; do
    IFS='|' read -r variant regime tag extra <<< "$spec"
    # shellcheck disable=SC2086
    sft "$variant" "$regime" "$tag" "$gpu" $extra
  done
}

# ================================== DRIVE =====================================
log "=== airline sweep start (GPUs $G0,$G1) ==="
POL_BEST=$(newest "$CKR/*swb=0*/step_best"); BAS_BEST=$(newest "$CKR/*swb=1*/step_best")
POL_LAST=$(newest "$CKR/*swb=0*/step_last"); BAS_LAST=$(newest "$CKR/*swb=1*/step_last")
log "ckpts: POL_BEST=$POL_BEST BAS_BEST=$BAS_BEST POL_LAST=$POL_LAST BAS_LAST=$BAS_LAST"

# --- Stage 1.5: step_best (in-flight relabels launched separately) then step_last ---
log "waiting for any in-flight runs..."; wait_gpu_free
# step_best: policy on GPU0, base on GPU1 (parallel)
( relabel_export policy "$POL_BEST" airline_s1relabel_policy_heldout "$CORPUS/policy" $G0 ) & p0=$!
( relabel_export base   "$BAS_BEST" airline_s1relabel_base_heldout   "$CORPUS/base"   $G1 ) & p1=$!
wait $p0; wait $p1
# step_last: policy on GPU0, base on GPU1 (parallel)
( relabel_export policy "$POL_LAST" airline_s1relabel_policy_last "$CORPUS/policy_last" $G0 ) & p0=$!
( relabel_export base   "$BAS_LAST" airline_s1relabel_base_last   "$CORPUS/base_last"   $G1 ) & p1=$!
wait $p0; wait $p1
# Sanity: corpora must have train examples (the no_train bug produced train:0)
for c in policy base policy_last base_last; do
  n=$(python3 -c "import json;print(len(json.load(open('$CORPUS/$c/train.json'))))" 2>/dev/null || echo 0)
  log "corpus $c: $n train tasks"; [ "${n:-0}" -lt 1 ] && log "WARN: corpus $c EMPTY (train:0) — check relabel!"
done

# --- Stage 2: 12 SFT runs (6 per GPU, serial per GPU, both GPUs in parallel) --------
# job = variant|regime|run_tag|extra-flags
# Balanced: each GPU gets 3 hide + 3 full (full-context runs are heavier, so don't
# pile all of them on one GPU).
JOBS_G0=(
  "actions_only|hide|airline_s2_actions_only_heldout|"
  "thoughts_policy|hide|airline_s2_thoughts_policy_heldout|--dataset_path $CORPUS/policy"
  "thoughts_policy|hide|airline_s2_thoughts_policy_last_heldout|--dataset_path $CORPUS/policy_last"
  "expert_thoughts|full|airline_s2_expert_thoughts_heldout_fullctx|"
  "thoughts_base|full|airline_s2_thoughts_base_heldout_fullctx|--dataset_path $CORPUS/base"
  "thoughts_base|full|airline_s2_thoughts_base_last_heldout_fullctx|--dataset_path $CORPUS/base_last"
)
JOBS_G1=(
  "expert_thoughts|hide|airline_s2_expert_thoughts_heldout|"
  "thoughts_base|hide|airline_s2_thoughts_base_heldout|--dataset_path $CORPUS/base"
  "thoughts_base|hide|airline_s2_thoughts_base_last_heldout|--dataset_path $CORPUS/base_last"
  "actions_only|full|airline_s2_actions_only_heldout_fullctx|"
  "thoughts_policy|full|airline_s2_thoughts_policy_heldout_fullctx|--dataset_path $CORPUS/policy"
  "thoughts_policy|full|airline_s2_thoughts_policy_last_heldout_fullctx|--dataset_path $CORPUS/policy_last"
)
log "=== Stage 2: launching 12 SFT runs (6/GPU, hide->GPU$G0, full->GPU$G1) ==="
run_stream $G0 "${JOBS_G0[@]}" &
s0=$!
run_stream $G1 "${JOBS_G1[@]}" &
s1=$!
wait $s0; wait $s1
log "=== airline sweep: Stage 1.5 + Stage 2 complete ==="

if [ -x .venv-tau2/bin/python ] && [ -d tau2-bench ]; then
  log "STAGE3: available (needs rl_eval->tau2 index map); e.g.:"
  log "  CUDA_VISIBLE_DEVICES=0 ./scripts/train_rl_from_sft.sh airline <stage2_step_best> --run_tag airline_s3_<variant>"
else
  log "STAGE3: SKIPPED — needs .venv-tau2 + ./tau2-bench + rl_eval->tau2 index map."
fi
