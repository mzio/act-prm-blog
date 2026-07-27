#!/usr/bin/env bash
# Generalized Act-PRM pipeline for ONE (env, model): Stage 1 EM -> Stage 1.5
# relabel(best+last)+export -> Stage 2 SFT x12. Env+model-parameterized so it works
# for tau2_airline / tau2_retail / snorkel_finance_split / ... on 4B or 8B.
# OFFLINE, uncapped, --gradient_checkpointing, resumable (skips existing outputs),
# 2 GPUs. Reads the split's act_prm_train count to size the relabel pass.
#
#   Stage 1    EM thought-gen (policy+base)  -> checkpoints .../<model>/<env>_s1_<scorer>
#   Stage 1.5  relabel(best+last)+export     -> data/sft_corpus/<envname>[_<mshort>]/{policy,base,policy_last,base_last}
#   Stage 2    SFT (fresh) x12               -> {actions_only, expert_thoughts,
#              thoughts_policy(best/last), thoughts_base(best/last)} x {hide-obs, full-context}
#              early-stop eval_action_ppl; logs eval_actiononly_ppl (action-subspan).
#
# Relies on the no-shuffle-in-no_train relabel fix (rl.py) so the export's
# pool[sample_id % n] mapping stays aligned for any num_batches (no corpus corruption).
#
# Usage:  MODEL=hf_qwen3_4b_instruct nohup ./scripts/run_actprm_pipeline.sh act_prm/snorkel_finance_split \
#           > /tmp/aprm/finance_4b/run.log 2>&1 &
#         (run again with MODEL=hf_qwen3_8b for the 8B arm — separate corpora/tags.)
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/usr/local/bin:$PATH"
export HF_HOME="${HF_HOME:-/data/users/$USER/models/hf_cache}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1 UV_FROZEN=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

ENVC="${1:?env_config, e.g. act_prm/snorkel_finance_split}"
MODEL="${MODEL:-hf_qwen3_4b_instruct}"
ENVNAME="${ENVC##*/}"
case "$MODEL" in *8b*) MSHORT=8b;; *) MSHORT=4b;; esac
TAGM=""; [ "$MSHORT" = 8b ] && TAGM="8b_"        # tag/corpus suffix only for non-4b
CORPUS="data/sft_corpus/${ENVNAME}"; [ "$MSHORT" = 8b ] && CORPUS="${CORPUS}_8b"
CKR="checkpoints_lora/${ENVC//\//_}/$MODEL"
LOGROOT="logs/${ENVC//\//_}/$MODEL"
G0=0; G1=1
MDIR="/tmp/aprm/${ENVNAME}_${MSHORT}"; mkdir -p "$MDIR"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$MDIR/orchestrator.log"; }
newest(){ ls -dt $1 2>/dev/null | head -1; }
wait_gpu_free(){ sleep 30; local f=0; while [ $f -lt 2 ]; do
  if pgrep -f '[m]ain_pytorch.py' >/dev/null 2>&1; then f=0; sleep 60; else f=$((f+1)); sleep 20; fi; done; }

# train tasks in the split -> relabel num_batches (cover all, +margin; no-shuffle fix keeps it aligned)
SPLIT_JSON="data/splits/${ENVNAME%_split}.json"; [ -f "$SPLIT_JSON" ] || SPLIT_JSON="data/splits/${ENVNAME}.json"
NTRAIN=$(python3 -c "import json;d=json.load(open('$SPLIT_JSON'));print(len(d['act_prm_train']))" 2>/dev/null || echo 64)
RELABEL_NB=$(python3 -c "print((int($NTRAIN)+3)//4 + 2)")     # ceil(n/4)+2, batch_size 4
EM_NB=$(python3 -c "print(max(25, ((int($NTRAIN)+3)//4)*2))") # ~2 epochs of coverage
log "env=$ENVC model=$MODEL | train_tasks=$NTRAIN | EM_nb=$EM_NB relabel_nb=$RELABEL_NB | corpus=$CORPUS"

em(){ local s=$1 g=$2 swb tag="${ENVNAME}_s1_${TAGM}${s}"
  [ "$s" = base ] && swb=--score_with_base || swb=--no-score_with_base
  [ -n "$(newest "$CKR/${tag}-*/step_best")" ] && { log "EM $s: skip (exists)"; return 0; }
  log "EM $s (GPU $g)"
  CUDA_VISIBLE_DEVICES=$g ./scripts/train.sh --env_config "$ENVC" --generator_config act_prm \
    --trainer_config pg --model_config $MODEL --lora_config r8_a16_linear --replay_buffer_config default \
    $swb --group_size 4 --batch_size 4 --num_batches $EM_NB --eval_every 5 --no_initial_eval \
    --length_penalty 0.15 --gradient_checkpointing --save_generations --run_tag "$tag" --verbose \
    > "$MDIR/s1_${s}.log" 2>&1 && log "EM $s: done" || log "EM $s: FAILED ($MDIR/s1_${s}.log)"; }

relabel_export(){ local s=$1 ckpt=$2 tag=$3 out=$4 g=$5 swb
  [ -f "$out/train.json" ] && { log "export $out: skip (exists)"; return 0; }
  { [ -z "$ckpt" ] || [ ! -f "$ckpt/adapter_model.safetensors" ]; } && { log "relabel $tag: no ckpt, skip"; return 1; }
  [ "$s" = base ] && swb=--score_with_base || swb=--no-score_with_base
  local gen; gen=$(newest "$LOGROOT/${tag}-*/generations.jsonl")
  if [ -z "$gen" ]; then log "relabel $tag (GPU $g)"
    CUDA_VISIBLE_DEVICES=$g ./scripts/train.sh --env_config "$ENVC" --generator_config act_prm \
      --trainer_config pg --model_config $MODEL --lora_config r8_a16_linear --replay_buffer_config default \
      $swb --no_train --resume_from "$ckpt" --advantage_mode best --group_size 4 --batch_size 4 \
      --num_batches $RELABEL_NB --eval_group_size 4 --no_initial_eval --length_penalty 0.15 \
      --save_generations --run_tag "$tag" --verbose > "$MDIR/${tag}.log" 2>&1
    gen=$(newest "$LOGROOT/${tag}-*/generations.jsonl"); fi
  [ -z "$gen" ] && { log "relabel $tag: no generations"; return 1; }
  uv run python scripts/export_sft_corpus.py --generations "$gen" --source-pools "data/$ENVNAME" \
    --out "$out" >> "$MDIR/${tag}.log" 2>&1 && log "export $out: OK" || log "export $out: FAILED"; }

sft(){ local v=$1 r=$2 tag=$3 g=$4; shift 4
  [ -n "$(newest "$CKR/${tag}-*/step_best")" ] && { log "SFT $tag: skip (exists)"; return 0; }
  local fc=(); [ "$r" = full ] && fc=(SFT_FULLCTX=1)
  log "SFT $tag (GPU $g, $r)"
  CUDA_VISIBLE_DEVICES=$g env "${fc[@]}" ./scripts/train_sft.sh "$ENVC" "$v" \
    --model_config $MODEL --run_tag "$tag" --best_metric eval_action_ppl --gradient_checkpointing "$@" \
    > "$MDIR/${tag}.log" 2>&1 && log "SFT $tag: done" || log "SFT $tag: FAILED ($MDIR/${tag}.log)"; }
run_stream(){ local g=$1; shift; for s in "$@"; do IFS='|' read -r v r t e <<< "$s"; sft "$v" "$r" "$t" "$g" $e; done; }

# ============================== DRIVE ==============================
log "=== pipeline start: waiting for GPUs ==="; wait_gpu_free
em policy $G0 & p=$!; sleep 60; em base $G1 & q=$!; wait $p; wait $q
PB=$(newest "$CKR/${ENVNAME}_s1_${TAGM}policy-*/step_best"); BB=$(newest "$CKR/${ENVNAME}_s1_${TAGM}base-*/step_best")
PL=$(newest "$CKR/${ENVNAME}_s1_${TAGM}policy-*/step_last"); BL=$(newest "$CKR/${ENVNAME}_s1_${TAGM}base-*/step_last")
log "ckpts best(p=$PB b=$BB) last(p=$PL b=$BL)"
( relabel_export policy "$PB" "${ENVNAME}_s1relabel_${TAGM}policy"      "$CORPUS/policy"      $G0 ) & p=$!
( relabel_export base   "$BB" "${ENVNAME}_s1relabel_${TAGM}base"        "$CORPUS/base"        $G1 ) & q=$!; wait $p; wait $q
( relabel_export policy "$PL" "${ENVNAME}_s1relabel_${TAGM}policy_last" "$CORPUS/policy_last" $G0 ) & p=$!
( relabel_export base   "$BL" "${ENVNAME}_s1relabel_${TAGM}base_last"   "$CORPUS/base_last"   $G1 ) & q=$!; wait $p; wait $q
for c in policy base policy_last base_last; do
  n=$(python3 -c "import json;print(len(json.load(open('$CORPUS/$c/train.json'))))" 2>/dev/null||echo 0)
  log "corpus $c: $n train"; [ "${n:-0}" -lt 1 ] && log "WARN: corpus $c EMPTY"; done
P="${ENVNAME}_s2_${TAGM}"   # SFT run_tag prefix
JOBS_G0=(
  "actions_only|hide|${P}actions_only_heldout|"
  "thoughts_policy|hide|${P}thoughts_policy_heldout|--dataset_path $CORPUS/policy"
  "thoughts_policy|hide|${P}thoughts_policy_last_heldout|--dataset_path $CORPUS/policy_last"
  "expert_thoughts|full|${P}expert_thoughts_heldout_fullctx|"
  "thoughts_base|full|${P}thoughts_base_heldout_fullctx|--dataset_path $CORPUS/base"
  "thoughts_base|full|${P}thoughts_base_last_heldout_fullctx|--dataset_path $CORPUS/base_last" )
JOBS_G1=(
  "expert_thoughts|hide|${P}expert_thoughts_heldout|"
  "thoughts_base|hide|${P}thoughts_base_heldout|--dataset_path $CORPUS/base"
  "thoughts_base|hide|${P}thoughts_base_last_heldout|--dataset_path $CORPUS/base_last"
  "actions_only|full|${P}actions_only_heldout_fullctx|"
  "thoughts_policy|full|${P}thoughts_policy_heldout_fullctx|--dataset_path $CORPUS/policy"
  "thoughts_policy|full|${P}thoughts_policy_last_heldout_fullctx|--dataset_path $CORPUS/policy_last" )
log "=== Stage 2: 12 SFT runs (6/GPU) ==="
run_stream $G0 "${JOBS_G0[@]}" & a=$!; run_stream $G1 "${JOBS_G1[@]}" & b=$!; wait $a; wait $b
log "=== pipeline complete for $ENVC ($MODEL) ==="
