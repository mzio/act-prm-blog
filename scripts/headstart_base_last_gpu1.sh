#!/usr/bin/env bash
# Fill the idle GPU 1 (base_best relabel was skipped since the head-start already made
# that corpus, and the pipeline is bottlenecked on policy relabel on GPU 0). Here we
# head-start the base_LAST arm on GPU 1: relabel from the now-final base step_last
# (EM finished -> step_last is the batch-58 checkpoint), export corpus/base_last, then
# SFT thoughts_base(_last) in both regimes. Pipeline run_tags/paths so its post-EM
# base_last relabel + SFTs SKIP these (this finishes first, alone on GPU 1).
#   nohup ./scripts/headstart_base_last_gpu1.sh > /tmp/aprm/snorkel_finance_split_4b/headstart_base_last.driver.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/usr/local/bin:$PATH"
export HF_HOME="${HF_HOME:-/data/users/$USER/models/hf_cache}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1 UV_FROZEN=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

GPU=1
ENVC=act_prm/snorkel_finance_split
ENVNAME=snorkel_finance_split
MODEL=hf_qwen3_4b_instruct
CKR="checkpoints_lora/act_prm_${ENVNAME}/$MODEL"
LOGROOT="logs/act_prm_${ENVNAME}/$MODEL"
CORPUS="data/sft_corpus/${ENVNAME}"
MDIR=/tmp/aprm/${ENVNAME}_4b
mkdir -p "$MDIR"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] headstart-base_last(GPU$GPU): $*" | tee -a "$MDIR/headstart_base_last.log"; }
newest(){ ls -dt $1 2>/dev/null | head -1; }
done_ckpt(){ ls -d "$CKR/${1}"-*/step_best/adapter_model.safetensors >/dev/null 2>&1; }

log "=== base_last Act-PRM head start on GPU $GPU ==="
BL="$(newest "$CKR/${ENVNAME}_s1_base-*/step_last")"
if [ -z "$BL" ] || [ ! -f "$BL/adapter_model.safetensors" ]; then log "no base step_last — abort"; exit 1; fi
log "base step_last = $BL"

RTAG="${ENVNAME}_s1relabel_base_last"
OUT="$CORPUS/base_last"
if [ -f "$OUT/train.json" ]; then
  log "corpus $OUT exists — skip relabel/export"
else
  gen="$(newest "$LOGROOT/${RTAG}-*/generations.jsonl")"
  if [ -z "$gen" ]; then
    log "relabel base_last (--score_with_base --no_train --advantage_mode best, num_batches 31)"
    CUDA_VISIBLE_DEVICES=$GPU ./scripts/train.sh --env_config "$ENVC" --generator_config act_prm \
      --trainer_config pg --model_config $MODEL --lora_config r8_a16_linear --replay_buffer_config default \
      --score_with_base --no_train --resume_from "$BL" --advantage_mode best --group_size 4 --batch_size 4 \
      --num_batches 31 --eval_group_size 4 --no_initial_eval --length_penalty 0.15 \
      --save_generations --run_tag "$RTAG" --verbose > "$MDIR/${RTAG}.log" 2>&1 \
      && log "relabel base_last: done" || { log "relabel base_last: FAILED ($MDIR/${RTAG}.log)"; exit 1; }
    gen="$(newest "$LOGROOT/${RTAG}-*/generations.jsonl")"
  fi
  [ -z "$gen" ] && { log "no generations produced — abort"; exit 1; }
  log "export corpus -> $OUT"
  uv run python scripts/export_sft_corpus.py --generations "$gen" --source-pools "data/$ENVNAME" \
    --out "$OUT" >> "$MDIR/${RTAG}.log" 2>&1 && log "export: OK" || { log "export: FAILED"; exit 1; }
fi
n=$(python3 -c "import json;print(len(json.load(open('$OUT/train.json'))))" 2>/dev/null || echo 0)
log "corpus base_last: $n train"; [ "${n:-0}" -lt 1 ] && { log "WARN corpus EMPTY — abort"; exit 1; }
./scripts/backup_results.sh >/dev/null 2>&1 || true

sft(){ local tag=$1 fullctx=$2
  if done_ckpt "$tag"; then log "$tag: skip (done)"; return 0; fi
  local fc=(); [ "$fullctx" = 1 ] && fc=(SFT_FULLCTX=1)
  log "SFT $tag (fullctx=$fullctx)"
  CUDA_VISIBLE_DEVICES=$GPU env "${fc[@]}" ./scripts/train_sft.sh "$ENVC" thoughts_base \
    --model_config $MODEL --run_tag "$tag" --best_metric eval_action_ppl --gradient_checkpointing \
    --dataset_path "$OUT" > "$MDIR/${tag}.log" 2>&1 && log "$tag: done" || log "$tag: FAILED ($MDIR/${tag}.log)"
  ./scripts/backup_results.sh >/dev/null 2>&1 || true
}
sft "${ENVNAME}_s2_thoughts_base_last_heldout"          0
sft "${ENVNAME}_s2_thoughts_base_last_heldout_fullctx"  1
log "=== base_last Act-PRM head start complete ==="
