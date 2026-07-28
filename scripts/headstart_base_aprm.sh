#!/usr/bin/env bash
# HEAD START for the base-scored Act-PRM variant, on GPU 0 (the underutilized
# policy-EM lane) IN PARALLEL with policy EM — starts NOW, no waiting. Relabels
# thoughts_base from the CURRENT base EM step_best (plateaued at batch ~15 -> stable,
# identical to what the pipeline would use post-EM), exports the thought+action SFT
# corpus, then SFTs thoughts_base in both regimes (hide-obs + full-context).
# GPU 0 sat at ~14 GB / ~23% util with policy EM alone, so there's ample headroom.
# Sequential within this lane (relabel -> hide -> full). Pipeline run_tags/paths, so
# run_actprm_pipeline.sh Stage 1.5/2 SKIP these later.
#
#   nohup ./scripts/headstart_base_aprm.sh > /tmp/aprm/snorkel_finance_split_4b/headstart_base.driver.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/usr/local/bin:$PATH"
export HF_HOME="${HF_HOME:-/data/users/$USER/models/hf_cache}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1 UV_FROZEN=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

GPU=0                                  # policy-EM lane (underutilized) — run base head start here in parallel
ENVC=act_prm/snorkel_finance_split
ENVNAME=snorkel_finance_split
MODEL=hf_qwen3_4b_instruct
CKR="checkpoints_lora/act_prm_${ENVNAME}/$MODEL"
LOGROOT="logs/act_prm_${ENVNAME}/$MODEL"
CORPUS="data/sft_corpus/${ENVNAME}"
MDIR=/tmp/aprm/${ENVNAME}_4b
mkdir -p "$MDIR"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] headstart-base(GPU$GPU): $*" | tee -a "$MDIR/headstart_base.log"; }
newest(){ ls -dt $1 2>/dev/null | head -1; }
done_ckpt(){ ls -d "$CKR/${1}"-*/step_best/adapter_model.safetensors >/dev/null 2>&1; }

log "=== base Act-PRM head start on GPU $GPU (parallel with policy EM) ==="
BB="$(newest "$CKR/${ENVNAME}_s1_base-*/step_best")"
if [ -z "$BB" ] || [ ! -f "$BB/adapter_model.safetensors" ]; then log "no base step_best — abort"; exit 1; fi
log "base step_best = $BB"

# 1) Stage 1.5 relabel (TOP-1 best) from base step_best -> generations.jsonl -> corpus
RTAG="${ENVNAME}_s1relabel_base"
OUT="$CORPUS/base"
if [ -f "$OUT/train.json" ]; then
  log "corpus $OUT exists — skip relabel/export"
else
  gen="$(newest "$LOGROOT/${RTAG}-*/generations.jsonl")"
  if [ -z "$gen" ]; then
    log "relabel base (--score_with_base --no_train --advantage_mode best, num_batches 31)"
    CUDA_VISIBLE_DEVICES=$GPU ./scripts/train.sh --env_config "$ENVC" --generator_config act_prm \
      --trainer_config pg --model_config $MODEL --lora_config r8_a16_linear --replay_buffer_config default \
      --score_with_base --no_train --resume_from "$BB" --advantage_mode best --group_size 4 --batch_size 4 \
      --num_batches 31 --eval_group_size 4 --no_initial_eval --length_penalty 0.15 \
      --save_generations --run_tag "$RTAG" --verbose > "$MDIR/${RTAG}.log" 2>&1 \
      && log "relabel base: done" || { log "relabel base: FAILED ($MDIR/${RTAG}.log)"; exit 1; }
    gen="$(newest "$LOGROOT/${RTAG}-*/generations.jsonl")"
  fi
  [ -z "$gen" ] && { log "no generations produced — abort"; exit 1; }
  log "export corpus -> $OUT"
  uv run python scripts/export_sft_corpus.py --generations "$gen" --source-pools "data/$ENVNAME" \
    --out "$OUT" >> "$MDIR/${RTAG}.log" 2>&1 && log "export: OK" || { log "export: FAILED"; exit 1; }
fi
n=$(python3 -c "import json;print(len(json.load(open('$OUT/train.json'))))" 2>/dev/null || echo 0)
log "corpus base: $n train"; [ "${n:-0}" -lt 1 ] && { log "WARN corpus EMPTY — abort"; exit 1; }
./scripts/backup_results.sh >/dev/null 2>&1 || true

# 2) Stage 2 SFT thoughts_base x {hide, full} on the base corpus (sequential, this lane)
sft(){ local tag=$1 fullctx=$2
  if done_ckpt "$tag"; then log "$tag: skip (done)"; return 0; fi
  local fc=(); [ "$fullctx" = 1 ] && fc=(SFT_FULLCTX=1)
  log "SFT $tag (fullctx=$fullctx)"
  CUDA_VISIBLE_DEVICES=$GPU env "${fc[@]}" ./scripts/train_sft.sh "$ENVC" thoughts_base \
    --model_config $MODEL --run_tag "$tag" --best_metric eval_action_ppl --gradient_checkpointing \
    --dataset_path "$OUT" > "$MDIR/${tag}.log" 2>&1 && log "$tag: done" || log "$tag: FAILED ($MDIR/${tag}.log)"
  ./scripts/backup_results.sh >/dev/null 2>&1 || true
}
sft "${ENVNAME}_s2_thoughts_base_heldout"          0
sft "${ENVNAME}_s2_thoughts_base_heldout_fullctx"  1
log "=== base Act-PRM head start complete (relabel + thoughts_base hide+full) ==="
