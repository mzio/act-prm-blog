#!/usr/bin/env bash
# HEAD START for the base-scored Act-PRM variant: relabel from the CURRENT base EM
# step_best (plateaued at batch ~15 — stable, so identical to what the pipeline would
# use post-EM), export the thought+action SFT corpus, then SFT thoughts_base in both
# regimes (hide-obs + full-context). Runs SEQUENTIALLY on the base-EM GPU lane
# (CUDA_VISIBLE_DEVICES=1) AFTER the existing head-start SFT queue, so GPU 1 never has
# >1 SFT/relabel at once alongside base EM. Uses the pipeline's exact run_tags/paths,
# so run_actprm_pipeline.sh's Stage 1.5/2 SKIP these later (corpus train.json + step_best
# exist) — no double-work, no conflict (this finishes before the post-EM relabel).
#
#   nohup ./scripts/headstart_base_aprm.sh > /tmp/aprm/snorkel_finance_split_4b/headstart_base.driver.log 2>&1 &
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
log(){ echo "[$(date '+%m-%d %H:%M:%S')] headstart-base: $*" | tee -a "$MDIR/headstart_base.log"; }
newest(){ ls -dt $1 2>/dev/null | head -1; }
done_ckpt(){ ls -d "$CKR/${1}"-*/step_best/adapter_model.safetensors >/dev/null 2>&1; }

# 0) base is PRIORITIZED ahead of expert_thoughts full-ctx: wait only for the
#    currently-running actions_only full-ctx SFT to clear, then base runs NEXT.
#    (headstart_sft_seq.sh is stopped so it won't launch expert_thoughts full;
#    expert_thoughts full is appended at the END of this runner instead.)
log "waiting for the running actions_only SFT to finish (base runs next, ahead of expert_thoughts full)..."
while pgrep -f 'run_tag snorkel_finance_split_s2_actions_only' >/dev/null 2>&1; do sleep 60; done
log "SFT lane free — starting base Act-PRM head start"

BB="$(newest "$CKR/${ENVNAME}_s1_base-*/step_best")"
if [ -z "$BB" ] || [ ! -f "$BB/adapter_model.safetensors" ]; then log "no base step_best — abort"; exit 1; fi
log "base step_best = $BB"

# 1) Stage 1.5 relabel (TOP-1 best) from base step_best -> generations.jsonl
RTAG="${ENVNAME}_s1relabel_base"
OUT="$CORPUS/base"
if [ -f "$OUT/train.json" ]; then
  log "corpus $OUT exists — skip relabel/export"
else
  gen="$(newest "$LOGROOT/${RTAG}-*/generations.jsonl")"
  if [ -z "$gen" ]; then
    log "relabel base (GPU $GPU, --score_with_base --no_train --advantage_mode best)"
    CUDA_VISIBLE_DEVICES=$GPU ./scripts/train.sh --env_config "$ENVC" --generator_config act_prm \
      --trainer_config pg --model_config $MODEL --lora_config r8_a16_linear --replay_buffer_config default \
      --score_with_base --no_train --resume_from "$BB" --advantage_mode best --group_size 4 --batch_size 4 \
      --num_batches 31 --eval_group_size 4 --no_initial_eval --length_penalty 0.15 \
      --save_generations --run_tag "$RTAG" --verbose > "$MDIR/${RTAG}.log" 2>&1 \
      && log "relabel base: done" || { log "relabel base: FAILED ($MDIR/${RTAG}.log)"; exit 1; }
    gen="$(newest "$LOGROOT/${RTAG}-*/generations.jsonl")"
  fi
  [ -z "$gen" ] && { log "no generations produced — abort"; exit 1; }
  log "export corpus -> $OUT (from $gen)"
  uv run python scripts/export_sft_corpus.py --generations "$gen" --source-pools "data/$ENVNAME" \
    --out "$OUT" >> "$MDIR/${RTAG}.log" 2>&1 && log "export: OK" || { log "export: FAILED"; exit 1; }
fi
n=$(python3 -c "import json;print(len(json.load(open('$OUT/train.json'))))" 2>/dev/null || echo 0)
log "corpus base: $n train"; [ "${n:-0}" -lt 1 ] && { log "WARN corpus EMPTY — abort"; exit 1; }
./scripts/backup_results.sh >/dev/null 2>&1 || true

# 2) Stage 2 SFT thoughts_base x {hide, full} on the base corpus
sft(){ local tag=$1 fullctx=$2
  if done_ckpt "$tag"; then log "$tag: skip (done)"; return 0; fi
  local fc=(); [ "$fullctx" = 1 ] && fc=(SFT_FULLCTX=1)
  log "SFT $tag (GPU $GPU, fullctx=$fullctx)"
  CUDA_VISIBLE_DEVICES=$GPU env "${fc[@]}" ./scripts/train_sft.sh "$ENVC" thoughts_base \
    --model_config $MODEL --run_tag "$tag" --best_metric eval_action_ppl --gradient_checkpointing \
    --dataset_path "$OUT" > "$MDIR/${tag}.log" 2>&1 && log "$tag: done" || log "$tag: FAILED ($MDIR/${tag}.log)"
  ./scripts/backup_results.sh >/dev/null 2>&1 || true
}
sft "${ENVNAME}_s2_thoughts_base_heldout"          0
sft "${ENVNAME}_s2_thoughts_base_heldout_fullctx"  1
log "base thoughts_base done; now the deprioritized expert_thoughts full-ctx head start"

# expert_thoughts full-ctx (moved behind base; the hide variant is already done).
ET="${ENVNAME}_s2_expert_thoughts_heldout_fullctx"
if done_ckpt "$ET"; then log "$ET: skip (done)"; else
  log "SFT $ET (GPU $GPU, fullctx=1, expert_thoughts)"
  CUDA_VISIBLE_DEVICES=$GPU SFT_FULLCTX=1 ./scripts/train_sft.sh "$ENVC" expert_thoughts \
    --model_config $MODEL --run_tag "$ET" --best_metric eval_action_ppl --gradient_checkpointing \
    > "$MDIR/${ET}.log" 2>&1 && log "$ET: done" || log "$ET: FAILED ($MDIR/${ET}.log)"
  ./scripts/backup_results.sh >/dev/null 2>&1 || true
fi
log "=== base Act-PRM + expert_thoughts full head starts complete ==="
