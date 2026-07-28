#!/usr/bin/env bash
# Sequentially run the no-corpus-needed head-start SFTs on the BASE-scored EM GPU
# (CUDA_VISIBLE_DEVICES=1) while Stage-1 EM is still running — so at most 2 EM + 1
# SFT run at once (safe unattended). These 4 variants (actions_only + expert_thoughts,
# each hide-obs + full-context) train from the PREBUILT pools (no EM corpora needed):
#   data/snorkel_finance_split            (action-only)
#   data/snorkel_finance_split_expert_thoughts  (expert thought+action / oracle)
# Uses the pipeline's exact run_tags, so run_actprm_pipeline.sh's Stage 2 SKIPS them
# later (step_best exists). Skip-guarded + resumable: re-running only does what's left.
#
# Usage:  nohup ./scripts/headstart_sft_seq.sh > /tmp/aprm/snorkel_finance_split_4b/headstart_seq.driver.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/usr/local/bin:$PATH"
export HF_HOME="${HF_HOME:-/data/users/$USER/models/hf_cache}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1 UV_FROZEN=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

GPU=1                                  # base-scored EM lane (verified CUDA_VISIBLE_DEVICES=1)
ENVC=act_prm/snorkel_finance_split
MODEL=hf_qwen3_4b_instruct
CKR="checkpoints_lora/act_prm_snorkel_finance_split/$MODEL"
MDIR=/tmp/aprm/snorkel_finance_split_4b
mkdir -p "$MDIR"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] headstart-seq: $*" | tee -a "$MDIR/headstart_seq.log"; }
done_tag(){ ls -d "$CKR/${1}"-*/step_best/adapter_model.safetensors >/dev/null 2>&1; }

run(){ local tag=$1 fullctx=$2 variant=$3
  if done_tag "$tag"; then log "$tag: skip (step_best exists)"; return 0; fi
  local fc=(); [ "$fullctx" = 1 ] && fc=(SFT_FULLCTX=1)
  log "$tag: START (GPU $GPU, fullctx=$fullctx, variant=$variant)"
  CUDA_VISIBLE_DEVICES=$GPU env "${fc[@]}" ./scripts/train_sft.sh "$ENVC" "$variant" \
    --model_config $MODEL --run_tag "$tag" --best_metric eval_action_ppl --gradient_checkpointing \
    > "$MDIR/headstart_${tag}.log" 2>&1 \
    && log "$tag: DONE" || log "$tag: FAILED (see $MDIR/headstart_${tag}.log)"
  ./scripts/backup_results.sh >/dev/null 2>&1 || true
}

log "=== head-start sequential SFT lane starting (GPU $GPU) ==="
# Wait for the in-flight expert_thoughts HIDE run to finish (trailing space -> not _fullctx).
log "waiting for in-flight expert_thoughts_heldout (hide) to finish..."
while pgrep -f 'run_tag snorkel_finance_split_s2_expert_thoughts_heldout ' >/dev/null 2>&1; do sleep 60; done
./scripts/backup_results.sh >/dev/null 2>&1 || true
log "expert_thoughts_heldout (hide) finished; running full-context head starts sequentially"

# Sequential full-context head starts on the base lane (hide-obs ones already done/running).
run snorkel_finance_split_s2_actions_only_heldout_fullctx    1 actions_only
run snorkel_finance_split_s2_expert_thoughts_heldout_fullctx 1 expert_thoughts
log "=== all 4 head-start SFTs complete (actions_only + expert_thoughts x {hide, full}) ==="
