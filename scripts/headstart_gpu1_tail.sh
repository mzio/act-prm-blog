#!/usr/bin/env bash
# GPU 1 (base-EM lane) tail: after the currently-running actions_only full-ctx SFT
# finishes, run the remaining no-corpus head start there — expert_thoughts full-ctx
# (its hide variant is already done). Sequential; pipeline run_tag so Stage 2 skips it.
# (base Act-PRM relabel + thoughts_base now run on GPU 0 in parallel — see
# headstart_base_aprm.sh.)
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
MDIR=/tmp/aprm/${ENVNAME}_4b
mkdir -p "$MDIR"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] headstart-gpu1: $*" | tee -a "$MDIR/headstart_gpu1.log"; }
done_ckpt(){ ls -d "$CKR/${1}"-*/step_best/adapter_model.safetensors >/dev/null 2>&1; }

log "waiting for the running actions_only SFT (GPU $GPU) to finish..."
while pgrep -f 'run_tag snorkel_finance_split_s2_actions_only' >/dev/null 2>&1; do sleep 60; done
log "actions_only done — running expert_thoughts full-ctx"

ET="${ENVNAME}_s2_expert_thoughts_heldout_fullctx"
if done_ckpt "$ET"; then log "$ET: skip (done)"; else
  CUDA_VISIBLE_DEVICES=$GPU SFT_FULLCTX=1 ./scripts/train_sft.sh "$ENVC" expert_thoughts \
    --model_config $MODEL --run_tag "$ET" --best_metric eval_action_ppl --gradient_checkpointing \
    > "$MDIR/${ET}.log" 2>&1 && log "$ET: done" || log "$ET: FAILED ($MDIR/${ET}.log)"
  ./scripts/backup_results.sh >/dev/null 2>&1 || true
fi
log "=== GPU1 tail complete (expert_thoughts full) ==="
