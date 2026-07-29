#!/usr/bin/env bash
# Head-start the thoughts_policy (best) SFTs. The `policy` corpus is already exported
# (strict-clean), but the pipeline serializes ALL relabels before Stage 2, so it won't
# start these until policy_last relabel finishes. Run them now on GPU 0 (the policy_last
# relabel there is generation-bound/low-util, so there's headroom). Pipeline run_tags
# so Stage 2 SKIPS them later.
#   nohup ./scripts/headstart_policy_best_sft.sh > /tmp/aprm/snorkel_finance_split_4b/headstart_policy_best.driver.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/usr/local/bin:$PATH"
export HF_HOME="${HF_HOME:-/data/users/$USER/models/hf_cache}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1 UV_FROZEN=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

GPU=0
ENVC=act_prm/snorkel_finance_split
ENVNAME=snorkel_finance_split
MODEL=hf_qwen3_4b_instruct
CK="checkpoints_lora/act_prm_${ENVNAME}/$MODEL"
CORPUS="data/sft_corpus/${ENVNAME}/policy"
MDIR=/tmp/aprm/${ENVNAME}_4b
mkdir -p "$MDIR"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] headstart-policy_best(GPU$GPU): $*" | tee -a "$MDIR/headstart_policy_best.log"; }
done_ckpt(){ ls -d "$CK/${1}"-*/step_best/adapter_model.safetensors >/dev/null 2>&1; }

[ -f "$CORPUS/train.json" ] || { log "policy corpus missing — abort"; exit 1; }
log "=== thoughts_policy (best) SFT head start on GPU $GPU (corpus=$CORPUS) ==="
sft(){ local tag=$1 fullctx=$2
  if done_ckpt "$tag"; then log "$tag: skip (done)"; return 0; fi
  local fc=(); [ "$fullctx" = 1 ] && fc=(SFT_FULLCTX=1)
  log "SFT $tag (fullctx=$fullctx)"
  CUDA_VISIBLE_DEVICES=$GPU env "${fc[@]}" ./scripts/train_sft.sh "$ENVC" thoughts_policy \
    --model_config $MODEL --run_tag "$tag" --best_metric eval_action_ppl --gradient_checkpointing \
    --dataset_path "$CORPUS" > "$MDIR/${tag}.log" 2>&1 && log "$tag: done" || log "$tag: FAILED ($MDIR/${tag}.log)"
  ./scripts/backup_results.sh >/dev/null 2>&1 || true
}
sft "${ENVNAME}_s2_thoughts_policy_heldout"          0
sft "${ENVNAME}_s2_thoughts_policy_heldout_fullctx"  1
log "=== thoughts_policy (best) head start complete (hide+full) ==="
