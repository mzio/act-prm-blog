#!/usr/bin/env bash
# The thoughts-redo controller re-ran only the BEST-corpus thoughts SFT (thoughts_policy,
# thoughts_base); the _last-corpus SFT (thoughts_policy_last, thoughts_base_last × hide/full)
# were archived but NOT re-run (it passed the *_last name as an SFT variant, which
# train_sft.sh rejects). This re-runs those 4 on the CORRECTED _last corpora (base variant
# = thoughts_{policy,base}, corpus = {policy,base}_last, run_tag = *_last_*), then relaunches
# stage3 → runs the _last RL arms + the re-run thoughts_policy + full-42 evals.
#
# Usage: CUDA_VISIBLE_DEVICES=0 nohup ./scripts/run_retail_lastsft_redo.sh > /tmp/aprm/lastsft_redo.out 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/usr/local/bin:$PATH"
CK=checkpoints_lora/act_prm_tau2_retail/hf_qwen3_4b_instruct
MDIR=/tmp/aprm/stage3; mkdir -p "$MDIR"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a /tmp/aprm/lastsft_redo.log; }
wait_gpu_free(){ while pgrep -f '[m]ain_pytorch.py' >/dev/null 2>&1; do sleep 60; done; sleep 10; }
newest(){ ls -dt $1 2>/dev/null | head -1; }

run_sft(){  # $1=tag_label (thoughts_policy_last|thoughts_base_last)  $2=corpus  $3=regime
  local tag="retail_s2_$1_heldout"; [ "$3" = full ] && tag="${tag}_fullctx"
  if [ -n "$(newest "$CK/${tag}-*/step_best/adapter_model.safetensors")" ]; then log "$tag: exists, skip"; return 0; fi
  local fc=(); [ "$3" = full ] && fc=(SFT_FULLCTX=1)
  local base_variant=thoughts_policy; [[ "$1" == thoughts_base_last ]] && base_variant=thoughts_base
  log "SFT $tag  (base_variant=$base_variant corpus=$2 $3-obs)"
  wait_gpu_free
  env "${fc[@]}" ./scripts/train_sft.sh act_prm/tau2_retail "$base_variant" \
      --run_tag "$tag" --dataset_path "data/sft_corpus/tau2_retail/$2" --best_metric eval_action_ppl \
      > "$MDIR/${tag}.log" 2>&1 && log "$tag: done" || log "$tag: FAILED (see $MDIR/${tag}.log)"
}

log "=== _last thoughts SFT redo start (waits for GPU; runs the 4 missing SFTs) ==="
for reg in hide full; do
  run_sft thoughts_policy_last policy_last "$reg"
  run_sft thoughts_base_last   base_last   "$reg"
done
log "=== _last SFT done -> relaunch stage3 (re-runs failed thoughts_policy + _last RL + full-42 evals; skips done arms) ==="
CUDA_VISIBLE_DEVICES=0 ./scripts/run_retail_stage3.sh >> /tmp/aprm/lastsft_redo.log 2>&1
log "=== lastsft redo + stage3 relaunch complete ==="
