#!/usr/bin/env bash
# Relabel the expert data with the EM *step_last* checkpoint (fully trained) instead of
# step_best (which peaked early — batch 5 of 24 on retail), and export an alternative SFT
# corpus: data/sft_corpus/<env>/{policy_last,base_last}. Resumable (skips if corpus exists).
# The step_best corpora (data/sft_corpus/<env>/{policy,base}) are produced by Stage 1.5.
#
# Usage:  CUDA_VISIBLE_DEVICES=0 ./scripts/run_relabel_last.sh act_prm/tau2_retail
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/usr/local/bin:$PATH"
export https_proxy="${https_proxy:-http://fwdproxy:8080}"
export http_proxy="${http_proxy:-http://fwdproxy:8080}"
export HF_TOKEN="${HF_TOKEN:-$(cat "$HOME/models/token" 2>/dev/null || true)}"

ENVCFG="${1:-act_prm/tau2_retail}"; ENVNAME="${ENVCFG##*/}"; DOM="${ENVNAME#tau2_}"
MODEL=hf_qwen3_4b_instruct
CKROOT="checkpoints_lora/${ENVCFG//\//_}/$MODEL"
LOGROOT="logs/${ENVCFG//\//_}/$MODEL"
CORPUS="data/sft_corpus/$ENVNAME"
MDIR="/tmp/aprm/relabel_last_$ENVNAME"; mkdir -p "$MDIR"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$MDIR/relabel.log"; }
wait_gpu_free(){ while pgrep -f '[m]ain_pytorch.py' >/dev/null 2>&1; do sleep 60; done; sleep 10; }
newest(){ ls -dt $1 2>/dev/null | head -1; }

relabel_export(){  # $1=scorer(policy|base)  $2=swb(0|1)
  local scorer=$1 swb=$2 out="$CORPUS/${scorer}_last"
  [ -f "$out/train.json" ] && { log "${scorer}_last: corpus exists, skip"; return 0; }
  # EM step_last (exclude the relabel runs, which have nt=1/am=best in their name)
  local last; last=$(ls -dt "$CKROOT"/*-nf=0-lp=0_15-swb=${swb}-*/step_last 2>/dev/null | head -1)
  [ -z "$last" ] && { log "${scorer}: no EM step_last under $CKROOT (swb=$swb), skip"; return 1; }
  local sflag; sflag=$([ "$scorer" = base ] && echo --score_with_base || echo --no-score_with_base)
  local tag="${DOM}_s1relabel_${scorer}_last_heldout"
  log "relabel ${scorer} from step_last: $last"
  wait_gpu_free
  ./scripts/train.sh --env_config "$ENVCFG" --generator_config act_prm --trainer_config pg \
    --model_config $MODEL --lora_config r8_a16_linear --replay_buffer_config default $sflag \
    --no_train --resume_from "$last" --advantage_mode best --group_size 4 --batch_size 4 \
    --num_batches 40 --no_initial_eval --length_penalty 0.15 --save_generations \
    --run_tag "$tag" --verbose > "$MDIR/relabel_${scorer}_last.log" 2>&1
  local gen; gen=$(newest "$LOGROOT/${tag}-*/generations.jsonl")
  [ -z "$gen" ] && { log "${scorer}_last: no generations.jsonl, skip export"; return 1; }
  uv run --no-sync python scripts/export_sft_corpus.py --generations "$gen" \
    --source-pools "data/$ENVNAME" --out "$out" >> "$MDIR/relabel_${scorer}_last.log" 2>&1 \
    && log "${scorer}_last: exported -> $out" || log "${scorer}_last: export FAILED (see log)"
}

log "=== step_last relabel + export for $ENVCFG ==="
relabel_export policy 0
relabel_export base 1
log "=== step_last relabel done ==="
