#!/usr/bin/env bash
# Relabel expert data with the EM checkpoints and export SFT corpora, for BOTH:
#   step_best -> data/sft_corpus/<env>/{policy,base}
#   step_last -> data/sft_corpus/<env>/{policy_last,base_last}
# best-of-G thought per logged step (--advantage_mode best). Resumable: a corpus is
# regenerated unless its train.json is NON-EMPTY (guards against the old --no_train bug
# that produced train:0 corpora — those are treated as missing).
#
# Usage:  CUDA_VISIBLE_DEVICES=0 ./scripts/run_relabel.sh act_prm/tau2_retail
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
MDIR="/tmp/aprm/relabel_$ENVNAME"; mkdir -p "$MDIR"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$MDIR/relabel.log"; }
wait_gpu_free(){ while pgrep -f '[m]ain_pytorch.py' >/dev/null 2>&1; do sleep 60; done; sleep 10; }
newest(){ ls -dt $1 2>/dev/null | head -1; }
n_train(){ python3 -c "import json;print(len(json.load(open('$1/train.json'))))" 2>/dev/null || echo 0; }
corpus_ok(){ [ -s "$1/train.json" ] && [ "$(n_train "$1")" -gt 0 ]; }

relabel_export(){  # $1=scorer(policy|base)  $2=swb(0|1)  $3=kind(best|last)
  local scorer=$1 swb=$2 kind=$3
  local suffix=""; [ "$kind" = last ] && suffix="_last"
  local out="$CORPUS/${scorer}${suffix}"
  corpus_ok "$out" && { log "${scorer}/${kind}: corpus non-empty ($(n_train "$out") train), skip"; return 0; }
  rm -rf "$out"   # clear any stale/empty corpus
  local ck; ck=$(ls -dt "$CKROOT"/*-nf=0-lp=0_15-swb=${swb}-*/step_${kind} 2>/dev/null | head -1)
  [ -z "$ck" ] && { log "${scorer}/${kind}: no EM step_${kind} under $CKROOT, skip"; return 1; }
  local sflag; sflag=$([ "$scorer" = base ] && echo --score_with_base || echo --no-score_with_base)
  local tag="${DOM}_s1relabel_${scorer}_${kind}_heldout"
  log "relabel ${scorer}/${kind} from $ck"
  wait_gpu_free
  ./scripts/train.sh --env_config "$ENVCFG" --generator_config act_prm --trainer_config pg \
    --model_config $MODEL --lora_config r8_a16_linear --replay_buffer_config default $sflag \
    --no_train --resume_from "$ck" --advantage_mode best --group_size 4 --batch_size 4 \
    --num_batches 20 --no_initial_eval --length_penalty 0.15 --save_generations \
    --run_tag "$tag" --verbose > "$MDIR/relabel_${scorer}_${kind}.log" 2>&1
  local gen; gen=$(newest "$LOGROOT/${tag}-*/generations.jsonl")
  [ -z "$gen" ] && { log "${scorer}/${kind}: no generations.jsonl, export skipped"; return 1; }
  uv run --no-sync python scripts/export_sft_corpus.py --generations "$gen" \
    --source-pools "data/$ENVNAME" --out "$out" >> "$MDIR/relabel_${scorer}_${kind}.log" 2>&1
  if corpus_ok "$out"; then log "${scorer}/${kind}: OK -> $out ($(n_train "$out") train)"
  else log "${scorer}/${kind}: export produced EMPTY train — INVESTIGATE $MDIR/relabel_${scorer}_${kind}.log"; fi
}

log "=== relabel + export (best + last) for $ENVCFG ==="
for kind in best last; do
  relabel_export policy 0 "$kind"
  relabel_export base   1 "$kind"
done
log "=== relabel done ==="
