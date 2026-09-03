#!/usr/bin/env bash
# Stage-1 EM with BASE-MODEL likelihoods -> the missing `base_adamw30` corpora, all domains.
#
# WHAT THIS IS. Every Stage-1 corpus we have (`policy_adamw30`) rewards a candidate thought
# by p(x|s,z) under the *current LoRA policy*. This run detaches the LoRA and scores with
# the frozen base model instead (`--score_with_base` -> `_action_logprobs(use_base=True)` in
# generator/act_prm/base.py:319). Everything else is byte-identical to the policy recipe.
#
# WHY IT MATTERS. `run_sft_sweep.sh` already has a `thoughts_base_adamw30` arm wired up, and
# it has been silently skipped in EVERY sweep with
#   "WARN: data/sft_corpus/<env>/base_adamw30 missing/empty (thoughts_base_adamw30 skipped)"
# because the corpus was never generated. This produces it.
#
# RECIPE — copied verbatim from the policy_adamw30 lineage, only the scorer differs:
#   Stage-1 EM : r32_a32_linear, adamw, lr 4e-5, nb 30, group 4, batch 4, length_penalty 0,
#                advantage_mode action_probs, gradient_checkpointing, no_initial_eval,
#                eval_every 30, save_every 10          [was: --no-score_with_base]
#   relabel    : --no_train --resume_from <step_best> --advantage_mode best, group 4,
#                batch 4, length_penalty 0, nb = per-domain (ceil(n_train/4))
#   export     : scripts/export_sft_corpus.py --source-pools <the pool the policy used>
#
# COST (measured on the policy runs): airline 4.6h, retail 7.0h, insurance 10.6h,
# finance 19.6h => ~41.8h total. Ordered cheapest-first so results land early.
#
# NOTE ON FINANCE: it is 19.6h and its rollout eval CANNOT discriminate between arms
# (floor effect -- 9 of 10 finance fair-set rollouts ever run scored 0/10; see cc-13.0
# challenge #6). Its base corpus is still worth having for the teacher-forced action-PPL
# comparison, but do not expect a rollout result from it. It runs LAST for that reason.
set -uo pipefail
cd /home/mzio/projects/act-prm-blog
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/usr/local/bin:$PATH"
[ -f scripts/wandb_preflight.sh ] && . scripts/wandb_preflight.sh
export PYTHONUNBUFFERED=1
export HF_HOME=/data/users/mzio/models/hf_cache
export HF_HUB_OFFLINE=1
export WANDB_ERROR_REPORTING=false
G=/tmp/aprm/stage1_base; mkdir -p "$G"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$G/chain.log"; }
reap(){ for p in $(pgrep -f 'main_pytorch\.py'); do kill -9 "$p" 2>/dev/null; done; sleep 15; }
newest(){ ls -dt $1 2>/dev/null | head -1; }

one_domain(){  # $1=envcfg $2=dom $3=corpusdir $4=relabel_nb $5=source_pool
  local envcfg=$1 dom=$2 cdir=$3 rnb=$4 pool=$5
  local out="data/sft_corpus/$cdir/base_adamw30"
  local ckroot="checkpoints_lora/$( [ "${envcfg#act_prm/}" = "$envcfg" ] && echo "$envcfg" || echo "act_prm_${envcfg#act_prm/}" )/hf_qwen3_4b_instruct"
  local logroot="logs/act_prm_${envcfg#act_prm/}/hf_qwen3_4b_instruct"

  # ---- 1. Stage-1 EM training, base-scored
  if [ ! -f "$G/em.$dom.done" ]; then
    log "STAGE-1 EM (base-scored) $dom"
    ./scripts/train.sh --env_config "$envcfg" --generator_config act_prm --trainer_config pg \
      --model_config hf_qwen3_4b_instruct --lora_config r32_a32_linear \
      --replay_buffer_config default --score_with_base \
      --run_tag "${dom}_s1em_base_adamw30" --group_size 4 --batch_size 4 --num_batches 30 \
      --learning_rate 4e-5 --length_penalty 0 --advantage_mode action_probs --optimizer adamw \
      --save_generations --no_initial_eval --eval_every 30 --gradient_checkpointing \
      --save_every 10 --verbose >>"$G/em.$dom.log" 2>&1
    log "  rc=$?"
    local n; n=$(ls -d "$ckroot/${dom}_s1em_base_adamw30-"*/step_* 2>/dev/null | wc -l)
    log "  snapshots: $n"
    [ "$n" -gt 0 ] || { log "  NO SNAPSHOTS for $dom -- skipping its relabel"; return 1; }
    touch "$G/em.$dom.done"; reap
  fi

  # ---- 2. relabel with the same base scorer
  local ck; ck=$(newest "$ckroot/${dom}_s1em_base_adamw30-*/step_best")
  [ -z "$ck" ] && { log "$dom: no step_best, skip relabel"; return 1; }
  if [ ! -f "$G/relabel.$dom.done" ]; then
    log "RELABEL (base-scored) $dom  nb=$rnb  <- $ck"
    ./scripts/train.sh --env_config "$envcfg" --generator_config act_prm --trainer_config pg \
      --model_config hf_qwen3_4b_instruct --lora_config r32_a32_linear \
      --replay_buffer_config default --score_with_base \
      --no_train --resume_from "$ck" --advantage_mode best --group_size 4 --batch_size 4 \
      --num_batches "$rnb" --no_initial_eval --length_penalty 0 --save_generations \
      --run_tag "${dom}_s1relabel_base_adamw30" --verbose >>"$G/relabel.$dom.log" 2>&1
    log "  rc=$?"; touch "$G/relabel.$dom.done"; reap
  fi

  # ---- 3. export
  local gen; gen=$(newest "$logroot/${dom}_s1relabel_base_adamw30-*/generations.jsonl")
  [ -z "$gen" ] && { log "$dom: no generations.jsonl, export skipped"; return 1; }
  rm -rf "$out"
  uv run --no-project python scripts/export_sft_corpus.py --generations "$gen" \
    --source-pools "$pool" --out "$out" >>"$G/export.$dom.log" 2>&1
  local nt; nt=$(python3 -c "import json;print(len(json.load(open('$out/train.json'))))" 2>/dev/null || echo 0)
  log "$dom: exported $out  train=$nt"
}

log "=== Stage-1 base-scored corpora: airline -> retail -> insurance -> finance ==="
one_domain act_prm/tau2_airline          airline   tau2_airline           6  data/tau2_airline            || true
one_domain act_prm/tau2_retail           retail    tau2_retail           13  data/tau2_retail             || true
one_domain act_prm/snorkel_insurance     insurance snorkel_insurance     45  data/snorkel_insurance_split || true
one_domain act_prm/snorkel_finance_split finance   snorkel_finance_split 29  data/snorkel_finance_split_v3 || true
log "=== Stage-1 base-scored corpora complete ==="
