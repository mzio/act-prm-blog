#!/usr/bin/env bash
# ROLLOUT eval of the finished Stage-2 hide-regime SFT checkpoints: let each policy act in
# the live tau2 gym and score TASK COMPLETION, not token perplexity.
#
# Why this and not an expanded teacher-forced eval: the thought arms are scored with an
# inferred thought in context, and inferred thoughts only exist for act_prm_train ∪
# act_prm_eval -- rl_eval was held out of Stage-1 EM. Expanding teacher-forced eval to
# rl_eval would mean running the EM generator over the hold-out first. A rollout eval
# needs no gold targets and no thought corpus at all: the model generates its own thought
# and action, and the tau2 evaluator scores the outcome. It also runs on the never-in-logs
# tasks, which have no expert demonstrations, and it measures the quantity the Act-PRM
# story is actually about -- does thinking make the agent ACT better.
#
# Task sets (never trained on, by construction):
#   retail  : 42 never-in-logs tau2 tasks (data/splits/tau2_retail_uid_to_tau2id.json)
#   airline : 18 unseen tau2 ids       (data/splits/tau2_airline_taskmap.json)
#
# Eval-only mode = --no_train --num_batches 1 --eval_every 1: batch 0 is also the last, so
# the trainer evals, generates one throwaway train batch, and skips the optimizer step.
# --hide_observations matches how these checkpoints were trained.
#
# Usage: setsid nohup ./scripts/run_sft_rollout_eval.sh > /tmp/aprm/rollout/driver.log 2>&1 &
#   SMOKE=1  one checkpoint, 1 task, 4 turns -- validates the tau2 user-sim path
set -uo pipefail
# fwdproxy began 403-ing huggingface.co on 2026-09-03, which killed model loading in
# load_hf_model_and_tokenizer -> hf_hub list_repo_tree (all 4 expert_thoughts_all runs
# and the insurance base rollout died in ~25s). The weights are cached locally under
# HF_HOME, so go offline and never touch the Hub. Verified: AutoConfig+AutoTokenizer
# for Qwen3-4B-Instruct-2507 load fine with HF_HUB_OFFLINE=1.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_HOME="${HF_HOME:-/data/users/mzio/models/hf_cache}"
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/usr/local/bin:$PATH"
# api.wandb.ai is NOT reachable through fwdproxy: without this the run blocks 90s in
# wandb.init and then dies on a broken CONNECT tunnel. Falls back to WANDB_MODE=offline.
[ -f scripts/wandb_preflight.sh ] && . scripts/wandb_preflight.sh
export PYTHONUNBUFFERED=1
# Dump the full conversation per rollout to <log_path>/trajectories.jsonl alongside the
# scores in rollouts_per_task.jsonl. Each rollout writes into its OWN run dir (the tag
# carries CKPT_TAG/CKPT_STEP), so this never overwrites prior work.
export ACT_PRM_DUMP_TRAJECTORIES="${ACT_PRM_DUMP_TRAJECTORIES:-1}"
MODEL="${MODEL_CFG:-hf_qwen3_4b_instruct}"; export MODEL_CFG="$MODEL"
MDIR=/tmp/aprm/rollout; mkdir -p "$MDIR"
SMOKE="${SMOKE:-0}"
MAX_TURNS="${MAX_TURNS:-20}"
# REGIME=hide  : --hide_observations, matching how the checkpoints were trained.
# REGIME=full  : same checkpoints, FULL context at rollout time. A train/test mismatch on
#                purpose -- it asks whether a policy trained to act on a compacted context
#                generalises when given the whole thing. Tagged _fullctx so the two passes
#                never collide.
REGIME="${REGIME:-hide}"
if [ "$REGIME" = full ]; then HIDE_ARGS=(); RTAG="_fullctx"; else HIDE_ARGS=(--hide_observations); RTAG=""; fi
VARIANTS="${VARIANTS:-actions_only expert_thoughts thoughts_policy thoughts_base}"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$MDIR/rollout.log"; }
wait_gpu_free(){ while pgrep -f '[m]ain_pytorch.py' >/dev/null 2>&1; do sleep 60; done; sleep 10; }
newest(){ ls -dt $1 2>/dev/null | head -1; }

RETAIL_IDS=$(python3 -c "import json;print(' '.join(json.load(open('data/splits/tau2_retail_uid_to_tau2id.json'))['never_in_logs']['ids']))")
AIRLINE_IDS=$(python3 -c "import json;print(' '.join(str(i) for i in json.load(open('data/splits/tau2_airline_taskmap.json'))['unseen_tau2_ids']))")
RETAIL_TRAIN=$(python3 -c "import json;print(json.load(open('data/splits/tau2_retail_uid_to_tau2id.json'))['coverage']['covered_tau2_ids'][0])")
AIRLINE_TRAIN=$(python3 -c "import json;print(json.load(open('data/splits/tau2_airline_taskmap.json'))['covered_tau2_ids'][0])")

run_one(){  # $1=domain  $2=variant  $3=eval ids  $4=throwaway train id
  local dom=$1 v=$2 ids=$3 tid=$4
  local envdir="act_prm_tau2_${dom}"
  # CKPT_PAT selects WHICH Stage-2 generation to roll out. Default is the historical
  # SGD-era lr3e_3/nb150 set; the 08-31 sft_flat arms are lr1e_3_adamw_nb200_flat32.
  # Hardcoding it silently evaluated obsolete checkpoints for hours on 08-29.
  local ck; ck=$(newest "checkpoints_lora/$envdir/$MODEL/${dom}_s2_${v}_${CKPT_PAT:-lr3e_3_nb150}_heldout-*/${CKPT_STEP:-step_best}")
  [ -z "$ck" ] && { log "ROLLOUT $dom/$v: no checkpoint, skip"; return; }
  # The tag MUST encode which Stage-2 generation is being rolled out. It was hardcoded
  # to lr3e_3, so the 08-31 sft_flat rollouts collided with the SGD-era ones: their .done
  # markers made actions_only "done, skip" (losing the baseline) and their log dirs would
  # have been appended to. CKPT_TAG defaults to the historical name for back-compat.
  local tag="${dom}_rollout_${v}_${CKPT_TAG:-lr3e_3}${CKPT_STEP:+_${CKPT_STEP}}${RTAG}"
  [ "$SMOKE" = 1 ] && tag="${tag}_smoke"
  if [ -f "$MDIR/${tag}.done" ]; then log "ROLLOUT $dom/$v: done, skip"; return; fi
  local nb=(--num_batches 1 --eval_every 1) turns=(--max_turns "$MAX_TURNS")
  if [ "$SMOKE" = 1 ]; then ids="$(echo $ids | awk '{print $1}')"; turns=(--max_turns 4); fi
  log "ROLLOUT $dom/$v  ($(echo $ids | wc -w) tasks) <- $ck"
  wait_gpu_free
  ./scripts/train_rl_from_sft.sh "$dom" "$ck" --run_tag "$tag" \
      --generator_config hf_rlvr --env_config "tau2bench/${dom}_rlvr" \
      --no_train "${nb[@]}" --group_size 2 --batch_size 1 \
      "${turns[@]}" --max_tokens 2048 --discount_factor 1.0 "${HIDE_ARGS[@]}" \
      --train_task_ids "$tid" --eval_task_ids $ids \
      > "$MDIR/${tag}.log" 2>&1 \
    || { log "ROLLOUT $dom/$v: FAILED (see $MDIR/${tag}.log)"; return; }
  # Validity gate. A transient Claude Agent SDK outage makes every episode die on turn 1,
  # which the trainer happily records as 0/N correct -- indistinguishable from a real
  # result unless you look at timesteps. Seen 08-22 01:35-02:09: all 8 full-context
  # rollouts returned 0/42 and 0/18 with timesteps=1.0 and exactly one generate call per
  # task. Refuse to mark such a run done, so it is retried instead of banked as a finding.
  if ! ROLLOUT_DIR="$(newest "logs/tau2bench_${dom}_rlvr/$MODEL/${tag}-*/")" \
       uv run --no-project python scripts/check_rollout_valid.py "$(newest "logs/tau2bench_${dom}_rlvr/$MODEL/${tag}-*/")"; then
    log "ROLLOUT $dom/$v: INVALID (episodes died on turn 1 -- user-sim outage?); not marking done"
    return
  fi
  touch "$MDIR/${tag}.done"; log "ROLLOUT $dom/$v: done"
}

log "=== SFT rollout eval (task completion) regime=$REGIME smoke=$SMOKE turns=$MAX_TURNS ==="
log "  retail hold-out: $(echo $RETAIL_IDS | wc -w) tasks | airline hold-out: $(echo $AIRLINE_IDS | wc -w) tasks"
for v in $VARIANTS; do run_one retail  "$v" "$RETAIL_IDS"  "$RETAIL_TRAIN";  done
for v in $VARIANTS; do run_one airline "$v" "$AIRLINE_IDS" "$AIRLINE_TRAIN"; done
log "=== rollout eval done ==="
