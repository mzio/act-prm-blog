#!/usr/bin/env bash
# Autonomous Stage-3: after the SFT sweep finishes, (1) collect + commit the SFT
# analysis, (2) SMOKE-validate the tau2-gym RL path cheaply, and only if that works
# (3) RL from each variant's best SFT checkpoint. Resumable; safe to re-run.
#
# NOTE: the tau2-gym RL path is UNTESTED end-to-end here, hence the smoke gate.
# RL now TRAINS on our 72 logged retail tasks and EVALS on the 42 never-in-logs
# tasks (the purer hold-out), via explicit --train_task_ids / --eval_task_ids read
# from data/splits/tau2_retail_uid_to_tau2id.json (see the split section below).
#
# Usage: CUDA_VISIBLE_DEVICES=0 nohup ./scripts/run_retail_stage3.sh > /tmp/aprm/stage3/run.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/usr/local/bin:$PATH"

# Model is parametrized: MODEL_CFG selects the <MODEL> path dir AND (exported) the
# --model_config used by train_rl_from_sft.sh / analyze_sft.py. Default keeps 4B behavior.
MODEL="${MODEL_CFG:-hf_qwen3_4b_instruct}"; export MODEL_CFG="$MODEL"
CKROOT="checkpoints_lora/act_prm_tau2_retail/$MODEL"   # SFT (Stage-2) checkpoints
S3ROOT="checkpoints_lora/tau2bench_retail/$MODEL"      # RL (Stage-3) runs (env tau2bench/retail)
S3LOGROOT="logs/tau2bench_retail/$MODEL"               # RL (Stage-3) LOGS root (metrics.jsonl lives here)
MDIR=/tmp/aprm/stage3; mkdir -p "$MDIR"
# DRY=1: resolve + print each arm's exact command (no GPU, no analysis/smoke/RL,
# no snapshot). Used to verify the matrix without touching the GPU.
DRY="${DRY:-0}"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$MDIR/stage3.log"; }
wait_gpu_free(){ while pgrep -f '[m]ain_pytorch.py' >/dev/null 2>&1; do sleep 60; done; sleep 10; }
newest(){ ls -dt $1 2>/dev/null | head -1; }

log "=== Stage-3 driver start ==="

# --- Explicit task-id split (train on the 72 logged, eval on the 42 never-in-logs) ---
# The uid->tau2id map (scripts/make_split.py) records which tau2 retail tasks appear
# in the expert logs used for Stage-1 EM + Stage-2 SFT. We RL-train on those 72
# (coverage.covered_tau2_ids) and RL-eval on the 42 that NEVER appear
# (never_in_logs.ids) — the purer hold-out. These are passed to main_pytorch via the
# override convention (--train_task_ids / --eval_task_ids beat the env yaml's nulls);
# the id lists must stay UNQUOTED so each id is a separate argv (argparse nargs="+").
SPLIT_JSON="data/splits/tau2_retail_uid_to_tau2id.json"
TRAIN_IDS=$(python3 -c "import json;print(' '.join(json.load(open('$SPLIT_JSON'))['coverage']['covered_tau2_ids']))")
EVAL_IDS=$(python3 -c "import json;print(' '.join(json.load(open('$SPLIT_JSON'))['never_in_logs']['ids']))")
# Cheap-eval plan: during RL we eval on only the FIRST 12 never-in-logs tasks (fast
# early-stop signal); the FULL 42 (EVAL_IDS) are reserved for a final eval-only pass
# on each arm's step_best after the RL matrix finishes.
EVAL_IDS_TRAIN=$(python3 -c "import json;print(' '.join(json.load(open('$SPLIT_JSON'))['never_in_logs']['ids'][:12]))")
[ -n "$TRAIN_IDS" ] && [ -n "$EVAL_IDS" ] && [ -n "$EVAL_IDS_TRAIN" ] || { log "FATAL: could not read task-id split from $SPLIT_JSON"; exit 1; }
log "task split: train=$(echo "$TRAIN_IDS" | wc -w) logged tasks, eval-during-RL=$(echo "$EVAL_IDS_TRAIN" | wc -w) subset, final-eval=$(echo "$EVAL_IDS" | wc -w) never-in-logs hold-out tasks"

[ "$DRY" = 1 ] || wait_gpu_free   # SFT sweep (hide + full) must be done

# 1) Analysis (synced): collect SFT results + commit
if [ "$DRY" != 1 ]; then
  log "collecting SFT analysis -> notes/cc-3.0-retail_sft_results.md"
  uv run --no-sync python scripts/analyze_sft.py act_prm/tau2_retail >> "$MDIR/stage3.log" 2>&1 || log "analyze_sft failed"
  ./scripts/snapshot.sh "cc-3.0 retail SFT results (auto-collected, 6 variants x hide/full) + summary CSV" >> "$MDIR/stage3.log" 2>&1 || true
fi

# 2) Best SFT init per variant — prefer full-context (RL sees full obs), fall back to hide-obs.
# All 6 SFT methods (incl. the _last corpus variants relabeled from the EM step_last ckpt).
declare -A CKPT
for v in actions_only expert_thoughts thoughts_policy thoughts_base thoughts_policy_last thoughts_base_last; do
  c=$(newest "$CKROOT/retail_s2_${v}_heldout_fullctx-*/step_best")
  [ -z "$c" ] && c=$(newest "$CKROOT/retail_s2_${v}_heldout-*/step_best")
  CKPT[$v]="$c"; log "  init[$v] = ${c:-<none>}"
done

# 3) SMOKE: validate the tau2-gym RL path (user sim + tools + evaluator) cheaply
if [ "$DRY" != 1 ]; then
  SMOKE_CK="${CKPT[actions_only]:-}"
  [ -z "$SMOKE_CK" ] && { log "no SFT ckpt found — aborting Stage 3"; exit 1; }
  log "RL SMOKE from $SMOKE_CK (1 batch, 1 task, 4 turns) ..."
  wait_gpu_free
  ./scripts/train_rl_from_sft.sh retail "$SMOKE_CK" --run_tag retail_s3_smoke \
    --num_batches 1 --batch_size 1 --group_size 2 --max_turns 4 --num_test_tasks 1 \
    --no_initial_eval > "$MDIR/s3_smoke.log" 2>&1
  SMOKE_RC=$?
  # metrics.jsonl lives under the LOGS root (not the checkpoints root $S3ROOT).
  # PASS if rc==0 AND (metrics.jsonl exists under $S3LOGROOT OR the log has "Rewards:").
  SMOKE_M=$(newest "$S3LOGROOT/retail_s3_smoke-*/metrics.jsonl")
  if [ "$SMOKE_RC" -ne 0 ] || { [ -z "$SMOKE_M" ] && ! grep -q "Rewards:" "$MDIR/s3_smoke.log" 2>/dev/null; }; then
    log "SMOKE FAILED (rc=$SMOKE_RC, metrics=${SMOKE_M:-none}). tau2-gym RL not validated —"
    log "  NOT launching the RL matrix. Diagnose: $MDIR/s3_smoke.log"
    exit 1
  fi
  log "SMOKE OK (metrics: ${SMOKE_M:-<none, matched Rewards: in log>}) — launching RL matrix"
fi

# 4) RL matrix — 7 arms (airline-aligned retail_rl_* naming), serial + resumable:
#   arm 1: base-direct (raw Qwen3-4B-Instruct, fresh LoRA, NO --resume_from) — RL floor
#   arms 2-7: warm-start from each SFT method's best (fullctx-preferred) step_best.
# INIT[tag]=<sft_ckpt | base>. run_one() resolves + runs (or prints, under DRY) one arm.
declare -A INIT
INIT[retail_rl_base]="base"
INIT[retail_rl_actions_only]="${CKPT[actions_only]:-}"
INIT[retail_rl_expert_thoughts]="${CKPT[expert_thoughts]:-}"
INIT[retail_rl_thoughts_policy]="${CKPT[thoughts_policy]:-}"
INIT[retail_rl_thoughts_base]="${CKPT[thoughts_base]:-}"
INIT[retail_rl_thoughts_policy_last]="${CKPT[thoughts_policy_last]:-}"
INIT[retail_rl_thoughts_base_last]="${CKPT[thoughts_base_last]:-}"

run_one(){
  local tag="$1" init="$2"
  [ -z "$init" ] && { log "RL $tag: no SFT ckpt, skip"; return; }
  [ -n "$(newest "$S3ROOT/${tag}-*/step_best")" ] && { log "RL $tag: done, skip"; return; }
  # $TRAIN_IDS / $EVAL_IDS_TRAIN UNQUOTED on purpose: word-split into one argv per id
  # (argparse nargs="+"). Trains on the 72 logged, evals (during RL) on the 12-task
  # subset — the cheap early-stop signal. Full-42 eval happens after the matrix.
  if [ "$DRY" = 1 ]; then
    echo "DRY $tag: ./scripts/train_rl_from_sft.sh retail $init --run_tag $tag --train_task_ids $TRAIN_IDS --eval_task_ids $EVAL_IDS_TRAIN"
    return
  fi
  log "RL $tag from $init ..."
  wait_gpu_free
  ./scripts/train_rl_from_sft.sh retail "$init" --run_tag "$tag" \
    --train_task_ids $TRAIN_IDS --eval_task_ids $EVAL_IDS_TRAIN > "$MDIR/s3_${tag}.log" 2>&1 \
    && log "RL $tag: done" || log "RL $tag: FAILED (see $MDIR/s3_${tag}.log)"
  ./scripts/snapshot.sh "Stage-3 RL: $tag (retail) checkpoint" >/dev/null 2>&1 || true
}

for tag in retail_rl_base retail_rl_actions_only retail_rl_expert_thoughts \
           retail_rl_thoughts_policy retail_rl_thoughts_base \
           retail_rl_thoughts_policy_last retail_rl_thoughts_base_last; do
  run_one "$tag" "${INIT[$tag]}"
done

# 5) FINAL FULL-42 EVAL — after the RL matrix, evaluate each arm's RL step_best on
# ALL 42 never-in-logs hold-out tasks (the true hold-out reward). Eval-only:
# --no_train + --num_batches 1 --eval_every 1 makes the single batch the last step,
# which forces exactly one eval on the 42-task hold-out with the resumed step_best
# (no PG, no training). Resumable: skip an arm whose eval42 metrics already exist.
run_eval42(){
  local tag="$1"
  local ckpt; ckpt=$(newest "$S3ROOT/${tag}-*/step_best")
  [ -z "$ckpt" ] && { log "EVAL42 $tag: no RL step_best, skip"; return; }
  [ -n "$(newest "$S3LOGROOT/${tag}_eval42-*/metrics.jsonl")" ] && { log "EVAL42 $tag: done, skip"; return; }
  if [ "$DRY" = 1 ]; then
    echo "DRY $tag: ./scripts/train_rl_from_sft.sh retail $ckpt --run_tag ${tag}_eval42 --no_train --num_batches 1 --eval_every 1 --train_task_ids $TRAIN_IDS --eval_task_ids $EVAL_IDS"
    return
  fi
  log "EVAL42 $tag from $ckpt (full 42-task hold-out) ..."
  wait_gpu_free
  ./scripts/train_rl_from_sft.sh retail "$ckpt" --run_tag "${tag}_eval42" \
    --no_train --num_batches 1 --eval_every 1 \
    --train_task_ids $TRAIN_IDS --eval_task_ids $EVAL_IDS > "$MDIR/s3_${tag}_eval42.log" 2>&1 \
    && log "EVAL42 $tag: done" || log "EVAL42 $tag: FAILED (see $MDIR/s3_${tag}_eval42.log)"
  ./scripts/snapshot.sh "Stage-3 RL: $tag full-42 eval (retail)" >/dev/null 2>&1 || true
}

for tag in retail_rl_base retail_rl_actions_only retail_rl_expert_thoughts \
           retail_rl_thoughts_policy retail_rl_thoughts_base \
           retail_rl_thoughts_policy_last retail_rl_thoughts_base_last; do
  run_eval42 "$tag"
done
log "=== Stage-3 driver done ==="
