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
MDIR=/tmp/aprm/stage3; mkdir -p "$MDIR"
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
[ -n "$TRAIN_IDS" ] && [ -n "$EVAL_IDS" ] || { log "FATAL: could not read task-id split from $SPLIT_JSON"; exit 1; }
log "task split: train=$(echo "$TRAIN_IDS" | wc -w) logged tasks, eval=$(echo "$EVAL_IDS" | wc -w) never-in-logs hold-out tasks"

wait_gpu_free   # SFT sweep (hide + full) must be done

# 1) Analysis (synced): collect SFT results + commit
log "collecting SFT analysis -> notes/cc-3.0-retail_sft_results.md"
uv run --no-sync python scripts/analyze_sft.py act_prm/tau2_retail >> "$MDIR/stage3.log" 2>&1 || log "analyze_sft failed"
./scripts/snapshot.sh "cc-3.0 retail SFT results (auto-collected, 4 variants x hide/full) + summary CSV" >> "$MDIR/stage3.log" 2>&1 || true

# 2) Best SFT init per variant — prefer full-context (RL sees full obs), fall back to hide-obs
declare -A CKPT
for v in actions_only expert_thoughts thoughts_policy thoughts_base thoughts_policy_last thoughts_base_last; do
  c=$(newest "$CKROOT/retail_s2_${v}_heldout_fullctx-*/step_best")
  [ -z "$c" ] && c=$(newest "$CKROOT/retail_s2_${v}_heldout-*/step_best")
  CKPT[$v]="$c"; log "  init[$v] = ${c:-<none>}"
done

# 3) SMOKE: validate the tau2-gym RL path (user sim + tools + evaluator) cheaply
SMOKE_CK="${CKPT[actions_only]:-}"
[ -z "$SMOKE_CK" ] && { log "no SFT ckpt found — aborting Stage 3"; exit 1; }
log "RL SMOKE from $SMOKE_CK (1 batch, 1 task, 4 turns) ..."
wait_gpu_free
./scripts/train_rl_from_sft.sh retail "$SMOKE_CK" --run_tag retail_s3_smoke \
  --num_batches 1 --batch_size 1 --group_size 2 --max_turns 4 --num_test_tasks 1 \
  --no_initial_eval > "$MDIR/s3_smoke.log" 2>&1
SMOKE_RC=$?
SMOKE_M=$(newest "$S3ROOT/retail_s3_smoke-*/metrics.jsonl")
if [ "$SMOKE_RC" -ne 0 ] || [ -z "$SMOKE_M" ]; then
  log "SMOKE FAILED (rc=$SMOKE_RC, metrics=${SMOKE_M:-none}). tau2-gym RL not validated —"
  log "  NOT launching the RL matrix. Diagnose: $MDIR/s3_smoke.log"
  exit 1
fi
log "SMOKE OK (metrics: $SMOKE_M) — launching RL matrix"

# 4) RL from each variant's best SFT checkpoint (serial, resumable)
for v in actions_only expert_thoughts thoughts_policy thoughts_base thoughts_policy_last thoughts_base_last; do
  ck="${CKPT[$v]:-}"; [ -z "$ck" ] && { log "RL $v: no ckpt, skip"; continue; }
  tag="retail_s3_${v}_fullctx"
  [ -n "$(newest "$S3ROOT/${tag}-*/step_best")" ] && { log "RL $v: done, skip"; continue; }
  log "RL $v from $ck ..."
  wait_gpu_free
  # $TRAIN_IDS / $EVAL_IDS UNQUOTED on purpose: word-split into one argv per id
  # (argparse nargs="+"). Trains on the 72 logged, evals on the 42 never-in-logs.
  ./scripts/train_rl_from_sft.sh retail "$ck" --run_tag "$tag" \
    --train_task_ids $TRAIN_IDS --eval_task_ids $EVAL_IDS > "$MDIR/s3_${v}.log" 2>&1 \
    && log "RL $v: done" || log "RL $v: FAILED (see $MDIR/s3_${v}.log)"
  ./scripts/snapshot.sh "Stage-3 RL: $v (retail) checkpoint" >/dev/null 2>&1 || true
done
log "=== Stage-3 driver done ==="
