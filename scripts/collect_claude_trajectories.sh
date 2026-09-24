#!/usr/bin/env bash
# Collect full thought+action expert trajectories with a Claude teacher.
#
#   DOMAIN=insurance ./scripts/collect_claude_trajectories.sh            # all 261 tasks
#   DOMAIN=retail SHARDS=4 SHARD=0 ./scripts/collect_claude_trajectories.sh
#   N_PILOT=8 ./scripts/collect_claude_trajectories.sh                   # small pilot
#
# WHY THIS EXISTS: the logged GPT-5-mini corpora carry a thought on only 50-64% of turns,
# and they only cover the tasks that happened to get logged. A Claude teacher fixes both.
# Measured on an 8-task insurance pilot (2026-09-23, claude-sonnet-4-6, thinking disabled):
# 27/32 attempts succeeded, 8/8 task coverage, and 193/193 tool calls carried a thought.
#
# FULL TASK SETS -- every task the gym has, not just the ones with logged demos:
#   insurance 261   retail 114 (72 logged + 42 never-in-logs)   airline 50 (32 + 18 unseen)
# Splits are deliberately NOT applied here: collect everything, then carve train/eval from
# the result (eval tasks reserved for rollouts; within train, split by state-action step
# for PPL / action-accuracy). `split` in the output is just bookkeeping.
#
# ATTEMPTS=3 by default. The pilot showed 1 attempt covered 7/8 tasks and 2 covered 8/8
# (per-attempt success 84%), so 4 was over-provisioned; 3 buys the tail without doubling
# cost. The exporter lists any task that got zero successes, for a short retry pass.
#
# WHY THE ODD FLAGS:
#   CLAUDECODE=   the SDK refuses to spawn Claude Code inside a nested session
#   --is_async    `--is_async` is store_true with NO default=None, so update_configs
#                 (main_pytorch.py:84) overwrites the env yaml's `is_async: true` with
#                 False on EVERY run; without it the sync env loads and the generator
#                 dies on `reset_async`. Fixing the flag globally would change behaviour
#                 for every other run, so force it here.
#   4b_cpu        main_pytorch always loads a policy, but the Claude teacher never runs
#                 it -- only its tokenizer, for length accounting. Loading it on CPU means
#                 shards cost ZERO VRAM and can run
#                 beside a training job.
#   .venv-tau2    the only venv with claude-agent-sdk installed
set -uo pipefail
cd /home/mzio/projects/act-prm-blog
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/usr/local/bin:$PATH"
export PYTHONUNBUFFERED=1 HF_HOME=/data/users/mzio/models/hf_cache HF_HUB_OFFLINE=1
export WANDB_MODE=offline ACT_PRM_DUMP_TRAJECTORIES=1

DOMAIN="${DOMAIN:-insurance}"
ATTEMPTS="${ATTEMPTS:-3}"
GEN_CFG="${GEN_CFG:-claude_teacher_s46}"
MODEL_CFG="${MODEL_CFG:-hf_qwen3_4b_instruct_cpu}"
SHARDS="${SHARDS:-1}"; SHARD="${SHARD:-0}"
N_PILOT="${N_PILOT:-0}"

case "$DOMAIN" in
  insurance) ENVC="act_prm/snorkel_insurance_gym" ;;
  retail)    ENVC="tau2bench/retail_rlvr" ;;
  airline)   ENVC="tau2bench/airline_rlvr" ;;
  *) echo "unknown DOMAIN '$DOMAIN' (insurance|retail|airline)"; exit 1 ;;
esac

# Full native task set per domain, then the shard slice.
IDS=$(python3 - "$DOMAIN" "$SHARDS" "$SHARD" "$N_PILOT" <<'PY'
import json, sys
dom, shards, shard, npilot = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
if dom == "insurance":
    m = json.load(open("data/splits/snorkel_insurance_uid_to_task.json"))
    ids = set(map(str, m["train_task_ids"])) | set(map(str, m["eval_task_ids"])) | set(map(str, m["rollout_task_ids"]))
elif dom == "retail":
    r = json.load(open("data/splits/tau2_retail_uid_to_tau2id.json"))
    ids = set(map(str, r["coverage"]["covered_tau2_ids"])) | set(map(str, r["never_in_logs"]["ids"]))
else:
    a = json.load(open("data/splits/tau2_airline_taskmap.json"))
    ids = set(map(str, a["covered_tau2_ids"])) | set(map(str, a["unseen_tau2_ids"]))
ids = sorted(ids, key=int)
if npilot > 0:
    ids = ids[:npilot]
# Stride-based sharding so every shard gets a comparable mix of easy/hard tasks; a
# contiguous split would hand one shard a whole difficulty band.
print(" ".join(ids[shard::shards]))
PY
)
N=$(echo $IDS | wc -w)
[ "$N" -eq 0 ] && { echo "no tasks selected"; exit 1; }

SUF=""; [ "$SHARDS" -gt 1 ] && SUF="_sh${SHARD}of${SHARDS}"
[ "$N_PILOT" -gt 0 ] && SUF="${SUF}_pilot${N_PILOT}"
TAG="${TAG:-${DOMAIN}_teacher${SUF}}"
mkdir -p runlogs/claude_teacher          # NOT /tmp: reaping has eaten run logs twice
LOG="runlogs/claude_teacher/${TAG}.log"

echo "[$(date '+%m-%d %H:%M:%S')] $TAG: $N tasks x $ATTEMPTS attempts ($((N*ATTEMPTS)) episodes)"
echo "  env=$ENVC generator=$GEN_CFG model=$MODEL_CFG log=$LOG"

# All tasks go in train_task_ids (batch_size must cover them, else only the first
# batch_size roll out and the rest silently contribute nothing). eval_task_ids gets one
# task purely so the eval env builds; its episodes are duplicates and cost ATTEMPTS extra.
EVAL_ONE=$(echo $IDS | awk '{print $1}')
CLAUDECODE= UV_PROJECT_ENVIRONMENT=.venv-tau2 uv run --no-sync python main_pytorch.py \
    --env_config "$ENVC" --model_config "$MODEL_CFG" \
    --lora_config r8_a16_linear --generator_config "$GEN_CFG" --trainer_config pg \
    --replay_buffer_config default --no_train --is_async \
    --num_batches 1 --eval_every 1 \
    --group_size "$ATTEMPTS" --eval_group_size "$ATTEMPTS" \
    --batch_size "$N" --max_tokens 4096 \
    --train_task_ids $IDS --eval_task_ids "$EVAL_ONE" \
    --run_tag "$TAG" --verbose >>"$LOG" 2>&1
echo "[$(date '+%m-%d %H:%M:%S')] rc=$?"

D=$(ls -dt logs/*/*/${TAG}-*/ 2>/dev/null | head -1)
if [ -n "$D" ] && [ -f "$D/trajectories.jsonl" ]; then
  echo "  $(wc -l < "$D/trajectories.jsonl") trajectories"
  uv run --no-sync python scripts/export_teacher_corpus.py \
      --runs "${D}" --out "data/sft_corpus/${DOMAIN}_claude/${TAG}"
else
  echo "  no trajectories.jsonl produced -- see $LOG"
fi
