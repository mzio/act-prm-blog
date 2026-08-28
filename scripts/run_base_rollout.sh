#!/usr/bin/env bash
# BASE-MODEL rollout eval: the missing control.
#
# Every rollout eval so far loads a Stage-2 LoRA (--resume_from). None measures what the
# raw instruct model does in the gym, so we can compare the arms to each other but cannot
# say what Stage-2 SFT bought in the first place -- and retail's 11.9% baseline is low
# enough that a reader will reasonably ask whether behavioural cloning helped at all.
#
# "Base" here = base weights + a FRESH r8/a16 LoRA and no --resume_from. Because lora_B is
# zero-initialised, B@A == 0, so the adapter is an exact no-op and the policy is bitwise the
# raw Qwen3-4B-Instruct-2507. Going through train_rl_from_sft.sh's base|none|direct mode
# (rather than dropping --lora_config) keeps every other knob identical to the SFT-arm
# rollouts, so the only difference is the adapter.
#
# Same hold-out task sets, same hide-observations regime, same max_turns/max_tokens and the
# same --no_train eval-only mode as scripts/run_sft_rollout_eval.sh.
#
# Usage: ./scripts/run_base_rollout.sh          (cron drives it via sweep_guard.sh)
#        NTRIES=3 ./scripts/run_base_rollout.sh (match the multi-rollout stage)
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/usr/local/bin:$PATH"
MODEL="${MODEL_CFG:-hf_qwen3_4b_instruct}"; export MODEL_CFG="$MODEL"
MDIR="${MDIR:-/tmp/aprm/base_rollout}"; mkdir -p "$MDIR"
_fail=0
NTRIES="${NTRIES:-1}"
SFX=""; [ "$NTRIES" != 1 ] && SFX="_x${NTRIES}"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$MDIR/base_rollout.log"; }
wait_gpu_free(){ while pgrep -f '[m]ain_pytorch.py' >/dev/null 2>&1; do sleep 60; done; sleep 10; }
newest(){ ls -dt $1 2>/dev/null | head -1; }

RETAIL_IDS=$(python3 -c "import json;print(' '.join(json.load(open('data/splits/tau2_retail_uid_to_tau2id.json'))['never_in_logs']['ids']))")
RETAIL_TID=$(python3 -c "import json;print(json.load(open('data/splits/tau2_retail_uid_to_tau2id.json'))['coverage']['covered_tau2_ids'][0])")
AIR_IDS=$(python3 -c "import json;print(' '.join(str(i) for i in json.load(open('data/splits/tau2_airline_taskmap.json'))['unseen_tau2_ids']))")
AIR_TID=$(python3 -c "import json;print(json.load(open('data/splits/tau2_airline_taskmap.json'))['covered_tau2_ids'][0])")

log "=== base-model rollout (no Stage-2 adapter), ${NTRIES} rollout(s)/task ==="
for dom in retail airline; do
  IDS=$RETAIL_IDS; TID=$RETAIL_TID
  [ "$dom" = airline ] && { IDS=$AIR_IDS; TID=$AIR_TID; }
  TAG="${dom}_rollout_base${SFX}"
  [ -f "$MDIR/${TAG}.done" ] && { log "$TAG: done, skip"; continue; }
  log "ROLLOUT $TAG ($(echo $IDS|wc -w) tasks x $NTRIES)"
  wait_gpu_free
  ./scripts/train_rl_from_sft.sh "$dom" base --run_tag "$TAG" \
      --generator_config hf_rlvr --env_config "tau2bench/${dom}_rlvr" \
      --no_train --num_batches 1 --eval_every 1 --group_size 2 --batch_size 1 \
      --eval_group_size "$NTRIES" \
      --max_turns 20 --max_tokens 2048 --discount_factor 1.0 --hide_observations \
      --train_task_ids "$TID" --eval_task_ids $IDS > "$MDIR/${TAG}.log" 2>&1 \
    || { _fail=$((_fail+1)); log "$TAG: FAILED (see $MDIR/${TAG}.log)"; continue; }
  D=$(newest "logs/tau2bench_${dom}_rlvr/$MODEL/${TAG}-*/")
  uv run --no-project python scripts/check_rollout_valid.py "$D" \
    && { touch "$MDIR/${TAG}.done"; log "$TAG: done"; } \
    || { _fail=$((_fail+1)); log "$TAG: INVALID; not marking done"; }
done
if [ "${_fail:-0}" -eq 0 ]; then
  touch "$MDIR/ALLDONE"; log "=== base rollout complete ==="
else
  log "=== base rollout INCOMPLETE: $_fail failure(s); not marking ALLDONE ==="
fi
