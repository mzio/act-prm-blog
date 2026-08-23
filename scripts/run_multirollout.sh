#!/usr/bin/env bash
# 3 rollouts per task for the key hide-regime arms -- the only thing that can turn the
# headline rollout result from "suggestive" into significant.
#
# Every rollout eval so far ran ONE attempt per task (eval_group_size: 1 in
# configs/trainer/pg.yaml). Paired McNemar on those: retail p=0.289, airline p=0.289,
# pooled p=0.077. More ARMS cannot fix that; only more rollouts of the arms we have.
# --eval_group_size 3 makes the generator produce 3 independent rollouts per task in a
# single pass, so n goes 42 -> 126 (retail) and 18 -> 54 (airline).
#
# MEASURED NOISE FLOOR (08-23). Two runs of expert_thoughts_all with identical config, data
# and hyperparameters -- differing only in GPU reduction nondeterminism, both selecting
# step_best=149, train losses agreeing to 3-4 decimals, adapters differing by a median 0.03%
# per tensor -- scored 9/42 and 6/42 on the SAME 42 tasks. A 7.1pp swing, agreeing on only
# 35/42 tasks. That is the reproducibility floor of the 1-rollout protocol, and it is the
# same size as the effects being reported (+9.5pp retail thoughts_policy, +7.1pp the matched
# control, +4.8pp thoughts_base). Only airline thoughts_policy (+22.2pp) clearly clears it.
#
# That swing conflates TWO sources: rollout sampling (stochastic generation + a stochastic
# LLM user simulator) and training nondeterminism. --eval_group_size only attacks the first.
# This run therefore also DECOMPOSES the noise: if the within-adapter spread across 3
# rollouts is ~7pp, sampling dominates and more rollouts is the right fix; if it is much
# smaller, training nondeterminism dominates and the fix is multiple training SEEDS instead.
#
# ANALYSIS: with 3 rollouts/task each task gets a success FRACTION (0, 1/3, 2/3, 1). Compare
# arms with a paired test on those continuous per-task scores (Wilcoxon signed-rank), not by
# binarising and running McNemar -- binarising throws away exactly the resolution this run
# was launched to buy.
#
# Measured cost: retail 74 min/arm at 1 try -> ~3.7h at 3; airline 38 min -> ~1.9h.
# ARMS default is the key contrast (baseline vs Act-PRM policy-scored) = ~11h.
# Set ARMS="actions_only expert_thoughts thoughts_policy thoughts_base" for all four (~22h).
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/usr/local/bin:$PATH"
MODEL="${MODEL_CFG:-hf_qwen3_4b_instruct}"; export MODEL_CFG="$MODEL"
MDIR=/tmp/aprm/multirollout; mkdir -p "$MDIR"
_fail=0
NTRIES="${NTRIES:-3}"
ARMS="${ARMS:-actions_only thoughts_policy}"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$MDIR/multirollout.log"; }
wait_gpu_free(){ while pgrep -f '[m]ain_pytorch.py' >/dev/null 2>&1; do sleep 60; done; sleep 10; }
newest(){ ls -dt $1 2>/dev/null | head -1; }

RETAIL_IDS=$(python3 -c "import json;print(' '.join(json.load(open('data/splits/tau2_retail_uid_to_tau2id.json'))['never_in_logs']['ids']))")
RETAIL_TID=$(python3 -c "import json;print(json.load(open('data/splits/tau2_retail_uid_to_tau2id.json'))['coverage']['covered_tau2_ids'][0])")
AIR_IDS=$(python3 -c "import json;print(' '.join(str(i) for i in json.load(open('data/splits/tau2_airline_taskmap.json'))['unseen_tau2_ids']))")
AIR_TID=$(python3 -c "import json;print(json.load(open('data/splits/tau2_airline_taskmap.json'))['covered_tau2_ids'][0])")

log "=== multi-rollout: ${NTRIES} per task, arms: $ARMS ==="
for dom in retail airline; do
  IDS=$RETAIL_IDS; TID=$RETAIL_TID
  [ "$dom" = airline ] && { IDS=$AIR_IDS; TID=$AIR_TID; }
  for arm in $ARMS; do
    CK=$(newest "checkpoints_lora/act_prm_tau2_${dom}/$MODEL/${dom}_s2_${arm}_lr3e_3_nb150_heldout-*/step_best")
    [ -z "$CK" ] && { log "$dom/$arm: no checkpoint, skip"; continue; }
    TAG="${dom}_rollout_${arm}_x${NTRIES}"
    [ -f "$MDIR/${TAG}.done" ] && { log "$TAG: done, skip"; continue; }
    log "ROLLOUT $TAG ($(echo $IDS|wc -w) tasks x $NTRIES)"
    wait_gpu_free
    ./scripts/train_rl_from_sft.sh "$dom" "$CK" --run_tag "$TAG" \
        --generator_config hf_rlvr --env_config "tau2bench/${dom}_rlvr" \
        --no_train --num_batches 1 --eval_every 1 --group_size 2 --batch_size 1 \
        --eval_group_size "$NTRIES" \
        --max_turns 20 --max_tokens 2048 --discount_factor 1.0 --hide_observations \
        --train_task_ids "$TID" --eval_task_ids $IDS > "$MDIR/${TAG}.log" 2>&1 \
      || { { _fail=$((_fail+1)); log "${TAG}: FAILED"; }; continue; }
    D=$(newest "logs/tau2bench_${dom}_rlvr/$MODEL/${TAG}-*/")
    uv run --no-project python scripts/check_rollout_valid.py "$D" \
      && { touch "$MDIR/${TAG}.done"; log "$TAG: done"; } \
      || { _fail=$((_fail+1)); log "$TAG: INVALID; not marking done"; }
  done
done
# Only claim completion if nothing failed. Touching ALLDONE unconditionally is how a
# wholesale failure (10/10 finance rollouts dying on HF DNS, 08-23) got recorded as a
# completed stage and the guard advanced straight past it.
if [ "${_fail:-0}" -eq 0 ]; then
  touch "$MDIR/ALLDONE"; log "=== multi-rollout complete ==="
else
  log "=== multi-rollout INCOMPLETE: $_fail failure(s); not marking ALLDONE ==="
fi
