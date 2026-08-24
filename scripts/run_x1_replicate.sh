#!/usr/bin/env bash
# Discriminating test: is the airline thoughts_policy x1 result (72.2%) a lucky draw, or
# does the BATCHED eval_group_size>1 path depress scores?
#
# The problem. At 1 rollout/task airline thoughts_policy scored 13/18 = 72.2%. At 3
# rollouts/task the same checkpoint on the same 18 tasks scored 10, 7, 9 (55.6 / 38.9 /
# 50.0, mean 48.1) -- x1 sits ~2.9 SD above its own x3 samples, about a 0.4% event if they
# are draws from one distribution. All three x3 streams also used more tool calls (7.5-9.2)
# than x1 did (5.8). Ruled out already: batch truncation (0 warnings), differing
# checkpoints/task sets/turn caps (identical), and a positional padding artifact (gen_id
# 0/1/2 scored 10/7/9 -- scatter, not a trend).
#
# The test. Re-run the SAME checkpoint on the SAME tasks at eval_group_size=1, i.e. an
# independent second draw of the original protocol, twice for a little resolution.
#   ~48-55%  -> 72.2% was a lucky draw; the 3-rollout null stands.
#   ~72%     -> the batched path is depressing x3 and the null is an artifact of it.
#
# Cheap: airline x1 measured at ~38 min per run.
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/usr/local/bin:$PATH"
MODEL="${MODEL_CFG:-hf_qwen3_4b_instruct}"; export MODEL_CFG="$MODEL"
MDIR="${MDIR:-/tmp/aprm/x1replicate}"; mkdir -p "$MDIR"
_fail=0
REPS="${REPS:-2}"
ARMS="${ARMS:-thoughts_policy actions_only}"
DOM="${DOM:-airline}"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$MDIR/x1replicate.log"; }
wait_gpu_free(){ while pgrep -f '[m]ain_pytorch.py' >/dev/null 2>&1; do sleep 60; done; sleep 10; }
newest(){ ls -dt $1 2>/dev/null | head -1; }

IDS=$(python3 -c "import json;print(' '.join(str(i) for i in json.load(open('data/splits/tau2_airline_taskmap.json'))['unseen_tau2_ids']))")
TID=$(python3 -c "import json;print(json.load(open('data/splits/tau2_airline_taskmap.json'))['covered_tau2_ids'][0])")

log "=== x1 replicate: dom=$DOM arms='$ARMS' reps=$REPS (eval_group_size=1, the ORIGINAL protocol) ==="
for arm in $ARMS; do
  CK=$(newest "checkpoints_lora/act_prm_tau2_${DOM}/$MODEL/${DOM}_s2_${arm}_lr3e_3_nb150_heldout-*/step_best")
  [ -z "$CK" ] && { log "$DOM/$arm: no checkpoint, skip"; continue; }
  for rep in $(seq 1 "$REPS"); do
    TAG="${DOM}_rollout_${arm}_x1rep${rep}"
    [ -f "$MDIR/${TAG}.done" ] && { log "$TAG: done, skip"; continue; }
    log "ROLLOUT $TAG ($(echo $IDS|wc -w) tasks x 1) <- $CK"
    wait_gpu_free
    ./scripts/train_rl_from_sft.sh "$DOM" "$CK" --run_tag "$TAG" \
        --generator_config hf_rlvr --env_config "tau2bench/${DOM}_rlvr" \
        --no_train --num_batches 1 --eval_every 1 --group_size 2 --batch_size 1 \
        --eval_group_size 1 \
        --max_turns 20 --max_tokens 2048 --discount_factor 1.0 --hide_observations \
        --train_task_ids "$TID" --eval_task_ids $IDS > "$MDIR/${TAG}.log" 2>&1 \
      || { _fail=$((_fail+1)); log "$TAG: FAILED"; continue; }
    D=$(newest "logs/tau2bench_${DOM}_rlvr/$MODEL/${TAG}-*/")
    uv run --no-project python scripts/check_rollout_valid.py "$D" \
      && { touch "$MDIR/${TAG}.done"; log "$TAG: done"; } \
      || { _fail=$((_fail+1)); log "$TAG: INVALID; not marking done"; }
  done
done
if [ "${_fail:-0}" -eq 0 ]; then
  touch "$MDIR/ALLDONE"; log "=== x1 replicate complete ==="
else
  log "=== x1 replicate INCOMPLETE: $_fail failure(s); not marking ALLDONE ==="
fi
