#!/usr/bin/env bash
# CONTROL: retrain retail thoughts_policy on a volume-MATCHED corpus, then rollout-eval it.
#
# The retail thought corpora carry 80 trajectories / 1043 assistant steps against the
# actions_only and expert_thoughts pools' 49 / 635 -- the Stage-1 relabel emitted up to two
# generations per task and the export kept both. So the retail Act-PRM advantage has an
# alternative explanation: more supervision, not better supervision.
#
# policy_1gen keeps one generation per uid (48 traj / 627 steps), matching the baseline.
# If the advantage survives it is the thoughts; if it collapses it was the data.
#
# NB airline needs no control -- its three arms are already matched at 21 traj / 221 steps,
# and airline is where the effect was LARGEST (+22.2pp), so the confound cannot explain
# the headline. This isolates the retail half.
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/usr/local/bin:$PATH"
MDIR=/tmp/aprm/control; mkdir -p "$MDIR"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$MDIR/control.log"; }
wait_gpu_free(){ while pgrep -f '[m]ain_pytorch.py' >/dev/null 2>&1; do sleep 60; done; sleep 10; }
newest(){ ls -dt $1 2>/dev/null | head -1; }

TAG=retail_s2_thoughts_policy_1gen_lr3e_3_nb150_heldout
# --- 1. SFT on the matched corpus (same hyperparameters as the arm it controls) ---
if [ ! -f "$MDIR/sft.done" ]; then
  log "SFT $TAG on the volume-matched corpus (48 traj / 627 steps)"
  wait_gpu_free
  ./scripts/train_sft.sh act_prm/tau2_retail thoughts_policy \
      --dataset_path data/sft_corpus/tau2_retail/policy_1gen \
      --run_tag "$TAG" --best_metric eval_action_ppl \
      --learning_rate 3e-3 --num_batches 150 --early_stop_patience 3 \
      > "$MDIR/sft.log" 2>&1 \
    && { touch "$MDIR/sft.done"; log "SFT: done"; } || { log "SFT: FAILED (see $MDIR/sft.log)"; exit 1; }
fi

# --- 2. rollout eval on the same 42 never-in-logs tasks ---
if [ ! -f "$MDIR/rollout.done" ]; then
  CK=$(newest "checkpoints_lora/act_prm_tau2_retail/hf_qwen3_4b_instruct/${TAG}-*/step_best")
  [ -z "$CK" ] && { log "no checkpoint for $TAG"; exit 1; }
  IDS=$(python3 -c "import json;print(' '.join(json.load(open('data/splits/tau2_retail_uid_to_tau2id.json'))['never_in_logs']['ids']))")
  TID=$(python3 -c "import json;print(json.load(open('data/splits/tau2_retail_uid_to_tau2id.json'))['coverage']['covered_tau2_ids'][0])")
  log "ROLLOUT retail/thoughts_policy_1gen ($(echo $IDS | wc -w) tasks) <- $CK"
  wait_gpu_free
  ./scripts/train_rl_from_sft.sh retail "$CK" --run_tag retail_rollout_thoughts_policy_1gen_lr3e_3 \
      --generator_config hf_rlvr --env_config tau2bench/retail_rlvr \
      --no_train --num_batches 1 --eval_every 1 --group_size 2 --batch_size 1 \
      --max_turns 20 --max_tokens 2048 --discount_factor 1.0 --hide_observations \
      --train_task_ids "$TID" --eval_task_ids $IDS \
      > "$MDIR/rollout.log" 2>&1 \
    || { log "ROLLOUT: FAILED (see $MDIR/rollout.log)"; exit 1; }
  D=$(newest "logs/tau2bench_retail_rlvr/hf_qwen3_4b_instruct/retail_rollout_thoughts_policy_1gen_lr3e_3-*/")
  if uv run --no-project python scripts/check_rollout_valid.py "$D"; then
    touch "$MDIR/rollout.done"; log "ROLLOUT: done"
  else
    log "ROLLOUT: INVALID (user-sim/auth failure); not marking done"; exit 1
  fi
fi
touch "$MDIR/ALLDONE"; log "=== matched control complete ==="
