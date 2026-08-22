#!/usr/bin/env bash
# Finance Stage-2 SFT, v3: QUESTION-level split, 40 train / 10 eval questions.
#
# Only 50 of the 79 finqa_reasoning questions have a SUCCESSFUL expert rollout (the split's
# "usable" filter is done + reward>0). The other 29 have trajectories, but ALL reward=0 --
# questions GPT-5-mini failed. The v1 split partitioned uids, and ~2.5 uids share each
# question, so questions leaked across splits (76% of eval, 91% of rl_eval).
#
# v3 partitions the 50 demo-bearing QUESTIONS 40/10 and subselects each arm's existing pool
# by question. No Stage-1 relabel and no new generation. Verified zero train/eval question
# overlap on all four arms.
#   train  40 questions: 116 traj / 1347 steps (thought arms 122 / 1409)
#   eval   10 questions:  25 traj /  363 steps (thought arms  27 /  398)
#
# Downstream evals this enables:
#   - teacher-forced curves + rollout on the 10 eval questions (fair: expert solved these)
#   - rollout on the 29 expert-FAILURE questions as a labelled hard test
#
# Distinct run tags (<variant>_v3) so nothing overwrites v1. hide-obs, lr 3e-3, 150 batches.
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/usr/local/bin:$PATH"
MODEL="${MODEL_CFG:-hf_qwen3_4b_instruct}"; export MODEL_CFG="$MODEL"
MDIR=/tmp/aprm/finance_v3; mkdir -p "$MDIR"
NB="${NUM_BATCHES:-150}"
ENV=act_prm/snorkel_finance_split
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$MDIR/finance_v3.log"; }
wait_gpu_free(){ while pgrep -f '[m]ain_pytorch.py' >/dev/null 2>&1; do sleep 60; done; sleep 10; }

# variant : sft_variant : clean pool : extra flags
ARMS=(
  "actions_only:actions_only:data/snorkel_finance_split_v3:"
  "expert_thoughts:expert_thoughts:data/snorkel_finance_split_expert_thoughts_v3:"
  "expert_thoughts_all:expert_thoughts_all:data/snorkel_finance_split_expert_thoughts_v3:"
  "thoughts_policy:thoughts_policy:data/sft_corpus/snorkel_finance_split/policy_v3:"
  "thoughts_base:thoughts_base:data/sft_corpus/snorkel_finance_split/base_v3:"
)

for spec in "${ARMS[@]}"; do
  IFS=":" read -r label variant pool extra <<< "$spec"
  TAG="snorkel_finance_split_s2_${label}_v3_lr3e_3_nb${NB}_heldout"
  if [ -f "$MDIR/${TAG}.done" ]; then log "$TAG: done, skip"; continue; fi
  log "SFT $TAG  (train: original pool, eval: clean 5 questions)"
  wait_gpu_free
  ./scripts/train_sft.sh "$ENV" "$variant" \
      --dataset_path "$pool" --run_tag "$TAG" --best_metric eval_action_ppl \
      --learning_rate 3e-3 --num_batches "$NB" --early_stop_patience 3 $extra \
      > "$MDIR/${TAG}.log" 2>&1 \
    && { touch "$MDIR/${TAG}.done"; log "$TAG: done"; } || log "$TAG: FAILED (see $MDIR/${TAG}.log)"
done
touch "$MDIR/ALLDONE"; log "=== finance v3 Stage-2 complete ==="
