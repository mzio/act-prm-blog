#!/usr/bin/env bash
# Finance Stage-2 SFT, v2: same training data, CLEAN eval.
#
# The v1 finance eval was 76% contaminated -- the aprm split partitioned uids, but many
# uids map to the same finqa_reasoning question, so 16 of act_prm_eval's 21 questions also
# sat in act_prm_train. Every finance Stage-2 curve and final number was therefore scored
# largely on memorised questions.
#
# No retraining of the DATA is needed: the training pools are untouched. Only the eval the
# trainer scores each eval_every is swapped, onto the *_cleaneval pools which keep the
# original train.json and an eval.json filtered to questions never in act_prm_train
# (5 questions / 7 trajectories / 127 assistant steps). That yields honest train and eval
# action-span PPL/accuracy CURVES, plus a final checkpoint for rollout eval.
#
# Distinct run tags (<variant>_v2) so nothing overwrites the v1 runs -- those stay on disk
# as the contaminated-baseline record.
#
# hide-obs, lr 3e-3, 150 batches: identical to every other Stage-2 arm.
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/usr/local/bin:$PATH"
MODEL="${MODEL_CFG:-hf_qwen3_4b_instruct}"; export MODEL_CFG="$MODEL"
MDIR=/tmp/aprm/finance_v2; mkdir -p "$MDIR"
NB="${NUM_BATCHES:-150}"
ENV=act_prm/snorkel_finance_split
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$MDIR/finance_v2.log"; }
wait_gpu_free(){ while pgrep -f '[m]ain_pytorch.py' >/dev/null 2>&1; do sleep 60; done; sleep 10; }

# variant : sft_variant : clean pool : extra flags
ARMS=(
  "actions_only:actions_only:data/snorkel_finance_split_cleaneval:"
  "expert_thoughts:expert_thoughts:data/snorkel_finance_split_expert_thoughts_cleaneval:"
  "expert_thoughts_all:expert_thoughts_all:data/snorkel_finance_split_expert_thoughts_cleaneval:"
  "thoughts_policy:thoughts_policy:data/sft_corpus/snorkel_finance_split/policy_cleaneval:"
  "thoughts_base:thoughts_base:data/sft_corpus/snorkel_finance_split/base_cleaneval:"
)

for spec in "${ARMS[@]}"; do
  IFS=":" read -r label variant pool extra <<< "$spec"
  TAG="snorkel_finance_split_s2_${label}_v2_lr3e_3_nb${NB}_heldout"
  if [ -f "$MDIR/${TAG}.done" ]; then log "$TAG: done, skip"; continue; fi
  log "SFT $TAG  (train: original pool, eval: clean 5 questions)"
  wait_gpu_free
  ./scripts/train_sft.sh "$ENV" "$variant" \
      --dataset_path "$pool" --run_tag "$TAG" --best_metric eval_action_ppl \
      --learning_rate 3e-3 --num_batches "$NB" --early_stop_patience 3 $extra \
      > "$MDIR/${TAG}.log" 2>&1 \
    && { touch "$MDIR/${TAG}.done"; log "$TAG: done"; } || log "$TAG: FAILED (see $MDIR/${TAG}.log)"
done
touch "$MDIR/ALLDONE"; log "=== finance v2 Stage-2 complete ==="
