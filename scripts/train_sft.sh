#!/usr/bin/env bash
# Stage 2 — SFT (with hide-observations) on tau2/finance expert data.
#
# Three variants (each SFTs a fresh LoRA; hide_observations=true compacts the
# context to: system + first user + last observation + ALL model messages):
#   actions_only    - expert action-only baseline (no thoughts)          -> generator act_prm_actions_only
#   thoughts_policy - thoughts+actions; best thought scored by the POLICY -> act_prm --advantage_mode best --no-score_with_base
#   thoughts_base   - thoughts+actions; best thought scored by the BASE   -> act_prm --advantage_mode best --score_with_base
#
# The SFTTrainer does plain (advantage-weighted) CE; advantage_mode=best +
# drop_zero_advantage (sft.yaml) means only the single best thought+action per
# step is trained (MLE). rl_eval tasks are held out by the split.
#
# NOTE on coverage: this trains on the split's act_prm_train tasks. To SFT over
# ALL non-rl_eval tasks (train+eval), regenerate the split with everything in
# train:  uv run python scripts/make_split.py --dataset <ds> --name <name> \
#           --train_frac 0.85 --eval_frac 0.0   (rl_eval = the remainder)
#
# Usage:
#   ./scripts/train_sft.sh <env> <variant> [extra main_pytorch flags]
#   env     : act_prm/tau2_retail | act_prm/tau2_airline | act_prm/snorkel_finance_split
#   variant : actions_only | thoughts_policy | thoughts_base
# e.g.:
#   CUDA_VISIBLE_DEVICES=0 ./scripts/train_sft.sh act_prm/tau2_retail thoughts_base
set -euo pipefail
cd "$(dirname "$0")/.."
ENVCFG="${1:?env, e.g. act_prm/tau2_retail}"
VARIANT="${2:?variant: actions_only | thoughts_policy | thoughts_base}"
shift 2 || true

export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/usr/local/bin:$PATH"
# HuggingFace access (stream datasets / load models): Meta forward proxy + HF token.
export https_proxy="${https_proxy:-http://fwdproxy:8080}"
export http_proxy="${http_proxy:-http://fwdproxy:8080}"
export HF_TOKEN="${HF_TOKEN:-$(cat "$HOME/models/token" 2>/dev/null || true)}"

case "$VARIANT" in
  actions_only)    GEN="act_prm_actions_only"; MODE=() ;;
  thoughts_policy) GEN="act_prm"; MODE=(--advantage_mode best --no-score_with_base) ;;
  thoughts_base)   GEN="act_prm"; MODE=(--advantage_mode best --score_with_base) ;;
  *) echo "bad variant '$VARIANT' (actions_only|thoughts_policy|thoughts_base)"; exit 1 ;;
esac

exec uv run python main_pytorch.py \
  --env_config "$ENVCFG" --model_config hf_qwen3_4b_instruct \
  --lora_config r8_a16_linear --generator_config "$GEN" --trainer_config sft \
  --replay_buffer_config default --hide_observations \
  --group_size 4 --batch_size 4 --num_batches 60 --eval_every 10 --no_initial_eval \
  --length_penalty 0.15 "${MODE[@]}" --verbose "$@"
