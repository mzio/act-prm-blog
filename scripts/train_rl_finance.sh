#!/usr/bin/env bash
# Stage 3 — RL (GRPO PG) on the interactive snorkel_finance gym, warm-started from a
# Stage-2 SFT LoRA. Runs in the base .venv (has claude-agent-sdk + litellm); the
# finqa_reasoning answer is graded by an LLM judge (Claude via the Agent SDK, headless
# over the Meta AI gateway). hide-obs regime (matches the hide SFT checkpoints).
#
# Usage:  ./scripts/train_rl_finance.sh <sft_ckpt_dir> <run_tag> <gpu> [extra flags]
#   sft_ckpt_dir : Stage-2 checkpoint dir with adapter_model.safetensors (step_best)
# NOTE: trains/evals on the gym's OWN frac split of finqa_reasoning, NOT the aprm
# rl_eval uids (uid->CSV mapping TODO; see cc-finance-2.0-rl-design.md).
set -euo pipefail
cd "$(dirname "$0")/.."
SFT="${1:?sft ckpt dir}"; TAG="${2:?run_tag}"; GPU="${3:?gpu id}"; shift 3 || true
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/usr/local/bin:$PATH"
export HF_HOME="${HF_HOME:-/data/users/$USER/models/hf_cache}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1 UV_FROZEN=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
[ -f "$SFT/adapter_model.safetensors" ] || { echo "no adapter_model.safetensors under $SFT"; exit 1; }
# CLAUDECODE cleared so the grader's Claude Agent SDK can spawn a (nested) claude subprocess.
# Run in .venv-tau2, same as train_rl_from_sft.sh: the grader's Claude Agent SDK lives
# there, not in the base .venv (the finance box's base venv had it; ours does not).
CUDA_VISIBLE_DEVICES=$GPU CLAUDECODE= UV_PROJECT_ENVIRONMENT=.venv-tau2 \
  exec uv run --no-sync python main_pytorch.py \
  --env_config act_prm/snorkel_finance_gym --model_config hf_qwen3_4b_instruct \
  --lora_config r8_a16_linear --generator_config hf_grpo --trainer_config pg \
  --replay_buffer_config default --resume_from "$SFT" \
  --group_size 4 --batch_size 2 --max_turns 8 --max_tokens 2048 \
  --num_batches 20 --eval_every 5 --no_initial_eval --hide_observations \
  --gradient_checkpointing --run_tag "$TAG" --verbose "$@"
