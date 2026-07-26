#!/usr/bin/env bash
# Stage 3 — RL (GRPO PG) on the live tau2-gym env, warm-started from a Stage-2 SFT
# LoRA checkpoint. Runs in .venv-tau2 (tau2 + litellm + claude-agent-sdk). The user
# simulator + judge are driven by the Claude Agent SDK (ambient Claude Code OAuth on
# a devserver, else CLAUDE_CODE_OAUTH_TOKEN in .env) — see the tau2bench env config.
#
# Usage:
#   ./scripts/train_rl_from_sft.sh <domain> <sft_ckpt_dir> [extra flags]
#   domain       : airline | retail
#   sft_ckpt_dir : a Stage-2 checkpoint dir containing adapter_model.safetensors
#                  (e.g. checkpoints_lora/act_prm_.../step_best)
# e.g.:
#   ./scripts/train_rl_from_sft.sh retail checkpoints_lora/.../step_best --num_batches 40
#
# NOTE (rl_eval hold-out): the tau2-gym env selects tasks by its own index-based
# split (num_train_tasks/num_test_tasks), NOT the aprm `rl_eval` uids. To evaluate
# strictly on the held-out `rl_eval` tasks, a uid->tau2-task-index mapping still
# needs wiring; until then this trains/evals on tau2's own split.
set -euo pipefail
cd "$(dirname "$0")/.."
DOMAIN="${1:?airline|retail}"
SFT_CKPT="${2:?path to an SFT LoRA dir (with adapter_model.safetensors)}"
shift 2 || true

export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/usr/local/bin:$PATH"
export https_proxy="${https_proxy:-http://fwdproxy:8080}"
export http_proxy="${http_proxy:-http://fwdproxy:8080}"
export HF_TOKEN="${HF_TOKEN:-$(cat "$HOME/models/token" 2>/dev/null || true)}"

[ -f "$SFT_CKPT/adapter_model.safetensors" ] || { echo "no adapter_model.safetensors under $SFT_CKPT"; exit 1; }

UV_PROJECT_ENVIRONMENT=.venv-tau2 exec uv run --no-sync python main_pytorch.py \
  --env_config "tau2bench/$DOMAIN" --model_config hf_qwen3_4b_instruct \
  --lora_config r8_a16_linear --generator_config hf_grpo --trainer_config pg \
  --replay_buffer_config default --resume_from "$SFT_CKPT" --gradient_checkpointing \
  --group_size 4 --batch_size 2 --max_turns 8 --max_tokens 2048 \
  --num_batches 25 --eval_every 5 --no_initial_eval --verbose "$@"
