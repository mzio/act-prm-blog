#!/usr/bin/env bash
# General Act-PRM training runner: sets the forward proxy + HF token (needed the
# first time a split is streamed; harmless once cached), then runs main_pytorch.py
# with whatever flags you pass.
#
# Examples:
#   ./scripts/train.sh --env_config act_prm/tau2_retail  --model_config hf_qwen3_4b_instruct \
#       --generator_config act_prm --trainer_config pg --lora_config r8_a16_linear \
#       --replay_buffer_config default --group_size 4 --batch_size 4 --num_batches 25 \
#       --eval_every 5 --no_initial_eval --length_penalty 0.15 --obs_max_chars 3000 --verbose
set -euo pipefail
cd "$(dirname "$0")/.."
export https_proxy="${https_proxy:-http://fwdproxy:8080}"
export http_proxy="${http_proxy:-http://fwdproxy:8080}"
export HF_TOKEN="${HF_TOKEN:-$(cat /home/mzio/models/token 2>/dev/null || true)}"
exec uv run python main_pytorch.py "$@"
