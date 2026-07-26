#!/usr/bin/env bash
# Real-data Act-PRM training on the Snorkel Agent Finance traces.
#
# - Streams the traces via Meta's forward proxy on the FIRST run and persists the
#   processed pools to ./data/snorkel_finance (--dataset_path), so later runs load
#   them from disk (no re-streaming).
# - Loads Qwen3-4B-Instruct-2507 from the local HF cache (configs' cache_dir).
#
# Usage:
#   ./scripts/train_real.sh                       # defaults below
#   ./scripts/train_real.sh --num_batches 100     # override / add any main_pytorch flag
#   ./scripts/train_real.sh --project_name act-prm-lenpen   # enable W&B logging
set -euo pipefail
cd "$(dirname "$0")/.."

# Egress + auth for the first (streaming) run; harmless once the pool is cached.
export https_proxy="${https_proxy:-http://fwdproxy:8080}"
export http_proxy="${http_proxy:-http://fwdproxy:8080}"
export HF_TOKEN="${HF_TOKEN:-$(cat /home/mzio/models/token 2>/dev/null || true)}"

uv run python main_pytorch.py \
  --env_config act_prm/snorkel_finance \
  --model_config hf_qwen3_4b_instruct \
  --lora_config r8_a16_linear \
  --generator_config act_prm \
  --trainer_config pg \
  --replay_buffer_config default \
  --dataset_path ./data/snorkel_finance \
  --num_trajectories 32 --eval_trajectories 8 --max_traj_timestep 20 \
  --group_size 4 --batch_size 4 \
  --num_batches 40 --eval_every 10 --no_initial_eval --save_every 10 \
  --length_penalty 0.15 --learning_rate 4e-5 \
  --verbose "$@"
