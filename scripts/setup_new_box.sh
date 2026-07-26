#!/usr/bin/env bash
# One-shot setup for a fresh devserver (run AFTER cloning the repo + cd'ing in).
# For the Act-PRM EM runs you only need the base venv — tau2-bench/.venv-tau2 are
# only for the tau2-*gym* RL env, not the EM-over-expert-datasets runs.
#
# Bootstrap on the new box (pick one to GET the repo first):
#   git clone git@github.com:mzio/act-prm-blog.git ~/projects/act-prm-blog          # after you push
#   # or, from the dotsynced bundle (once dotsync populates ~/.claude):
#   git clone ~/.claude/act-prm-backups/act-prm-blog.bundle ~/projects/act-prm-blog
#   cd ~/projects/act-prm-blog && git checkout act-prm-pytorch && bash scripts/setup_new_box.sh
set -euo pipefail
cd "$(dirname "$0")/.."

export https_proxy="${https_proxy:-http://fwdproxy:8080}"
export http_proxy="${http_proxy:-http://fwdproxy:8080}"
export HF_TOKEN="${HF_TOKEN:-$(cat "$HOME/models/token" 2>/dev/null || true)}"
# Keep the model cache where the model configs' cache_dir points (per-box, not dotsynced).
export HF_HOME="${HF_HOME:-/data/users/$USER/models/hf_cache}"

# uv isn't on a non-interactive shell's PATH by default — add the common install
# locations, and install it if still missing (uv installer honors the proxy above).
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/usr/local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
  echo "==> uv not found; installing (via proxy)"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi
command -v uv >/dev/null 2>&1 || { echo "ERROR: uv still not found — install it and re-run"; exit 1; }

echo "==> 1/3 base venv (uv sync)"
uv sync

echo "==> 2/3 policy model into $HF_HOME (per-box cache; re-download on a fresh box)"
# HuggingFace access from a devserver: huggingface.co is NOT directly reachable, but
# Meta's forward proxy is — https_proxy=http://fwdproxy:8080 (set above). Auth for
# gated/private repos uses HF_TOKEN (a hf_... token); put it at ~/models/token (this
# script reads it) or `export HF_TOKEN=...`. github.com is BLOCKED by fwdproxy, so
# git deps (e.g. tau2-bench) must be cloned from a shell that reaches github.
[ -n "${HF_TOKEN:-}" ] || echo "  WARN: no HF_TOKEN (set ~/models/token or export HF_TOKEN) — gated/private repos will 401"
uv run hf download Qwen/Qwen3-4B-Instruct-2507 \
  || echo "  (couldn't download — check https_proxy=$http_proxy + HF_TOKEN, or set cache_dir/HF_HUB_OFFLINE=1)"

echo "==> 3/3 data: act-prm task splits are committed in data/splits/; the per-task"
echo "    trajectory pools (data/*) rebuild from the HF datasets via the proxy on first run."

cat <<'EOF'

Ready. Two uncapped EM runs, one per GPU (2-GPU box):
  CUDA_VISIBLE_DEVICES=0 ./scripts/train.sh --env_config act_prm/tau2_retail  \
      --generator_config act_prm --trainer_config pg --model_config hf_qwen3_4b_instruct \
      --lora_config r8_a16_linear --replay_buffer_config default --no-score_with_base \
      --group_size 4 --batch_size 4 --num_batches 25 --eval_every 5 --no_initial_eval \
      --length_penalty 0.15 --verbose > /tmp/retail.log 2>&1 &
  CUDA_VISIBLE_DEVICES=1 ./scripts/train.sh --env_config act_prm/tau2_airline ... > /tmp/airline.log 2>&1 &
EOF
