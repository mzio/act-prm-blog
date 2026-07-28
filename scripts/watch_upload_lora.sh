#!/usr/bin/env bash
# Watcher: auto-upload finished SFT LoRA checkpoints to HF as they complete.
# Every INTERVAL secs it runs the (skip-if-already-on-hub) uploader for retail 4B + 8B,
# so each newly-finished step_best is published shortly after it lands. CPU/network only
# — never touches the GPU or the training runs. Pidfile-guarded (single instance).
#
# Usage:  nohup ./scripts/watch_upload_lora.sh [interval_secs] >> /tmp/aprm/upload_watch.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/usr/local/bin:$PATH"
export HF_HOME=/data/users/mzio/models/hf_cache
export https_proxy=http://fwdproxy:8080 http_proxy=http://fwdproxy:8080 HF_HUB_DISABLE_XET=1
export HF_TOKEN="${HF_TOKEN:-$(cat "$HOME/models/token" 2>/dev/null)}"
INTERVAL="${1:-900}"   # default 15 min
mkdir -p /tmp/aprm

echo "[$(date '+%m-%d %H:%M:%S')] upload watcher start (interval=${INTERVAL}s)"
while true; do
  for MC in hf_qwen3_4b_instruct hf_qwen3_8b; do
    MODEL_CFG="$MC" CUDA_VISIBLE_DEVICES="" uv run --no-sync python scripts/upload_lora_hf.py \
      --env act_prm/tau2_retail --repo mzio/aprm-sft-tau2-retail 2>&1 \
      | grep -E 'uploaded|DONE' | sed "s/^/[$(date '+%H:%M:%S')][$MC] /"
  done
  sleep "$INTERVAL"
done
