#!/usr/bin/env bash
# Pre-fetch the Qwen3 instruct models + the Act-PRM traces dataset into the local
# HuggingFace cache, so training can run offline afterwards.
#
# The HF CLI lives inside the uv venv, so everything goes through `uv run`.
# Downloads land in $HF_HOME (default: /data/users/mzio/models/hf_cache, the same
# cache the instruct model configs point their `cache_dir` at — so new pulls sit
# next to Qwen3-4B-Instruct-2507 / Qwen3-8B, which are already there).
#
# Usage:
#   ./scripts/prefetch_models.sh                 # default set below
#   ./scripts/prefetch_models.sh Qwen/Qwen3-8B   # override with your own repo(s)
#   HF_HOME=/some/other/cache ./scripts/prefetch_models.sh
set -euo pipefail

export HF_HOME="${HF_HOME:-/data/users/mzio/models/hf_cache}"

# Egress: huggingface.co is not directly reachable here, but Meta's forward proxy
# is. Route through it unless the caller already set a proxy. Override/omit as needed.
: "${FWDPROXY:=http://fwdproxy:8080}"
export https_proxy="${https_proxy:-$FWDPROXY}"
export http_proxy="${http_proxy:-$FWDPROXY}"
# Auth for private repos (e.g. the Act-PRM dataset): reuse the cached HF token.
export HF_TOKEN="${HF_TOKEN:-$(cat /home/mzio/models/token 2>/dev/null || true)}"

MODELS=("$@")
if [ ${#MODELS[@]} -eq 0 ]; then
  MODELS=(
    "Qwen/Qwen3-4B-Instruct-2507"   # recommended default (paper model)
    "Qwen/Qwen3-0.6B"               # tiny / CPU-friendly
    # "Qwen/Qwen3-8B"               # uncomment if you want the 8B chat model
  )
fi

echo "HF_HOME=$HF_HOME"
for repo in "${MODELS[@]}"; do
  echo "==> downloading model: $repo"
  uv run hf download "$repo"
done

echo "==> downloading dataset: mzio/aprm-snorkelai_agent_finance_reasoning"
uv run hf download --repo-type dataset mzio/aprm-snorkelai_agent_finance_reasoning

echo "done."
