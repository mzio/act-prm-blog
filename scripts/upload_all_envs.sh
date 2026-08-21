#!/usr/bin/env bash
# Publish finished Stage-2 SFT adapters for ALL THREE domains to their HF repos.
#
# watch_upload_lora.sh only covers retail; airline and finance adapters produced on this
# box were never being published, so the only copies were on local disk (checkpoints are
# excluded from backup_artifacts.sh as "re-runnable" -- ~2.5 GPU-hours each to rebuild).
# HF is reachable through fwdproxy (unlike github), so this is the durable store for them.
#
# upload_lora_hf.py is idempotent: it skips runs still training and re-uploads only when a
# checkpoint's mtime changes, so this is safe to run on a timer alongside the sweep.
# Cron: */15 * * * * /home/mzio/projects/act-prm-blog/scripts/upload_all_envs.sh
set -uo pipefail
cd "$(dirname "$0")/.." || exit 0
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/usr/local/bin:/usr/bin:/bin:$PATH"
export https_proxy="${https_proxy:-http://fwdproxy:8080}"
export http_proxy="${http_proxy:-http://fwdproxy:8080}"
export HF_TOKEN="${HF_TOKEN:-$(cat "$HOME/models/token" 2>/dev/null || true)}"
export HF_HUB_DISABLE_XET=1   # Xet CDN is blocked here
G=/tmp/aprm; mkdir -p "$G"
exec 7>"$G/upload_all.lock" || exit 0
flock -n 7 || exit 0

for spec in "act_prm/tau2_retail:mzio/aprm-sft-tau2-retail" \
            "act_prm/tau2_airline:mzio/aprm-sft-tau2-airline" \
            "act_prm/snorkel_finance_split:mzio/aprm-sft-snorkel-finance"; do
  env="${spec%%:*}"; repo="${spec##*:}"
  out=$(MODEL_CFG=hf_qwen3_4b_instruct uv run --no-project python scripts/upload_lora_hf.py \
          --env "$env" --repo "$repo" 2>&1 | tail -2)
  echo "[$(date '+%m-%d %H:%M:%S')] ${env##*/}: $out" >> "$G/upload_all.log"
done
