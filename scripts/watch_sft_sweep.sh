#!/usr/bin/env bash
# Babysit the Stage-2 SFT LR sweep: restart the driver if it dies, snapshot results
# periodically, and append a compact status line to /tmp/aprm/watch.log.
#
# The sweep itself is resumable (skips any arm whose step_best exists), so restarting
# the driver is always safe. Run detached:
#   setsid nohup ./scripts/watch_sft_sweep.sh > /tmp/aprm/watch_driver.log 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/usr/local/bin:$PATH"
W=/tmp/aprm/watch.log; mkdir -p /tmp/aprm
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" >> "$W"; }
i=0
while true; do
  i=$((i+1))
  if ! ps -eo args | grep -q '[r]un_sft_lr_matrix.sh'; then
    if [ -f /tmp/aprm/lrmatrix/DONE ]; then log "matrix DONE — watcher exiting"; break; fi
    log "driver not running — restarting (resumable, completed arms are skipped)"
    setsid nohup ./scripts/run_sft_lr_matrix.sh > /tmp/aprm/lrmatrix_driver.log 2>&1 < /dev/null &
    sleep 20
  fi
  done_n=$(ls -d checkpoints_lora/act_prm_*/hf_qwen3_4b_instruct/*_lr1e_[34]_heldout*/step_best 2>/dev/null | wc -l)
  cur=$(pgrep -af '[m]ain_pytorch.py' | grep -o 'run_tag [^ ]*' | head -1)
  util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader 2>/dev/null | head -1)
  log "arms_done=$done_n gpu=$util ${cur:-<idle>}"
  # every ~30 min: refresh notes/CSVs and back the metrics up to dotsync
  if [ $((i % 6)) -eq 0 ]; then
    for e in act_prm/tau2_retail act_prm/tau2_airline act_prm/snorkel_finance_split; do
      uv run --no-project python scripts/analyze_sft.py "$e" >/dev/null 2>&1 || true
    done
    ./scripts/backup_artifacts.sh >/dev/null 2>&1 || true
    log "refreshed SFT notes/CSVs + artifact backup"
  fi
  sleep 300
done
