#!/usr/bin/env bash
# Periodic artifact backup — runs backup_results.sh every INTERVAL seconds so
# overnight pipeline progress (EM/relabel/SFT metrics + step_best adapters) is
# continuously mirrored to dotsynced ~/.claude, surviving a devserver recycle even
# when no one is actively monitoring. Launch:
#   nohup ./scripts/periodic_backup.sh > /tmp/aprm/periodic_backup.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
INTERVAL="${1:-1800}"   # default 30 min
while true; do
  sleep "$INTERVAL"
  ./scripts/backup_results.sh >/dev/null 2>&1 || true
  echo "[$(date '+%m-%d %H:%M:%S')] periodic_backup ran"
done
