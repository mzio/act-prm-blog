#!/usr/bin/env bash
# Periodic wrapper around backup_results.sh (crontab is not permitted for this user).
# Start:  setsid nohup ./scripts/backup_results_daemon.sh > /tmp/backup_results.daemon.log 2>&1 &
# Stop:   pkill -f backup_results_daemon.sh
# Interval: BACKUP_INTERVAL_SEC (default 1800 = 30 min)
set -uo pipefail
cd "$(dirname "$0")/.."
INT="${BACKUP_INTERVAL_SEC:-1800}"
while true; do
  ./scripts/backup_results.sh auto 2>&1 | sed "s/^/[$(date '+%m-%d %H:%M')] /"
  sleep "$INT"
done
