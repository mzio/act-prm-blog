#!/usr/bin/env bash
# Keep the Stage-2 AdamW sweep alive until every arm has a .done marker.
#
# Why: on 2026-08-29 the sweep died at 19:19 for reasons still unknown (log ends
# mid-output, no traceback, no readable dmesg/cgroup counters) and nothing noticed
# until 02:07 -- seven idle hours. The crash cause is unfixed, so the mitigation is
# fast detection: poll every 2 min, relaunch if the driver is gone and work remains.
# run_stage2_adamw.sh skips completed domains via its own .done markers, and
# run_sft_sweep.sh skips completed arms, so a relaunch resumes rather than restarts.
set -uo pipefail
cd /home/mzio/projects/act-prm-blog
L=/tmp/aprm/stage2_adamw/watchdog.log
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" >> "$L"; }
ARMS_PER_DOMAIN=3   # actions_only, expert_thoughts, thoughts_policy_adamw30
log "watchdog up"
while true; do
  done_n=$(ls /tmp/aprm/sft_sweep_*/*lr1e_3_adamw_nb100_heldout.done 2>/dev/null | wc -l)
  if [ "$done_n" -ge $((4 * ARMS_PER_DOMAIN)) ]; then
    log "all $done_n arms done -- watchdog exiting"; exit 0
  fi
  if ! pgrep -f 'run_stage2_adamw\.sh' >/dev/null 2>&1; then
    # don't relaunch on top of a live trainer (a slow arm can outlive a driver blip)
    if pgrep -f 'main_pytorch\.py' >/dev/null 2>&1; then
      log "driver gone but a trainer is still alive -- waiting"
    else
      log "driver GONE with $done_n/12 arms done -- relaunching"
      env -u HF_HOME -u WANDB_MODE setsid nohup ./scripts/run_stage2_adamw.sh >/dev/null 2>&1 &
      sleep 60
    fi
  fi
  sleep 120
done
