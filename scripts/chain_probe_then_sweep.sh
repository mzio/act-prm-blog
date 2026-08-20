#!/usr/bin/env bash
# Let the in-flight SFT arm finish, run an LR probe on a THOUGHT arm, then hand the GPU
# back to the sweep. Sequenced because there is one GPU and the watchdog would otherwise
# race the probe for it.
#
# Why probe a thought arm separately: actions_only has the shortest target span and the
# least headroom (it already sits at the worst PPL). The thought arms train on a target
# that is ~50% thought tokens, so the loss landscape -- and the LR that suits it -- can
# differ. That choice applies to all remaining arms, so it is worth settling first.
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/usr/local/bin:$PATH"
L=/tmp/aprm/chain.log; mkdir -p /tmp/aprm
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$L"; }

log "waiting for the in-flight arm to finish..."
while pgrep -f '[m]ain_pytorch.py' >/dev/null 2>&1; do sleep 60; done
log "GPU free — starting the thoughts_base LR probe"

# 30 batches, not 8: the eval curve only became legible around b20-b30 on actions_only,
# so an 8-batch probe would be uninformative again (it was, the first time). 3e-3 is in
# the grid because 1e-3 was still descending at b59 -- we may be step-limited, not
# LR-limited, and a higher LR is the cheap way to tell.
VARIANT=thoughts_base BATCHES=30 EVAL_EVERY=10 LRS="1e-4 1e-3 3e-3" \
  ./scripts/probe_sft_lr.sh >> "$L" 2>&1
log "probe done — results:"
uv run --no-project python scripts/report_lr_probe.py 2>&1 | tee -a "$L"

log "restarting the sweep watchdog"
setsid nohup ./scripts/watch_sft_sweep.sh > /tmp/aprm/watch_driver.log 2>&1 < /dev/null &
log "=== chain complete ==="
