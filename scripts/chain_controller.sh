#!/usr/bin/env bash
# Sequence the two queued chains on one GPU so the v2 Stage-1 RL is not starved.
#
# THE PROBLEM: chain_insurance_fewshot_v2.sh originally waited on the WHOLE
# chain_expert_unfiltered.sh run (~17h: insurance SFT + 5 rollouts, retail/airline SFT + 3
# tau2 repeats, finance SFT + rollout). Stage-1 EM + relabel is the long pole of the v2
# work (10.7h) and the thing actually worth starting early, so it should go right after the
# insurance SFT that is already running -- not a day later.
#
# ORDER:
#   1. let the running insurance expert_thoughts SFT finish   (marker: sft.snorkel_insurance.done)
#   2. stop the expert chain                                   (it is fully .done-guarded, so
#      relaunching resumes exactly where it left off)
#   3. v2 Stage-1: EM -> relabel -> export -> contamination gate   ~10.7h
#   4. resume the expert chain: insurance rollouts, retail/airline, finance  ~15h
#   5. v2 Stage-2 SFT + 5 seed rollouts                             ~6.9h
#
# Only one main_pytorch runs at a time. Act-PRM EM peaks ~72 GiB uncapped (CLAUDE.md), so
# co-scheduling would need --group_size 2 or an --obs_max_chars cap -- either of which
# changes the Stage-1 recipe and breaks the match with insurance_s1em_policy_adamw30. Not
# worth it; keep them sequential.
set -uo pipefail
cd /home/mzio/projects/act-prm-blog
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/usr/local/bin:$PATH"

GE=/tmp/aprm/expert_unfiltered
GC=/tmp/aprm/controller; mkdir -p "$GC"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] CTL $*" | tee -a "$GC/controller.log"; }

EXPERT_PID="${EXPERT_PID:-}"
gpu_free(){ while pgrep -f '[m]ain_pytorch.py' >/dev/null 2>&1; do sleep 60; done; sleep 20; }

# ---- 1. wait for the insurance SFT marker
log "waiting for $GE/sft.snorkel_insurance.done"
while [ ! -f "$GE/sft.snorkel_insurance.done" ]; do
  # if the expert chain died before writing the marker, do not wait forever
  if [ -n "$EXPERT_PID" ] && ! kill -0 "$EXPERT_PID" 2>/dev/null; then
    log "expert chain (PID $EXPERT_PID) exited before the marker appeared; continuing anyway"
    break
  fi
  sleep 60
done
log "insurance SFT done"

# ---- 2. stop the expert chain so it does not take the GPU for its rollouts
if [ -n "$EXPERT_PID" ] && kill -0 "$EXPERT_PID" 2>/dev/null; then
  log "pausing expert chain (PID $EXPERT_PID)"
  kill "$EXPERT_PID" 2>/dev/null
  sleep 5
  for p in $(pgrep -f 'main_pytorch\.py'); do kill -9 "$p" 2>/dev/null; done
fi
gpu_free

# ---- 3. v2 Stage-1 (the priority)
log "=== v2 Stage-1: EM -> relabel -> export -> gate ==="
STOP_AFTER_GATE=1 ./scripts/chain_insurance_fewshot_v2.sh >>"$GC/fs2_stage1.log" 2>&1
rc=$?
log "v2 Stage-1 rc=$rc"
if [ "$rc" -ne 0 ]; then
  log "v2 Stage-1 FAILED (gate or a run) -- see /tmp/aprm/ins_fs2/chain.log. Resuming the"
  log "expert chain anyway so the GPU is not idle; v2 Stage-2 is skipped."
fi
gpu_free

# ---- 4. resume the expert chain (skips the completed insurance SFT via its markers)
log "=== resuming expert chain ==="
./scripts/chain_expert_unfiltered.sh >>"$GC/expert_resume.log" 2>&1
log "expert chain rc=$?"
gpu_free

# ---- 5. v2 Stage-2 + rollouts (only if Stage-1 produced a clean corpus)
if [ "$rc" -eq 0 ]; then
  log "=== v2 Stage-2 SFT + rollouts ==="
  ./scripts/chain_insurance_fewshot_v2.sh >>"$GC/fs2_stage2.log" 2>&1
  log "v2 Stage-2 rc=$?"
fi
log "=== controller complete ==="
