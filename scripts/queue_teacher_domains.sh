#!/usr/bin/env bash
# Queue Claude-teacher collection for retail + airline behind the running insurance shards.
#
# Sequenced, not parallel: 4 concurrent Claude streams is measured-good, but the API rate
# limit is unknown and a 429 storm mid-run is far messier to recover from than waiting.
# Wall-clock is the same either way -- these are API-bound, not GPU-bound, and each shard
# costs ZERO VRAM (the dead policy model sits on CPU), so nothing is idle while they wait.
#
# Remaining after insurance (261 tasks / 783 episodes, already running):
#   retail   114 tasks -> 342 episodes on 4 shards  ~1 h
#   airline   50 tasks -> 150 episodes on 2 shards  ~35 min
#
# Every stage is resumable: re-running skips nothing, but the per-domain corpora land in
# separate dirs so a partial run is never silently merged with a complete one.
set -uo pipefail
cd /home/mzio/projects/act-prm-blog
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/usr/local/bin:$PATH"

G=runlogs/claude_teacher; mkdir -p "$G"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] QUEUE $*" | tee -a "$G/queue.log"; }

# Match the collection processes only -- `main_pytorch.py` plus a *_teacher_* run tag --
# so this never waits on the unrelated EM / SFT chains sharing the box.
running(){ ps -eo args | grep -c "[m]ain_pytorch.py.*_teacher_"; }

wait_for_domain(){   # wait_for_domain <name>
  log "waiting for $1 shards to finish"
  while [ "$(running)" -gt 0 ]; do sleep 60; done
  sleep 20
  log "$1 shards done"
}

run_domain(){        # run_domain <domain> <n_shards>
  local dom="$1" n="$2" s
  log "launching $dom on $n shard(s)"
  for ((s=0; s<n; s++)); do
    setsid env DOMAIN="$dom" ATTEMPTS="${ATTEMPTS:-3}" SHARDS="$n" SHARD="$s" \
      ./scripts/collect_claude_trajectories.sh > "$G/drv_${dom}_sh${s}.log" 2>&1 < /dev/null &
    sleep 3
  done
  sleep 60          # let them all register before the wait loop samples
  wait_for_domain "$dom"
}

wait_for_domain "insurance (already running)"
run_domain retail 4
run_domain airline 2

# Merge each domain's shards into one corpus. Done per domain, not globally: the shards of
# a domain are disjoint task slices of the same collection, whereas different domains have
# different tools and system prompts and must stay separate.
for dom in insurance retail airline; do
  out="data/sft_corpus/${dom}_claude/all"
  log "merging $dom shards -> $out"
  uv run --no-sync python scripts/export_teacher_corpus.py \
      --runs "logs/*/*/${dom}_teacher_sh*/" --out "$out" >>"$G/queue.log" 2>&1 \
      && log "  ok" || log "  FAILED (see $G/queue.log)"
done
log "=== all teacher collection complete ==="
