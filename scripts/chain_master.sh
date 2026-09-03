#!/usr/bin/env bash
# Single serial runner for everything remaining, replacing the PID-chained scripts.
#
# WHY A MASTER CHAIN: the previous design had 6 chains each polling the PREVIOUS chain's
# pid. When chain_corpus_ablation aborted early (0 snapshots) the whole tail cascaded --
# expert_all ran and no-op'd, insurance_base started against a broken proxy, all within
# 10 minutes. One process running phases in order cannot cascade like that.
#
# Ordered cheapest-and-most-load-bearing first:
#   1 insurance_base  (~1h)  -- without it the 52.5% insurance numbers are uninterpretable
#   2 repeats         (~9h)  -- measures the rollout noise floor; the surviving claim rests here
#   3 expert_all 1e-3 (~17h) -- the arm MZ asked for
#   4 corpus_ablation (~4h)  -- old vs adamw30 Stage-1 corpus
#   5 expert_all 3e-3 (~17h)
set -uo pipefail
cd /home/mzio/projects/act-prm-blog
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/usr/local/bin:$PATH"
[ -f scripts/wandb_preflight.sh ] && . scripts/wandb_preflight.sh
export PYTHONUNBUFFERED=1
export HF_HOME=/data/users/mzio/models/hf_cache
export HF_HUB_OFFLINE=1          # fwdproxy 403s huggingface.co as of 2026-09-03
export WANDB_ERROR_REPORTING=false   # sentry.io is proxy-blocked; its retries stall the run
export ACT_PRM_DUMP_TRAJECTORIES=1
G=/tmp/aprm/master; mkdir -p "$G"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$G/chain.log"; }
reap(){ for p in $(pgrep -f 'main_pytorch\.py'); do kill -9 "$p" 2>/dev/null; done; sleep 15; }
phase(){ # $1=marker name, $2..=command
  local m="$G/$1.done"; shift
  [ -f "$m" ] && { log "SKIP $m (already done)"; return; }
  log "PHASE START: $*"
  "$@" >>"$G/chain.log" 2>&1
  log "PHASE rc=$? : $*"
  touch "$m"; reap
}

log "=== master chain start ==="
phase p1_insurance_base ./scripts/run_insurance_base_rollout.sh
phase p2_repeats        ./scripts/chain_repeat_rollouts_body.sh
phase p3_expert_1e3     ./scripts/chain_expert_all_body.sh 1e-3
phase p4_corpus_abl     ./scripts/chain_corpus_ablation_body.sh
phase p5_expert_3e3     ./scripts/chain_expert_all_body.sh 3e-3
log "=== master chain complete ==="
