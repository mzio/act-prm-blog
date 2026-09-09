#!/usr/bin/env bash
# Seed-varied repeats at step_0020 for retail + airline (one driver call covers both).
#
# tau2's user simulator is an external Claude call we do not seed, so repeats are already
# independent -- but varying --seed also varies the POLICY's sampling, giving a second
# independent source and making the methodology uniform with the snorkel domains. Requires
# the SEED passthrough added to run_sft_rollout_eval.sh on 2026-09-09; main_pytorch encodes
# the seed in the run dir (s=<N>) so distinct seeds cannot collide.
#
# EXPECTATIONS: same-checkpoint spread is 16.6pt on retail (sd 9.0) and 22.2pt on airline.
# Three samples per arm tightens the error bars and confirms the noise model; it does NOT
# make 5pt arm differences resolvable there (that needs ~51 / ~77 runs). Insurance stays
# the only domain that can rank arms.
set -uo pipefail
cd /home/mzio/projects/act-prm-blog
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/usr/local/bin:$PATH"
[ -f scripts/wandb_preflight.sh ] && . scripts/wandb_preflight.sh
export PYTHONUNBUFFERED=1 HF_HOME=/data/users/mzio/models/hf_cache HF_HUB_OFFLINE=1
export WANDB_ERROR_REPORTING=false ACT_PRM_DUMP_TRAJECTORIES=1
G=/tmp/aprm/tau2_seeds; mkdir -p "$G"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$G/chain.log"; }
reap(){ for p in $(pgrep -f 'main_pytorch\.py'); do kill -9 "$p" 2>/dev/null; done; sleep 15; }

WAIT_PID="${WAIT_PID:-}"
if [ -n "$WAIT_PID" ]; then
  log "waiting on pid=$WAIT_PID"
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
  log "pid $WAIT_PID exited"; reap
fi

for SEED in 1234 777; do
  for arm in actions_only thoughts_policy_adamw30 expert_thoughts_all; do
    m="$G/$arm.$SEED.done"; [ -f "$m" ] && continue
    log "TAU2 (retail+airline) $arm seed=$SEED"
    SEED="$SEED" VARIANTS="$arm" CKPT_PAT="lr1e_3_nb200_flat32sgd" CKPT_STEP="step_0020" \
    CKPT_TAG="sgdlr1e_3_s${SEED}" ./scripts/run_sft_rollout_eval.sh >>"$G/chain.log" 2>&1
    log "  rc=$?"; touch "$m"; reap
  done
done
log "=== tau2 seed sweep complete ==="
