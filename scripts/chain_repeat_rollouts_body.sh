#!/usr/bin/env bash
# Triplicate rollouts on the load-bearing checkpoints, to measure the rollout noise
# distribution instead of assuming it.
#
# WHY: on 2026-09-03 two NUMERICALLY IDENTICAL step_0020 checkpoints (adapter diff
# mean|A-B|/mean|A| = 0.0000, max 4.99e-05 over 504 tensors) scored 31.0% and 16.7% on the
# same 42 retail tasks -- a 14.3pt swing from sampling temperature + the Claude user
# simulator + the LLM judge. Wilson CIs model only binomial sampling over TASKS and assume
# a deterministic policy, so they understate this badly. Every comparison under ~15pts in
# cc-12.0/cc-13.0 is therefore unresolved.
#
# Targets, chosen because they are what the surviving claim rests on:
#   insurance thoughts_policy / actions_only x {step_0020, step_best}  (the p<0.001 result)
#   retail    thoughts_policy step_0020                               (already has 2 discordant runs)
# Distinct CKPT_TAG per repeat so .done markers and log dirs never collide.
set -uo pipefail
cd /home/mzio/projects/act-prm-blog
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/usr/local/bin:$PATH"
[ -f scripts/wandb_preflight.sh ] && . scripts/wandb_preflight.sh
export PYTHONUNBUFFERED=1 HF_HOME=/data/users/mzio/models/hf_cache
export ACT_PRM_DUMP_TRAJECTORIES=1
G=/tmp/aprm/repeats; mkdir -p "$G"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$G/chain.log"; }
reap(){ for p in $(pgrep -f 'main_pytorch\.py'); do kill -9 "$p" 2>/dev/null; done; sleep 20; }

CKPAT="lr1e_3_nb200_flat32sgd"
# --- insurance: 2 extra runs each for 4 configs (existing run = rep1)
for rep in rep2 rep3; do
  for arm in thoughts_policy_adamw30 actions_only; do
    for step in step_0020 step_best; do
      m="$G/ins.$arm.$step.$rep.done"; [ -f "$m" ] && { log "ins $arm $step $rep done, skip"; continue; }
      log "REPEAT insurance $arm/$step ($rep)"
      ARMS="$arm" CKPT_PAT="$CKPAT" CKPT_STEP="$step" CKPT_TAG="sgdlr1e_3_$rep" \
        ./scripts/run_insurance_rollout.sh >>"$G/chain.log" 2>&1
      log "  rc=$?"; touch "$m"; reap
    done
  done
done
# --- retail thoughts_policy step_0020: one more run (nb200 gave 31.0%, nb1000 gave 16.7%)
m="$G/retail.tp.step_0020.rep3.done"
if [ ! -f "$m" ]; then
  log "REPEAT retail thoughts_policy/step_0020 (rep3)"
  VARIANTS="thoughts_policy_adamw30" CKPT_PAT="$CKPAT" CKPT_STEP="step_0020" \
  CKPT_TAG="sgdlr1e_3_rep3" ./scripts/run_sft_rollout_eval.sh >>"$G/chain.log" 2>&1
  log "  rc=$?"; touch "$m"; reap
fi
log "=== repeat rollouts complete ==="
