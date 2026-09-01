#!/usr/bin/env bash
# Rollout phase for the lr 1e-4 multidomain sweep. See notes/cc-11.0.
#
# Per domain, a LADDER of snapshots rather than only step_best, because step_best (PPL)
# was the WORST rollout checkpoint on retail (0/42 vs 5/42 at step_0020). Primary read is
# user-turns/episode, not the completion count: the same checkpoint scored 11.9% at n=42
# and 15.1% at n=126, so 1-3 task differences are noise.
#
# airline  -> tau2 gym via run_sft_rollout_eval.sh (proven path)
# finance  -> run_finance_rollout.sh (proven; MAX_TURNS=24 -- at the env default of 12
#             every arm scored 0/10 because the policy never reached respond_user)
# insurance-> NOT ATTEMPTED: the gym has never been used for a rollout eval and the
#             rl_eval uid -> task-index mapping is an open TODO (cc-10.0). Training
#             results for insurance are still produced; only its rollout is skipped.
set -uo pipefail
cd /home/mzio/projects/act-prm-blog
[ -f scripts/wandb_preflight.sh ] && . scripts/wandb_preflight.sh
export PYTHONUNBUFFERED=1 HF_HOME=/data/users/mzio/models/hf_cache
MDIR=/tmp/aprm/lr1e4_multi; mkdir -p "$MDIR"
L="$MDIR/rollouts.log"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$L"; }

kill_orphans(){   # killing a driver alone leaves its main_pytorch child holding the GPU
  for p in $(ps -eo pid,args | awk '$2=="bash" && ($3 ~ /run_sft_rollout_eval\.sh$/ || $3 ~ /run_finance_rollout\.sh$/) {print $1}'); do
    kill -9 "$p" 2>/dev/null; done
  for p in $(pgrep -f 'main_pytorch\.py'); do kill -9 "$p" 2>/dev/null; done
  sleep 8
}

log "=== rollout phase ==="
# ---- airline: tau2 gym -------------------------------------------------------------
for spec in "thoughts_policy_adamw30:step_0020" "thoughts_policy_adamw30:step_best" \
            "expert_thoughts_all:step_best"    "actions_only:step_best"; do
  arm="${spec%%:*}"; step="${spec##*:}"
  [ -f "$MDIR/airline.$arm.$step.done" ] && { log "airline/$arm@$step done, skip"; continue; }
  ck=$(ls -d checkpoints_lora/act_prm_tau2_airline/hf_qwen3_4b_instruct/airline_s2_${arm}_lr1e_4_adamw_nb200_flat32_heldout-*/${step} 2>/dev/null | head -1)
  [ -z "$ck" ] && { log "airline/$arm@$step: checkpoint MISSING, skip"; continue; }
  log "ROLLOUT airline/$arm@$step"
  VARIANTS="$arm" CKPT_PAT="lr1e_4_adamw_nb200_flat32" CKPT_TAG="lr1e4" CKPT_STEP="$step" \
    ./scripts/run_sft_rollout_eval.sh >> "$L" 2>&1 || log "  airline/$arm@$step FAILED"
  touch "$MDIR/airline.$arm.$step.done"
  kill_orphans
done

# ---- finance: its own gym ----------------------------------------------------------
for spec in "thoughts_policy_adamw30:step_0020" "thoughts_policy_adamw30:step_best" \
            "expert_thoughts_all:step_best"    "actions_only:step_best"; do
  arm="${spec%%:*}"; step="${spec##*:}"
  [ -f "$MDIR/finance.$arm.$step.done" ] && { log "finance/$arm@$step done, skip"; continue; }
  ck=$(ls -d checkpoints_lora/act_prm_snorkel_finance_split/hf_qwen3_4b_instruct/snorkel_finance_split_s2_${arm}_lr1e_4_adamw_nb200_flat32_heldout-*/${step} 2>/dev/null | head -1)
  [ -z "$ck" ] && { log "finance/$arm@$step: checkpoint MISSING, skip"; continue; }
  log "ROLLOUT finance/$arm@$step"
  # NB: finance's driver uses ARMS, not VARIANTS
  MAX_TURNS=24 ARMS="$arm" CKPT_PAT="lr1e_4_adamw_nb200_flat32" CKPT_TAG="lr1e4" CKPT_STEP="$step" \
    ./scripts/run_finance_rollout.sh >> "$L" 2>&1 || log "  finance/$arm@$step FAILED"
  touch "$MDIR/finance.$arm.$step.done"
  kill_orphans
done

log "insurance rollout SKIPPED by design (gym never exercised; uid->task map is a TODO)"
log "=== rollout phase complete ==="
