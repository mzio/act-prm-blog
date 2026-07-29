#!/usr/bin/env bash
# Redo ONLY the corrupted retail "thoughts_*" arms after the clean RL arms finish.
#
# Why: scripts/export_sft_corpus.py used to join generations->state via
# pool[sample_id % len(pool)], but the relabel dataloader SHUFFLES the pool between
# epochs, so ~31% of steps got the WRONG trajectory's state. That corrupted the 4
# thoughts_* SFT corpora -> their SFT arms -> their RL inits. (actions_only,
# expert_thoughts, base RL floor are UNAFFECTED.) export_sft_corpus.py is now fixed
# (target_action-sequence match) and the 4 corpora under
# data/sft_corpus/tau2_retail/{policy,base,policy_last,base_last} were rebuilt clean.
#
# This controller, when launched LATER, waits for the clean RL arms to finish, stops
# the running stage3 driver so it can't RL a corrupted init, archives the 4 corrupted
# thoughts SFT runs, re-SFTs them (x {hide,full}) on the CORRECTED corpora, then
# relaunches the (resumable) stage3 driver to RL the corrected thoughts arms + do the
# final full-42 evals.
#
# Launch:  nohup ./scripts/run_retail_thoughts_redo.sh >/tmp/aprm/thoughts_redo.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/usr/local/bin:$PATH"

MODEL="hf_qwen3_4b_instruct"; export MODEL_CFG="$MODEL"
CKROOT="checkpoints_lora/act_prm_tau2_retail/$MODEL"   # Stage-2 SFT checkpoints
LOGROOT="logs/act_prm_tau2_retail/$MODEL"              # Stage-2 SFT logs
S3ROOT="checkpoints_lora/tau2bench_retail/$MODEL"      # Stage-3 RL checkpoints
ARCH_CK="$CKROOT/_archive_corpusbug"                   # where corrupted SFT ckpts go
ARCH_LOG="$LOGROOT/_archive_corpusbug"                 # where corrupted SFT logs go

LOG=/tmp/aprm/thoughts_redo.log
mkdir -p "$(dirname "$LOG")"
# Mirror everything (incl. the relaunched stage3 driver) to the redo log.
exec > >(tee -a "$LOG") 2>&1

log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
# Wait until no training process is on the GPU (our own SFT runs foreground, so this
# only fires while OTHER main_pytorch.py runs are alive).
wait_gpu_free(){ while pgrep -f '[m]ain_pytorch.py' >/dev/null 2>&1; do sleep 60; done; sleep 10; }
# newest matching path (dir or file), or empty
newest(){ ls -dt $1 2>/dev/null | head -1; }

# (SFT variant -> corrected corpus subdir under data/sft_corpus/tau2_retail)
VARIANTS=(thoughts_policy thoughts_base thoughts_policy_last thoughts_base_last)
declare -A CORPUS=(
  [thoughts_policy]=policy
  [thoughts_base]=base
  [thoughts_policy_last]=policy_last
  [thoughts_base_last]=base_last
)

log "=== retail thoughts redo: start ==="

# ---------------------------------------------------------------------------
# 1) WAIT until the CURRENT arm (retail_rl_actions_only) is DONE, then take over BEFORE
#    expert_thoughts starts — so we redo the 4 thoughts arms next, and expert_thoughts
#    runs LAST (in the relaunch, whose arm order now puts it last). base is already done.
# ---------------------------------------------------------------------------
S3LOG=/tmp/aprm/stage3/stage3.log
log "waiting for retail_rl_actions_only to finish (log line '... retail_rl_actions_only: done') ..."
until grep -q "retail_rl_actions_only: done" "$S3LOG" 2>/dev/null; do
  sleep 120
done
log "actions_only RL done -> stopping driver before expert_thoughts, then redo thoughts"

# ---------------------------------------------------------------------------
# 2) STOP the running stage3 driver so it cannot RL a corrupted thoughts init,
#    then kill any live training process and wait for the GPU to free.
# ---------------------------------------------------------------------------
log "stopping stage3 driver + any live training ..."
pkill -f '[r]un_retail_stage3' && log "  killed run_retail_stage3 driver" || log "  no run_retail_stage3 driver running"
sleep 2
pkill -f '[m]ain_pytorch.py' && log "  killed main_pytorch.py" || log "  no main_pytorch.py running"
wait_gpu_free
log "GPU free."

# ---------------------------------------------------------------------------
# 3) ARCHIVE the corrupted thoughts SFT runs (ckpts + logs) into _archive_corpusbug/.
#    The globs catch BOTH the hide (..._heldout-*) and full (..._heldout_fullctx-*)
#    runs; the _last variants are matched by their own globs (policy_heldout* does
#    NOT match policy_last_heldout*).
# ---------------------------------------------------------------------------
mkdir -p "$ARCH_CK" "$ARCH_LOG"
for v in "${VARIANTS[@]}"; do
  for d in "$CKROOT/retail_s2_${v}_heldout"*; do
    [ -e "$d" ] || continue
    case "$d" in "$ARCH_CK"*) continue;; esac      # already-archived / the archive dir itself
    log "  archive ckpt: $(basename "$d")"
    mv "$d" "$ARCH_CK/" 2>/dev/null || log "    (could not move $d; may already be archived)"
  done
  for d in "$LOGROOT/retail_s2_${v}_heldout"*; do
    [ -e "$d" ] || continue
    case "$d" in "$ARCH_LOG"*) continue;; esac
    log "  archive log : $(basename "$d")"
    mv "$d" "$ARCH_LOG/" 2>/dev/null || log "    (could not move $d; may already be archived)"
  done
done
log "archive done -> $ARCH_CK , $ARCH_LOG"

# ---------------------------------------------------------------------------
# 4) RE-SFT the 4 thoughts variants x {hide, full} on the CORRECTED corpora.
#    run_tags EXACTLY match the originals so the stage3 RL init-glob resolves them.
#    Idempotent-ish: skip a variant/regime whose fresh (post-archive, top-level)
#    step_best already exists.
# ---------------------------------------------------------------------------
run_sft(){
  local variant="$1" regime="$2" corpus="${CORPUS[$1]}"
  local run_tag="retail_s2_${variant}_heldout"; local full=0
  [ "$regime" = full ] && { run_tag="${run_tag}_fullctx"; full=1; }
  # Fresh run already present (post-archive)? skip.
  if [ -n "$(newest "$CKROOT/${run_tag}-*/step_best/adapter_model.safetensors")" ]; then
    log "SFT $run_tag: fresh step_best already exists, skip"
    return
  fi
  log "SFT $run_tag (corpus=$corpus, regime=$regime) ..."
  wait_gpu_free
  if [ "$full" = 1 ]; then
    SFT_FULLCTX=1 ./scripts/train_sft.sh act_prm/tau2_retail "$variant" \
      --run_tag "$run_tag" --dataset_path "data/sft_corpus/tau2_retail/$corpus" \
      --best_metric eval_action_ppl \
      && log "SFT $run_tag: done" || log "SFT $run_tag: FAILED"
  else
    ./scripts/train_sft.sh act_prm/tau2_retail "$variant" \
      --run_tag "$run_tag" --dataset_path "data/sft_corpus/tau2_retail/$corpus" \
      --best_metric eval_action_ppl \
      && log "SFT $run_tag: done" || log "SFT $run_tag: FAILED"
  fi
}

for v in "${VARIANTS[@]}"; do
  for regime in hide full; do
    run_sft "$v" "$regime"
  done
done
log "SFT re-runs done."

# ---------------------------------------------------------------------------
# 5) RELAUNCH the resumable stage3 driver. It SKIPS base/actions_only/expert_thoughts
#    RL (step_bests exist), resolves the CORRECTED thoughts SFT step_bests as inits,
#    RLs the 4 thoughts arms, and does the final full-42 evals for all arms.
# ---------------------------------------------------------------------------
log "relaunching stage3 driver (resumable) ..."
wait_gpu_free
CUDA_VISIBLE_DEVICES=0 ./scripts/run_retail_stage3.sh
log "=== retail thoughts redo: done ==="
