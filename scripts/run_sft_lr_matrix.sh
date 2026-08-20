#!/usr/bin/env bash
# Stage-2 SFT re-run at higher learning rates, across every reported config.
#
# WHY: the entire PyTorch path ran at lr=4e-5 (configs/trainer/{sft,pg}.yaml, never
# tuned since the initial commit 88d1068) and every stage produced a near-null LoRA:
# max|(alpha/r)*B@A| ~ 1e-6..4e-5 against base weights of order 1e-2, with lora_A
# still pinned at its seeded init (two runs on DIFFERENT corpora agree to 1.5e-6).
# Held-out PPL/accuracy consequently moved 0.03-0.26% over 60 batches on all 36 runs.
# So the shipped Stage-2 checkpoints are ~the base model, and the Stage-3 RL arms all
# warm-started from the same place -- which is why they were indistinguishable.
#
# Matrix: {3 datasets} x {6 variants} x {hide,full} x {LRS} runs, serial on one GPU,
# resumable (a run whose step_best exists is skipped). At ~70 min/run the default
# 2-LR sweep is 72 runs ~= 84 GPU-hours. Run ./scripts/probe_sft_lr.sh first.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 nohup ./scripts/run_sft_lr_matrix.sh > /tmp/aprm/lrmatrix.log 2>&1 &
#   LRS="1e-4" ENVS="act_prm/tau2_retail" ./scripts/run_sft_lr_matrix.sh   # narrow it
#   DRY=1 ./scripts/run_sft_lr_matrix.sh                                   # plan only
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/usr/local/bin:$PATH"

# 3e-3, chosen by probe on a THOUGHT arm (retail thoughts_base, 30 batches):
#   lr 1e-4  b29  3.1756 -> 3.1772   -0.05%   dead (matches actions_only)
#   lr 1e-3  b29  3.1673 -> 3.1297   +1.19%
#   lr 3e-3  b29  3.1281 -> 2.9112   +6.94%   <- ~6x faster, monotonic, no instability
# 3e-3 also moves accuracy (0.7824 -> 0.7838) faster than any LR tried, and reaches 2.911
# by b29 -- already past the old expert_thoughts "oracle" number (2.9165) from the
# untrained regime. Early stopping (PATIENCE=3) guards against late instability.
LRS="${LRS:-3e-3}"
# TRAIN on the full thought+action span (Act-PRM and expert_thoughts must learn to
# produce the thought). REPORTING is action-only on both splits: eval_actiononly_* on
# eval, train/actiononly_* on train. ACTION_ONLY=1 is the loss-masking ablation only.
export ACTION_ONLY="${ACTION_ONLY:-0}"
# 150-batch cap (was 60). The completed retail actions_only lr1e-3 arm was still
# descending at b59 -- b10..b59 = 3.8355 3.8099 3.7839 3.7576 3.7112 3.6702, monotonic,
# -4.31%, largest drop in the LAST interval -- so at 60 batches every arm would be
# compared at an arbitrary under-trained point. Early stopping (PATIENCE=3 on held-out
# action PPL) makes a generous cap cheap: arms that plateau stop themselves; only arms
# that are genuinely still learning spend the extra time.
export NUM_BATCHES="${NUM_BATCHES:-150}"
# Best-checkpoint thought corpora only. The _last flavours (relabelled from the EM
# step_last rather than step_best) were within noise of their best counterparts in the
# lr=4e-5 results (retail hide subspan: thoughts_base 3.173 vs _last 3.193;
# thoughts_policy 3.182 vs _last 3.208), and dropping them takes the matrix from 72
# runs to 48. Set VARIANTS="" to sweep all six again.
export VARIANTS="${VARIANTS-actions_only expert_thoughts thoughts_policy thoughts_base}"
ENVS="${ENVS:-act_prm/tau2_retail act_prm/tau2_airline act_prm/snorkel_finance_split}"
DRY="${DRY:-0}"
MDIR=/tmp/aprm/lrmatrix; mkdir -p "$MDIR"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$MDIR/matrix.log"; }

log "=== Stage-2 SFT LR matrix: lrs='$LRS' envs='$ENVS' ==="
for env in $ENVS; do
  corpus="data/sft_corpus/${env##*/}"
  [ -d "$corpus" ] || { log "WARN: no corpus at $corpus — thought arms for $env will be skipped"; }
done

# LR-major so a full LR tier finishes across all datasets before the next starts:
# if 1e-3 turns out to diverge, the 1e-4 tier is already complete and usable.
# REGIME-MAJOR: every hide arm across all three datasets first, then full-context.
# The hide regime is the one the Stage-3 RL pipeline consumes, so it is the set worth
# completing first; full-context is the replication check.
for regime in ${REGIMES:-hide full}; do
for lr in $LRS; do
  for env in $ENVS; do
    log "--- regime=$regime lr=$lr env=$env"
    if [ "$DRY" = 1 ]; then
      LR="$lr" SFT_DRY_RUN=1 ./scripts/run_sft_sweep.sh "$env" 2>&1 | sed 's/^/    /'
      continue
    fi
    LR="$lr" REGIMES="$regime" ./scripts/run_sft_sweep.sh "$env" >> "$MDIR/lr${lr}_${regime}_${env##*/}.log" 2>&1 \
      && log "regime=$regime lr=$lr $env: sweep done" || log "regime=$regime lr=$lr $env: sweep FAILED"
  done
  log "=== regime=$regime lr=$lr complete across all datasets ==="
  # Refresh the per-dataset SFT notes/CSVs as each tier lands, so results are
  # readable without waiting for the whole matrix.
  for env in $ENVS; do
    uv run --no-project python scripts/analyze_sft.py "$env" >/dev/null 2>&1 || true
  done
done
done

# Marker so watch_sft_sweep.sh stops relaunching a no-op matrix every 5 min and exits.
touch "$MDIR/DONE"
log "=== LR matrix done ==="
