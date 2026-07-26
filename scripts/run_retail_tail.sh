#!/usr/bin/env bash
# Autonomous tail of the retail pipeline, run AFTER the hide-obs best-corpus SFTs
# (run_retail_matrix.sh). Sequential; each sub-step self-gates on the GPU + is resumable:
#   1) step_last relabel + export  -> data/sft_corpus/tau2_retail/{policy_last,base_last}
#   2) full SFT sweep              -> {actions_only, expert_thoughts,
#                                      thoughts_{policy,base}x{best,last}} x {hide,full}
#   3) Stage-3                     -> analysis (synced) + RL smoke gate + RL-from-SFT matrix
#
# Usage: CUDA_VISIBLE_DEVICES=0 nohup ./scripts/run_retail_tail.sh > /tmp/aprm/tail.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
G="${CUDA_VISIBLE_DEVICES:-0}"
L=/tmp/aprm/tail.log; mkdir -p /tmp/aprm
echo "[$(date '+%m-%d %H:%M:%S')] === retail tail start (GPU $G) ===" | tee -a "$L"

CUDA_VISIBLE_DEVICES=$G ./scripts/run_relabel.sh act_prm/tau2_retail >> "$L" 2>&1 \
  || echo "[tail] run_relabel nonzero (continuing)" | tee -a "$L"
CUDA_VISIBLE_DEVICES=$G ./scripts/run_sft_sweep.sh    act_prm/tau2_retail >> "$L" 2>&1 \
  || echo "[tail] run_sft_sweep nonzero (continuing)" | tee -a "$L"
CUDA_VISIBLE_DEVICES=$G ./scripts/run_retail_stage3.sh                    >> "$L" 2>&1 \
  || echo "[tail] run_retail_stage3 nonzero" | tee -a "$L"

echo "[$(date '+%m-%d %H:%M:%S')] === retail tail done ===" | tee -a "$L"
