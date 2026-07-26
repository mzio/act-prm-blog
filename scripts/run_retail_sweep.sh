#!/usr/bin/env bash
# Per-model Stage 1.5 + Stage 2 unit (NO RL): relabel(best+last) + export SFT corpora,
# then the full SFT matrix. Mirrors airline's run_airline_sweep.sh. Both sub-steps are
# already MODEL_CFG-aware (Task 1) and self-gate on the GPU + are resumable; this just
# chains them and threads MODEL_CFG through.
#
#   1) run_relabel.sh   -> data/sft_corpus/<envname>/{policy,base,policy_last,base_last}
#   2) run_sft_sweep.sh -> {actions_only, expert_thoughts, thoughts_{policy,base}x{best,last}}
#                          x {hide,full}, under checkpoints_lora/<envdir>/<MODEL>/...
#
# Assumes Stage-1 EM step_best/step_last already exist for this MODEL (run_em.sh for 8B;
# 4B EM is already done).
#
# Usage:  CUDA_VISIBLE_DEVICES=0 MODEL_CFG=hf_qwen3_8b ./scripts/run_retail_sweep.sh act_prm/tau2_retail
set -uo pipefail
cd "$(dirname "$0")/.."

ENVCFG="${1:?env config, e.g. act_prm/tau2_retail}"
# Model is parametrized: MODEL_CFG threads through to both sub-scripts (they resolve the
# same default and the <MODEL> path dir). Default keeps 4B behavior.
MODEL="${MODEL_CFG:-hf_qwen3_4b_instruct}"; export MODEL_CFG="$MODEL"

echo "[$(date '+%m-%d %H:%M:%S')] === retail sweep (relabel + SFT, no RL) for $ENVCFG (model=$MODEL) ==="
./scripts/run_relabel.sh   "$ENVCFG"
./scripts/run_sft_sweep.sh "$ENVCFG"
echo "[$(date '+%m-%d %H:%M:%S')] === retail sweep done for $ENVCFG (model=$MODEL) ==="
