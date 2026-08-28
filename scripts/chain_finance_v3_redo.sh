#!/usr/bin/env bash
# Wait for the running insurance Stage-1 to finish, then redo finance Stage-1 on the
# v3 (question-level, leak-free) split.
#
# Why the redo: the 08-28 finance adamw30 corpus was exported against a DIFFERENT
# finance split than the run trained on, so it landed with 94/116 train and 3/25 eval
# trajectories -- and the run itself trained/evaluated on the leaky uid-level split.
# snorkel_finance_split.yaml now points at _v3 (matching the driver's --source-pools)
# and export_sft_corpus.py aborts under --min-coverage, so this rerun is clean on both.
set -uo pipefail
cd /home/mzio/projects/act-prm-blog
L=/tmp/aprm/stage1_r32/chain_finance_v3.log
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" >> "$L"; }

log "waiting for insurance Stage-1 to clear the GPU"
while pgrep -f 'main_pytorch\.py' >/dev/null || pgrep -f 'run_stage1_r32\.sh' >/dev/null; do sleep 300; done
log "GPU free; starting finance v3 redo"

# force a fresh run even though finance_s1em_policy_adamw30.done exists
rm -f /tmp/aprm/stage1_r32/finance_s1em_policy_adamw30.done
ONLY_DOMAINS="finance" OPTIMIZER=adamw LR=4e-5 EM_NB=30 LENGTH_PENALTY=0 \
  TAGSFX=_adamw30 SAVE_EVERY=10 ./scripts/run_stage1_r32.sh >> "$L" 2>&1
log "finance v3 redo exited rc=$?"
