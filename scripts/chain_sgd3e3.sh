#!/usr/bin/env bash
# After the SGD 1e-3 experiment (train + 4 rollouts) completes, train sft_flat at SGD 3e-3.
#
# Why 3e-3: it is the LR behind the only Stage-2 checkpoints that ever rolled out well
# (20.5%), and SGD 1e-3 on the current pipeline landed at median |B@A| 3.90e-05 -- within
# 20% of that checkpoint's 4.71e-05 -- but with a worse PPL fit (2.75 vs 2.34). 3e-3 should
# close the fit gap at the same drift regime, giving a like-for-like reproduction on
# sft_flat. NOTE 3e-3 was never actually probed against anything ABOVE it (the original
# probe was {1e-4, 1e-3, 3e-3}), so it is a boundary value, not a located optimum.
#
# TRAINING ONLY here, as asked -- rollouts are a separate decision once we see the fit.
set -uo pipefail
cd /home/mzio/projects/act-prm-blog
L=/tmp/aprm/sgd_flat/chain3e3.log
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$L"; }
log "waiting for the SGD 1e-3 experiment (train + rollouts) to finish"
while ps -eo args | grep -qE '[r]un_sgd_flat\.sh|[c]hain_sgd_newcorpus\.sh'; do sleep 120; done
while pgrep -f 'main_pytorch\.py' >/dev/null 2>&1; do sleep 60; done
log "GPU free -> SGD 3e-3 training (sft_flat, nb200, spb32)"
ARMS="thoughts_policy_adamw30 actions_only" LRS="3e-3" \
  ./scripts/run_sgd_flat.sh >> "$L" 2>&1
log "SGD 3e-3 phase rc=$?"
