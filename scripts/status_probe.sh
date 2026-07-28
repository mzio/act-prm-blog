#!/usr/bin/env bash
# Lightweight status probe: append a compact pipeline snapshot to
# /tmp/aprm/status_probe.log every INTERVAL seconds (default 30 min). Read it any
# time with:  tail -n 40 /tmp/aprm/status_probe.log
#   nohup ./scripts/status_probe.sh > /dev/null 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
INTERVAL="${1:-1800}"
LOG=/tmp/aprm/status_probe.log
CK=checkpoints_lora/act_prm_snorkel_finance_split/hf_qwen3_4b_instruct
LG=logs/act_prm_snorkel_finance_split/hf_qwen3_4b_instruct
emc(){ local m; m=$(ls $LG/snorkel_finance_split_s1_$1-*/metrics.jsonl 2>/dev/null|head -1)
  python3 -c "import json;print(sum(1 for l in open('$m') if l.strip() and 'train/loss' in l))" 2>/dev/null || echo '?'; }
while true; do
  {
    echo "===== $(date '+%m-%d %H:%M:%S') ====="
    echo "EM: policy $(emc policy)/58  base $(emc base)/58"
    printf "SFT step_best exists: "
    for t in actions_only_heldout actions_only_heldout_fullctx \
             expert_thoughts_heldout expert_thoughts_heldout_fullctx \
             thoughts_base_heldout thoughts_base_heldout_fullctx \
             thoughts_policy_heldout thoughts_policy_heldout_fullctx \
             thoughts_policy_last_heldout thoughts_policy_last_heldout_fullctx \
             thoughts_base_last_heldout thoughts_base_last_heldout_fullctx; do
      ls -d $CK/snorkel_finance_split_s2_${t}-*/step_best/adapter_model.safetensors >/dev/null 2>&1 && printf "%s " "$t"
    done; echo
    for c in policy base policy_last base_last; do
      n=$(python3 -c "import json;print(len(json.load(open('data/sft_corpus/snorkel_finance_split/$c/train.json'))))" 2>/dev/null || echo '-')
      printf "corpus %s: %s  " "$c" "$n"
    done; echo
    echo "running: $(pgrep -af '[m]ain_pytorch.py' | grep -oE 'run_tag [^ ]+' | sed 's/run_tag //' | sort | uniq -c | tr '\n' ';' | sed 's/  */ /g')"
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | sed 's/^/  GPU /'
  } >> "$LOG" 2>&1
  sleep "$INTERVAL"
done
