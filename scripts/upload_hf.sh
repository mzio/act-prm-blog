#!/usr/bin/env bash
# Upload the snorkel_finance Act-PRM SFT LoRA checkpoints (+ the relabel corpora) to
# HuggingFace, mirroring https://huggingface.co/mzio/aprm-sft-tau2-airline.
#
# RUN FROM A SHELL WITH HF WRITE EGRESS + TOKEN (the agent's egress is CDN-blocked for
# LFS uploads, so this must be run by you):
#   export HF_TOKEN=<your write token>
#   export https_proxy=http://fwdproxy:8080 http_proxy=http://fwdproxy:8080 HF_HUB_DISABLE_XET=1
#   ./scripts/upload_hf.sh
#
# Re-runnable: stages whatever step_best checkpoints exist now (uploads partial today,
# re-run later to add the rest). Override repo ids via env:
#   MODEL_REPO=mzio/aprm-sft-snorkel-finance  DATASET_REPO=mzio/aprm-sft-corpus-snorkel-finance
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL_REPO="${MODEL_REPO:-mzio/aprm-sft-snorkel-finance}"
DATASET_REPO="${DATASET_REPO:-mzio/aprm-sft-corpus-snorkel-finance}"
CK=checkpoints_lora/act_prm_snorkel_finance_split/hf_qwen3_4b_instruct
STAGE=/tmp/aprm/hf_upload/aprm-sft-snorkel-finance
rm -rf "$STAGE"; mkdir -p "$STAGE"

# run_tag suffix : clean checkpoint name (variant-regime)
MAP="
actions_only_heldout:actions_only-hide
actions_only_heldout_fullctx:actions_only-full
expert_thoughts_heldout:expert_thoughts-hide
expert_thoughts_heldout_fullctx:expert_thoughts-full
thoughts_policy_heldout:aprm_policy_best-hide
thoughts_policy_heldout_fullctx:aprm_policy_best-full
thoughts_policy_last_heldout:aprm_policy_last-hide
thoughts_policy_last_heldout_fullctx:aprm_policy_last-full
thoughts_base_heldout:aprm_base_best-hide
thoughts_base_heldout_fullctx:aprm_base_best-full
thoughts_base_last_heldout:aprm_base_last-hide
thoughts_base_last_heldout_fullctx:aprm_base_last-full
"
n=0
for row in $MAP; do
  tag="${row%%:*}"; name="${row##*:}"
  sb=$(ls -dt "$CK"/snorkel_finance_split_s2_${tag}-*/step_best 2>/dev/null | head -1)
  if [ -n "$sb" ] && [ -f "$sb/adapter_model.safetensors" ]; then
    mkdir -p "$STAGE/$name"; cp -f "$sb"/adapter_config.json "$sb"/adapter_model.safetensors "$STAGE/$name/" 2>/dev/null || true
    echo "staged: $name"; n=$((n+1))
  fi
done
echo "staged $n / 12 SFT checkpoints"

cat > "$STAGE/README.md" <<'EOF'
---
license: apache-2.0
base_model: Qwen/Qwen3-4B-Instruct-2507
library_name: peft
tags: [act-prm, lora, snorkel-finance, sft]
---
# aprm-sft-snorkel-finance

Act-PRM SFT LoRA adapters (Qwen3-4B-Instruct-2507) on the **snorkel_finance** agent
traces. Each dir is a fresh SFT checkpoint (early-stopped on held-out action ppl):

- `actions_only-{hide,full}` — expert action-only baseline
- `expert_thoughts-{hide,full}` — expert reasoning+action ("oracle")
- `aprm_{policy,base}_{best,last}-{hide,full}` — Act-PRM inferred thought+action,
  thoughts relabeled TOP-1 (argmax length-penalized action likelihood) from the EM
  policy/base scorer at step_best/step_last.

`hide` = observations hidden at SFT context (system + first user + last obs + model
turns); `full` = full context. See the act-prm-blog repo (cc-finance-1.x) for the
pipeline + the whole-span vs action-subspan analysis.
EOF

echo "=== uploading $n checkpoints -> $MODEL_REPO (model) ==="
hf upload "$MODEL_REPO" "$STAGE" . --repo-type model

echo "=== uploading corpora -> $DATASET_REPO (dataset) ==="
CORP=data/sft_corpus/snorkel_finance_split
for c in base base_last policy policy_last; do
  [ -f "$CORP/$c/train.json" ] && hf upload "$DATASET_REPO" "$CORP/$c" "$c" --repo-type dataset && echo "uploaded corpus $c"
done
echo "DONE. Model: https://huggingface.co/$MODEL_REPO  Dataset: https://huggingface.co/datasets/$DATASET_REPO"
