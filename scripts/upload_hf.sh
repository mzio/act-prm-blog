#!/usr/bin/env bash
# Publish the snorkel_finance Act-PRM artifacts to HuggingFace (PUBLIC), mirroring the
# airline repos:
#   model:   mzio/aprm-sft-snorkel-finance
#     <tag>/adapter_{config.json,model.safetensors}   # 12 SFT step_best LoRA adapters,
#            one per-variant subfolder (variant_regime, e.g. thoughts_policy_heldout,
#            actions_only_heldout_fullctx)
#   dataset: mzio/aprm-thought-generations-snorkel-finance
#     generations/{policy,base,policy_last,base_last}.jsonl   # PRIMARY: full candidate
#            pool per logged action (thoughts[G], likelihoods p(x|s,z), rewards, advantages,
#            thought_tokens, best index) -> consumers can take top-1 OR reweight
#     sft_corpus_top1/<variant>/{train,eval,meta}.json        # baked top-1 SFT corpus
#     README.md
# Requires HF_TOKEN (write) in env. Idempotent.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${HF_TOKEN:?set HF_TOKEN=<write token>}"
export https_proxy="${https_proxy:-http://fwdproxy:8080}" http_proxy="${http_proxy:-http://fwdproxy:8080}" HF_HUB_DISABLE_XET=1
MODEL_REPO="${MODEL_REPO:-mzio/aprm-sft-snorkel-finance}"
DS_REPO="${DS_REPO:-mzio/aprm-thought-generations-snorkel-finance}"
CK=checkpoints_lora/act_prm_snorkel_finance_split/hf_qwen3_4b_instruct
LG=logs/act_prm_snorkel_finance_split/hf_qwen3_4b_instruct
HF=.venv/bin/hf

echo "===== 1) MODEL repo: $MODEL_REPO (12 SFT adapters) ====="
n=0
for d in $(ls -d $CK/snorkel_finance_split_s2_*/step_best 2>/dev/null); do
  [ -f "$d/adapter_model.safetensors" ] || continue
  tag=$(echo "$d" | grep -oE 'snorkel_finance_split_s2_[a-z_]+heldout(_fullctx)?' | head -1 | sed 's/snorkel_finance_split_s2_//')
  echo "== $tag =="
  $HF upload "$MODEL_REPO" "$d" "$tag" --repo-type model 2>&1 | tail -1
  n=$((n+1))
done
echo "uploaded $n adapters"

echo "===== 2) DATASET repo: $DS_REPO ====="
STAGE=/tmp/aprm/hf_ds_stage; rm -rf "$STAGE"; mkdir -p "$STAGE/generations" "$STAGE/sft_corpus_top1"
for v in policy base policy_last base_last; do
  gen=$(ls -dt $LG/snorkel_finance_split_s1relabel_${v}-*/generations.jsonl 2>/dev/null | head -1)
  [ -n "$gen" ] && cp "$gen" "$STAGE/generations/$v.jsonl" && echo "staged generations/$v.jsonl ($(wc -l <"$gen") rows)"
  [ -d "data/sft_corpus/snorkel_finance_split/$v" ] && cp -r "data/sft_corpus/snorkel_finance_split/$v" "$STAGE/sft_corpus_top1/$v"
done
cat > "$STAGE/README.md" <<'EOF'
---
license: apache-2.0
task_categories: [question-answering]
tags: [act-prm, thought-generation, snorkel-finance, agent, sft]
---
# aprm-thought-generations-snorkel-finance

Act-PRM inferred-thought generations for **snorkel_finance** agent traces
(Qwen3-4B-Instruct-2507). For each logged (state `s`, action `x`), an offline EM
samples `G=4` candidate thoughts `z`, scores each by the length-penalized action
likelihood, and commits the top-1.

## `generations/{policy,base,policy_last,base_last}.jsonl` (primary)
One row per logged action step. Full candidate pool so you can take top-1 OR recompute
any weighting:
- `thoughts` (G): candidate thoughts z
- `likelihoods` (G): p(x | s, z), policy per-action-token likelihood
- `rewards` (G): length-penalized score = `p(x|s,z) - 0.15 * (|z| / 96)`
- `advantages` (G): group-normalized EM weights
- `thought_tokens` (G): |z| per candidate
- `best`: index of the committed top-1 (argmax reward)
- `target_action`, `sample_id`, `timestep`, `split`

Variants = EM scorer (`policy` = LoRA policy likelihood, `base` = frozen base model)
× EM snapshot (`_last` = final step_last checkpoint; else step_best).

## `sft_corpus_top1/<variant>/{train,eval,meta}.json` (convenience)
The baked top-1 SFT corpus: assistant target = `thoughts[best] + "\n\n" + action`.

## Result
On next-action prediction (action-subspan ppl), all Act-PRM variants beat both the
expert-reasoning oracle and the action-only baseline in hide-obs and full-context.
See the act-prm-blog `cc-finance-1.x` notes/notebooks.
EOF
$HF upload "$DS_REPO" "$STAGE" . --repo-type dataset 2>&1 | tail -1

echo "===== 3) set PUBLIC + verify ====="
.venv/bin/python - "$MODEL_REPO" "$DS_REPO" <<'PY'
import os, sys
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
for repo, typ in [(sys.argv[1], "model"), (sys.argv[2], "dataset")]:
    for fn in ("update_repo_settings", "update_repo_visibility"):
        try:
            getattr(api, fn)(repo_id=repo, private=False, repo_type=typ); break
        except Exception as e:
            last = e
    info = api.model_info(repo) if typ == "model" else api.dataset_info(repo)
    sib = len(info.siblings or [])
    print(f"{typ:8s} {repo}: private={info.private}  siblings={sib}")
PY
echo "DONE:"
echo "  https://huggingface.co/$MODEL_REPO"
echo "  https://huggingface.co/datasets/$DS_REPO"
