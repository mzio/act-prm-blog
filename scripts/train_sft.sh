#!/usr/bin/env bash
# Stage 2 — SFT (with hide-observations) on tau2/finance expert data.
#
# Four variants (each SFTs a fresh LoRA; hide_observations=true compacts the
# context to: system + first user + last observation + ALL model messages):
#   actions_only    - expert action-only baseline (no thoughts)          -> generator act_prm_actions_only
#   thoughts_policy - Act-PRM thoughts+actions, best thought scored by the POLICY
#   thoughts_base   - Act-PRM thoughts+actions, best thought scored by the BASE
#   expert_thoughts - target = ORIGINAL expert reasoning+action (keep_expert_thoughts)
#
# The SFTTrainer does plain (advantage-weighted) CE; advantage_mode=best +
# drop_zero_advantage (sft.yaml) means only the single best thought+action per
# step is trained (MLE). rl_eval tasks are held out by the split. Eval logs
# action-token PPL + accuracy every --eval_every; add --best_metric eval_action_ppl
# to early-stop on lowest eval PPL.
#
# FIXED CORPUS (recommended for the aprm variants; Stage 1.5): first export a
# committed thought corpus from a Stage-1 relabel pass, then SFT directly on it:
#   uv run python scripts/export_sft_corpus.py --generations logs/<run>/generations.jsonl \
#       --source-pools data/tau2_retail --out data/sft_corpus/tau2_retail/policy
#   ./scripts/train_sft.sh act_prm/tau2_retail thoughts_policy \
#       --dataset_path data/sft_corpus/tau2_retail/policy
# When --dataset_path is passed, the thought variants SFT on that corpus (its
# assistant content is already thought+action) via the actions_only generator —
# no on-the-fly re-generation. Without it they generate thoughts on the fly.
#
# NOTE on coverage: this trains on the split's act_prm_train tasks. To SFT over
# ALL non-rl_eval tasks (train+eval), regenerate the split with everything in
# train:  uv run python scripts/make_split.py --dataset <ds> --name <name> \
#           --train_frac 0.85 --eval_frac 0.0   (rl_eval = the remainder)
#
# Usage:
#   ./scripts/train_sft.sh <env> <variant> [extra main_pytorch flags]
#   env     : act_prm/tau2_retail | act_prm/tau2_airline | act_prm/snorkel_finance_split
#   variant : actions_only | thoughts_policy | thoughts_base | expert_thoughts | expert_thoughts_all
# e.g.:
#   CUDA_VISIBLE_DEVICES=0 ./scripts/train_sft.sh act_prm/tau2_retail expert_thoughts
#
# Set SFT_DRY_RUN=1 to print the resolved command instead of running it.
set -euo pipefail
cd "$(dirname "$0")/.."
ENVCFG="${1:?env, e.g. act_prm/tau2_retail}"
VARIANT="${2:?variant: actions_only | thoughts_policy | thoughts_base | expert_thoughts}"
shift 2 || true
ENVNAME="${ENVCFG##*/}"
# Model is parametrized: MODEL_CFG selects both the --model_config AND (downstream, via
# main_pytorch) the <MODEL> dir in checkpoint/log paths. Default keeps the 4B behavior.
MODEL_CFG="${MODEL_CFG:-hf_qwen3_4b_instruct}"

export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/usr/local/bin:$PATH"
# HuggingFace access (stream datasets / load models): Meta forward proxy + HF token.
export https_proxy="${https_proxy:-http://fwdproxy:8080}"
export http_proxy="${http_proxy:-http://fwdproxy:8080}"
export HF_TOKEN="${HF_TOKEN:-$(cat "$HOME/models/token" 2>/dev/null || true)}"
# fwdproxy began 403-ing huggingface.co on 2026-09-03, which killed model loading in
# load_hf_model_and_tokenizer -> hf_hub list_repo_tree (all 4 expert_thoughts_all runs
# and the insurance base rollout died in ~25s). The weights are cached locally under
# HF_HOME, so go offline and never touch the Hub. Verified: AutoConfig+AutoTokenizer
# for Qwen3-4B-Instruct-2507 load fine with HF_HUB_OFFLINE=1.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_HOME="${HF_HOME:-/data/users/mzio/models/hf_cache}"

# Did the caller point at a pre-exported SFT corpus (piece A)?
HAS_DATASET_PATH=0
for a in "$@"; do [[ "$a" == "--dataset_path" ]] && HAS_DATASET_PATH=1; done

case "$VARIANT" in
  actions_only)
    GEN="act_prm_actions_only"; MODE=() ;;
  thoughts_policy)
    if [[ "$HAS_DATASET_PATH" == 1 ]]; then
      GEN="act_prm_actions_only"; MODE=()                       # SFT on the exported policy corpus
    else
      GEN="act_prm"; MODE=(--advantage_mode best --no-score_with_base)  # generate on the fly
    fi ;;
  thoughts_base)
    if [[ "$HAS_DATASET_PATH" == 1 ]]; then
      GEN="act_prm_actions_only"; MODE=()                       # SFT on the exported base corpus
    else
      GEN="act_prm"; MODE=(--advantage_mode best --score_with_base)
    fi ;;
  expert_thoughts_all)
    # Expert reasoning+action, but trained ONLY on the turns that actually HAVE reasoning.
    # The plain expert_thoughts arm is ~50%% bare <tool_call> targets (46%% on airline), which
    # taught it not to think: at rollout it reasons before 0-8%% of its tool calls. Filtering
    # the targets keeps every turn in the context, so trajectories stay coherent.
    GEN="act_prm_actions_only"; MODE=(--keep_expert_thoughts --require_thought)
    if [ "$HAS_DATASET_PATH" = 0 ]; then
      # EXPERT_POOL, as in the expert_thoughts branch: the derived path resolves to a pool
      # whose eval set differs from the base (retail 10 vs 8; finance 3/25 overlap), which
      # silently makes the arm non-comparable.
      MODE+=(--dataset_path "${EXPERT_POOL:-data/${ENVNAME}_expert_thoughts}")
    fi
    ;;
  expert_thoughts)
    # Target = original expert reasoning+action (keep_expert_thoughts loader flag).
    # Cached pools are keyed by dataset_path, so use a distinct one unless overridden.
    GEN="act_prm_actions_only"; MODE=(--keep_expert_thoughts)
    if [[ "$HAS_DATASET_PATH" == 0 ]]; then
      MODE+=(--dataset_path "${EXPERT_POOL:-data/${ENVNAME}_expert_thoughts}")
    fi ;;
  *) echo "bad variant '$VARIANT' (actions_only|thoughts_policy|thoughts_base|expert_thoughts)"; exit 1 ;;
esac

# Context regime: hide-obs by default. SFT_FULLCTX=1 -> full context (omit the flag;
# --hide_observations is store_true/default None, so omitting it lets the env yaml's
# hide_observations:false win). Same pools either way — hide-obs is applied at
# tokenization, not baked into the cached corpus.
HIDE_OBS=(--hide_observations); [ "${SFT_FULLCTX:-0}" = 1 ] && HIDE_OBS=()

CMD=(uv run python main_pytorch.py
  --env_config "$ENVCFG" --model_config "$MODEL_CFG"
  --lora_config r8_a16_linear --generator_config "$GEN" --trainer_config "${TRAINER_CFG:-sft}"
  --replay_buffer_config default "${HIDE_OBS[@]}"
  --group_size 4 --batch_size 4 --num_batches 60 --eval_every 10 --no_initial_eval
  --length_penalty 0.15 "${MODE[@]}" --verbose "$@")

if [[ "${SFT_DRY_RUN:-0}" == 1 ]]; then
  printf '%s ' "${CMD[@]}"; echo; exit 0
fi
exec "${CMD[@]}"
