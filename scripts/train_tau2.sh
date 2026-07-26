#!/usr/bin/env bash
# tau2-bench agentic RL: a Qwen3 LoRA policy acts against real tau2 tasks/tools,
# with the USER SIMULATOR + NL-assertion judge driven by Claude via the Claude
# Agent SDK (claude_agent_sdk/<model>). GRPO-style advantages (hf_grpo) + PG (pg).
#
# Runs in the dedicated .venv-tau2 (which has the training stack + tau2 + litellm
# + claude-agent-sdk), so it never disturbs the base .venv used by other runs.
#
# ── One-time setup ─────────────────────────────────────────────────────────────
#   1. Clone tau2-bench. GitHub is blocked by the agent's fwdproxy, so clone it
#      from a shell that reaches github (your interactive terminal does):
#        git clone --branch v1.0.0 https://github.com/sierra-research/tau2-bench.git
#      (lands at ./tau2-bench — gitignored; the env reads data from tau2-bench/data)
#   2. Build the tau2 venv (base deps hardlink from uv's cache; only tau2/litellm
#      /claude-agent-sdk are new):
#        export https_proxy=http://fwdproxy:8080 http_proxy=http://fwdproxy:8080
#        UV_PROJECT_ENVIRONMENT=.venv-tau2 uv sync --extra tau2
#
# ── Auth for the Claude user simulator ─────────────────────────────────────────
#   On a devserver with Claude Code logged in, the SDK uses that ambient session —
#   no key needed. For a HEADLESS run (no Claude login), put your OAuth token in
#   .env:  CLAUDE_CODE_OAUTH_TOKEN=sk-ant-oat01-...   (.env is gitignored)
#   To use the Llama-passthrough user sim instead, set the env config's user_llm to
#   metagen/<model> and put LLAMA_API_KEY (LLM|<id>|<secret>) in .env.
#
# ── Usage ──────────────────────────────────────────────────────────────────────
#   ./scripts/train_tau2.sh                                    # airline defaults
#   ./scripts/train_tau2.sh --env_config tau2bench/retail
#   ./scripts/train_tau2.sh --num_batches 50 --max_turns 12   # override anything
set -euo pipefail
cd "$(dirname "$0")/.."

# Model loads from the local HF cache; tau2 data is local; the Claude user-sim
# reaches the API via the SDK. No fwdproxy needed at run time.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

UV_PROJECT_ENVIRONMENT=.venv-tau2 exec uv run --no-sync python main_pytorch.py \
  --env_config tau2bench/airline \
  --model_config hf_qwen3_4b_instruct \
  --lora_config r8_a16_linear \
  --generator_config hf_grpo \
  --trainer_config pg \
  --replay_buffer_config default \
  --num_train_tasks 40 --num_test_tasks 10 \
  --batch_size 2 --group_size 4 \
  --max_turns 8 --max_tokens 256 \
  --num_batches 25 --eval_every 5 --no_initial_eval \
  --verbose "$@"
