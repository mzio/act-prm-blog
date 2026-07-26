# CLAUDE.md

Guidance for working in this repository.

## What this is

**Act-PRM** (Action Process Reward Models): infer the latent *thoughts* behind
logged, action-only agent demonstrations via an offline **EM**. For each logged
action `x` in state `s`: sample candidate thoughts `z`, reward each by the
length-penalized action likelihood `p(x | s, z)`, group-normalize to EM weights,
and take a policy-gradient / REINFORCE step on the `(thought + action)` tokens.

This repo is **both** a static blog (`index.html`, `assets/`, `notebooks/`) and a
uv Python package for the PyTorch training path. The distribution is `act-prm`;
the import package is `act_prm` (under `src/`).

The PyTorch training stack is a **lean fork of `show-don't-tell-rl`** (import
package `strl`, at `../hatch-strl`): the generator-harness registry, the
policy-gradient `trainer`, `lora`, and `replay_buffer` types were copied and
their imports rewritten `strl.` → `act_prm.`. The Act-PRM-specific pieces are:
- `src/act_prm/generator/act_prm/` — the EM E-step harness (`ActPrmGenerator`).
- `src/act_prm/environments/act_prm_traces/` — logged action-only trajectories.

The original **Tinker** training path still lives in `scripts/` +
`notebooks/act_prm_tinker.ipynb` (unchanged, reference only); the didactic
from-scratch PyTorch version is `notebooks/act_prm_transformers.ipynb`.

## Commands

- Setup: `uv sync`. (`uv sync --extra tinker` also installs the Tinker SDK for the
  legacy `scripts/` path.)
- Train (PyTorch LoRA): `uv run python main_pytorch.py [flags]`. Offline smoke test:
  ```bash
  uv run python main_pytorch.py \
    --env_config act_prm/snorkel_finance --model_config hf_qwen3_0_6b \
    --lora_config r8_a16_linear --generator_config act_prm --trainer_config pg \
    --replay_buffer_config default --synthetic \
    --group_size 4 --batch_size 2 --max_steps_per_traj 2 --num_batches 2 \
    --length_penalty 0.15 --verbose
  ```
  Drop `--synthetic` (and raise `--num_trajectories`) for the real Snorkel traces
  (the env streams them from the HF Hub). Default/recommended model is
  `--model_config hf_qwen3_4b_instruct` (Qwen3-4B-Instruct-2507, the paper's model);
  `hf_qwen3_8b` and `hf_qwen3_0_6b` (tiny/CPU) are also provided.
- Model/dataset access: the instruct models are cached at
  `/data/users/mzio/models/hf_cache/hub` (the configs' `cache_dir`); load them
  offline with `HF_HUB_OFFLINE=1`. This box has no *direct* internet, but Meta's
  forward proxy reaches HF — for real-trace runs (streaming) or fresh downloads,
  `export https_proxy=http://fwdproxy:8080 http_proxy=http://fwdproxy:8080` and
  `HF_TOKEN=$(cat /home/mzio/models/token)`. `scripts/prefetch_models.sh` sets both.

## How a run is wired

Config-driven via OmegaConf yamls under `configs/<kind>/<name>.yaml`
(`model/`, `lora/`, `environments/`, `generator/`, `trainer/`, `replay_buffer/`),
selected by CLI flags. **Override convention:** CLI override flags default to
`None`, so the yaml value wins unless the flag is explicitly passed; a flag whose
name matches a config key overrides that key in every config that has it
(`update_configs` in `main_pytorch.py`).

`main_pytorch.py`: configs → `load_llm` (HF policy) → `get_lora_model` (LoRA on by
default) → `load_env` (`act_prm_traces`) → `get_replay_buffer` → `get_trainer("rl")`
→ `trainer.train()`. The `RLTrainer` loop calls `ActPrmGenerator.do_group_rollout`
(the E-step) per logged trajectory, then does the importance-weighted M-step.

## Key contract (why the EM weights survive)

`ActPrmGenerator` builds each `EpisodeStep` with `advantage = EM weight` and
`advantage_is_computed = True`. `TrajectoryGroup.compute_advantages()` therefore
returns the stored weights unchanged (no return/mean-centering). `prepare_minibatch`
then trains the `(thought+action)` span (`state_action_tokens[state_len:]`) with
that advantage; `RLTrainer.compute_loss` is the importance-weighted PG loss.

## tau2-bench agentic RL (the eventual RL phase)

`src/act_prm/environments/tau2bench/` wraps τ²-bench's `AgentGymEnv` (real
airline/retail tasks, tools, and end-of-episode evaluator) as an `Environment`.
The LoRA policy does agentic rollouts (`hf_grpo` generator: env.reset → tool call
→ env.step, GRPO-mean-centered advantages) and the `pg` trainer takes the step —
no new trainer code. The **user simulator + NL-assertion judge** are driven by an
LLM through a litellm custom provider; two backends are wired:
- `claude_agent_sdk/<model>` → `litellm_claude_agent_sdk.py` → `ClaudeQueryLLM`
  (Claude Code OAuth; uses the ambient login on a devserver, else set
  `CLAUDE_CODE_OAUTH_TOKEN` in `.env` for headless runs). **Default.**
- `metagen/<model>` → `litellm_metagen.py` → Llama-API passthrough (needs
  `LLAMA_API_KEY` = `LLM|<id>|<secret>`).

Runs in a **dedicated `.venv-tau2`** (has the training stack + `tau2` + `litellm` +
`claude-agent-sdk`) so it never disturbs the base `.venv`. Setup + run:
`./scripts/train_tau2.sh` (see its header — clone `tau2-bench` from a github-capable
shell, `UV_PROJECT_ENVIRONMENT=.venv-tau2 uv sync --extra tau2`). tau2 tasks are
task-split-agnostic here; the `rl_eval` hold-outs from `data/splits/*` are the
intended eval tasks for this phase.

## Full pipeline (thought-gen → SFT → RL) + scripts

1. **Stage 1 — thought generation (Act-PRM EM)** on the expert datasets:
   `./scripts/train.sh --env_config act_prm/tau2_retail --generator_config act_prm
   --trainer_config pg [--score_with_base | --no-score_with_base] ...` (or add
   `--no_train` for a generate-only relabel pass). Generates over `act_prm_train`
   (+`act_prm_eval`), holding out `rl_eval`; saves every group to
   `<log_path>/generations.jsonl`.
2. **Stage 2 — SFT with hide-observations** (`scripts/train_sft.sh <env> <variant>`):
   `actions_only` (expert baseline), `thoughts_policy`, `thoughts_base`. Uses the
   `SFTTrainer` (plain weighted CE) + `advantage_mode best` + `--hide_observations`.
   For SFT over ALL non-`rl_eval` tasks, regenerate the split with `--eval_frac 0`.
3. **Stage 3 — RL from the SFT checkpoint** (`scripts/train_rl_from_sft.sh <domain>
   <sft_ckpt>`): warm-starts the SFT LoRA (`--resume_from`) and RLs on the tau2-gym
   env. (rl_eval hold-out → tau2-task-index mapping is still a TODO; see the script.)

Setup on a fresh box: `scripts/setup_new_box.sh`. Multi-run VRAM (H100 ~95 GiB):
uncapped Act-PRM EM peaks ~72 GiB/run (one per GPU); cap with `--obs_max_chars` or
`--group_size 2` to fit two per GPU.

## Gotchas

- The env tokenizer is loaded from `pretrained_model_config` then overwritten with
  `llm.tokenizer` in `main_pytorch.py` — keep them consistent (same model).
- `llm_handlers/__init__.py`, `generator/__init__.py`, `environments/__init__.py`
  are **trimmed** registries (HF / Act-PRM only) — don't reintroduce the dropped
  Claude/OpenAI/Meta/Tinker/AMI-bench branches from upstream `strl`.
- Logging uses `tinker_cookbook.utils.ml_log` (local JSON under `--log_path`, plus
  W&B if `--project_name` / `WANDB_API_KEY` are set) — importing it does **not**
  require a Tinker API key.
- **HuggingFace / GitHub access from the devserver:** huggingface.co isn't directly
  reachable but Meta's forward proxy is — `export https_proxy=http://fwdproxy:8080
  http_proxy=http://fwdproxy:8080` and `HF_TOKEN=$(cat ~/models/token)` (the run
  scripts do this). **github.com is BLOCKED by fwdproxy** (403), so git deps like
  `tau2-bench` must be cloned from a shell that reaches github, and pushes happen from
  your own terminal. Model/dataset caches live at `/data/users/$USER/models/hf_cache`
  (per-box, not dotsynced) — re-download on a fresh box.
- No test suite or linter is configured.
