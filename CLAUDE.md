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

## Gotchas

- The env tokenizer is loaded from `pretrained_model_config` then overwritten with
  `llm.tokenizer` in `main_pytorch.py` — keep them consistent (same model).
- `llm_handlers/__init__.py`, `generator/__init__.py`, `environments/__init__.py`
  are **trimmed** registries (HF / Act-PRM only) — don't reintroduce the dropped
  Claude/OpenAI/Meta/Tinker/AMI-bench branches from upstream `strl`.
- Logging uses `tinker_cookbook.utils.ml_log` (local JSON under `--log_path`, plus
  W&B if `--project_name` / `WANDB_API_KEY` are set) — importing it does **not**
  require a Tinker API key.
- No test suite or linter is configured.
