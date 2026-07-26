# cc-2.1 — SFT context regimes: hide-obs vs full-context

A second Stage-2 axis (orthogonal to the 4 target variants in `cc-2.0`): **how much of
the observation history is in the SFT context.** Each variant is run under both regimes,
so the full Stage-2 matrix is **4 variants × 2 regimes = 8 runs**.

## The two regimes
- **hide-obs** (`ho=1`, default): context compacted to `system + first user msg +
  first_obs_to_show + last_obs_to_show observations + ALL model (thought+action) messages`;
  the middle tool observations are dropped. This is the Act-PRM setup — it forces the
  thoughts to carry the observation information forward.
- **full-context** (`ho=0`): every observation kept. This is the env default
  (`hide_observations: false`) and matters for train/eval consistency with Stage-3 RL,
  where the tau2 gym always provides full observations.

Both regimes read the **same** cached corpus/pools — hide-obs is applied at
tokenization/context-construction time, not baked into the data.

## How it's wired (the flag)
`--hide_observations` in `main_pytorch.py` is `action="store_true", default=None`, so:
- **pass it** → `True` (hide-obs).
- **omit it** → `None` → the env yaml wins (`hide_observations: false` → full-context).
- There is **no** `--no-hide_observations` form.

`configs/environments/act_prm/*.yaml` set `hide_observations: false`,
`first_obs_to_show: 1`, `last_obs_to_show: 1` (the last two only take effect when hiding).

## How to run each

### `scripts/train_sft.sh` — a single run
Controlled by the `SFT_FULLCTX` env var (default hide-obs):
```bash
# hide-obs (default):
./scripts/train_sft.sh act_prm/tau2_retail thoughts_policy --dataset_path data/sft_corpus/tau2_retail/policy
# full-context:  set SFT_FULLCTX=1  ->  train_sft.sh drops --hide_observations
SFT_FULLCTX=1 ./scripts/train_sft.sh act_prm/tau2_retail thoughts_policy --dataset_path data/sft_corpus/tau2_retail/policy
```
Internals (`train_sft.sh`): `HIDE_OBS=(--hide_observations); [ "$SFT_FULLCTX" = 1 ] && HIDE_OBS=()`.

### `scripts/run_sft_sweep.sh <env>` — the whole 8-run sweep
Runs all 4 variants under **both** regimes, resumable, GPU-serialized:
```bash
CUDA_VISIBLE_DEVICES=0 ./scripts/run_sft_sweep.sh act_prm/tau2_retail
```
It loops `for regime in hide full`, setting `SFT_FULLCTX=1` for the `full` pass and
appending a `_fullctx` suffix to the run_tag. (`run_retail_matrix.sh` only did the hide-obs
pass inline; the queued full-context pass on retail is `run_sft_sweep.sh`, which skips the
already-done hide-obs runs.)

## Naming / where results land
- Run tags: `<domain>_s2_<variant>_heldout` (hide-obs) and `<domain>_s2_<variant>_heldout_fullctx`
  (full-context) → distinct `checkpoints_lora/…/<tag>-…/step_best` and `logs/…/<tag>-…/metrics.jsonl`.
- The `ho=0` vs `ho=1` token in the auto-encoded run_name is a second confirmation of the regime.
- Compare curves across regimes with `notebooks/cc-1.1` (both write `train/loss`, `train/ppl`,
  `eval/eval_action_ppl`, `eval/eval_action_accuracy`).

## Caveat
For a full-context SFT, the model sees observations during training; if the downstream use
hides them, that's a mismatch — keep the regime consistent between SFT init and its Stage-3
RL / eval. See `cc-2.0` for the target variants and `cc-1.0` for the pipeline.
