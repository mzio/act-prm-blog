# cc-airline-1.0 — Act-PRM tau2 **airline** pipeline (state, decisions, how-to)

Companion to the retail notes (`cc-1.0`, `cc-2.0`, `cc-2.1`). This file tracks the
**airline** domain run on the 2-GPU box (`devvm27322`), which mirrors the retail
matrix. All airline artifacts are prefixed `cc-airline-*` / `_airline` / `airline_`
to avoid clobbering retail's.

## Domain / split
- Dataset: `mzio/aprm-tau2-airline-gpt5m_med-gs4-s0-train` (GPT-5-mini rollouts,
  action-only for Act-PRM; expert reasoning is present in the raw assistant turns).
- Split (`data/splits/tau2_airline.json`, seed 0): **21 act_prm_train / 4 act_prm_eval
  / 6 rl_eval**. `rl_eval` is held out for Stage-3 RL eval only; Stage 1 + Stage 2 use
  the 21/4 train/eval.
- Model: Qwen3-4B-Instruct-2507 + LoRA (r8_a16_linear).

## Box specifics (offline)
This box's **agent egress can't reach HF's CDN/Xet** (fwdproxy 403 for
`agent_id:claude_code`) — only `huggingface.co` metadata. So model + dataset were
pre-fetched by the user; everything runs OFFLINE:
`HF_HOME=/data/users/mzio/models/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
HF_HUB_DISABLE_XET=1 UV_FROZEN=1`. `uv` was installed from the PyPI wheel (astral.sh +
pip both blocked). Trace pools were built offline from the cached parquet with
`scripts/prebuild_pools_offline.py` (the env's normal streaming load fails offline).

## Stages
### Stage 1 — Act-PRM EM thought-gen (DONE)
Two runs (one per scorer), uncapped, group 4, 25 batches, `--gradient_checkpointing`
(uncapped airline OOMs the M-step without it; ~77 GiB → ~34 GiB peak):
- policy-scored (`--no-score_with_base`, `swb=0`), base-scored (`--score_with_base`,
  `swb=1`). Both completed 25/25, 1241 generations each, `step_best` + `step_last` saved.
- `best_metric=final_reward` (mean held-out `p(x|s,z)`). **NOTE:** for both scorers the
  held-out eval reward peaked at the FIRST eval (step 5) and never improved
  (policy 0.449, base 0.421) → `step_best` is an *early* checkpoint. The EM eval reward
  is roughly flat (~0.44–0.55) on the small 4-task eval. → we relabel from BOTH
  `step_best` AND `step_last` and compare (user decision).

### Stage 1.5 — relabel (best-of-G) + export corpus
For each scorer × {best, last}: `main_pytorch --no_train --resume_from <ckpt>
--advantage_mode best --group_size 4 …` writes `generations.jsonl`; then
`scripts/export_sft_corpus.py --generations … --source-pools data/tau2_airline --out
data/sft_corpus/tau2_airline/<scorer>[_last]` reconstructs full-context state +
best thought+action per step. Corpora:
`data/sft_corpus/tau2_airline/{policy,base,policy_last,base_last}`.

### Stage 2 — SFT (fresh Qwen3-4B each), 12 runs
`scripts/run_airline_sweep.sh` (resumable, 2-GPU, 6/GPU). 6 variants × {hide-obs,
full-context}:
- `actions_only` (expert actions), `expert_thoughts` (raw reasoning+action, offline
  pool `data/tau2_airline_expert_thoughts`), `thoughts_policy`/`thoughts_base` from the
  **best** corpora, and `thoughts_policy`/`thoughts_base` from the **last** corpora.
- All: `--hide_observations` (default) vs `SFT_FULLCTX=1`, 60 batches, uncapped +
  `--gradient_checkpointing`, early-stop `--best_metric eval_action_ppl`.
- Context regimes (see cc-2.1): hide-obs = system + first user + first/last N obs + all
  model msgs (middle obs → "…"); full = keep all obs. Same corpus reused; hiding at
  tokenization.

### Stage 3 — RL from SFT (guarded)
`scripts/train_rl_from_sft.sh airline <step_best>` on the tau2-gym env (Claude user-sim).
**Open TODO:** map `rl_eval` uids → tau2 task indices so eval is strictly on the hold-out
(retail flagged this too). Needs `.venv-tau2` (built) + `./tau2-bench` (cloned).

## SFT metrics logged (per step, train + eval)
- train: `train/loss`, `train/ppl` (token ppl), `train/action_accuracy` (next-action-token acc).
- eval: `eval_action_ppl`, `eval_action_accuracy`, `eval_action_loss` (over the target span).
- Cross-regime eval (hide vs full for a given model) is done POST-HOC in a notebook
  (`cc-airline-*-sft-crossregime`) to keep the live eval path simple/reliable.

## Deliverables (this domain)
- notes: `cc-airline-1.0-pipeline.md` (this).
- notebooks (one per component): `cc-airline-1.1-stage1-em`, `cc-airline-1.2-stage15-corpus`,
  `cc-airline-1.3-stage2-sft`, `cc-airline-1.4-sft-crossregime` (+ Stage-3 later).

## Ops
- Orchestrator: `nohup ./scripts/run_airline_sweep.sh > /tmp/aprm/airline_sweep/run.log 2>&1 &`
  (watch `/tmp/aprm/airline_sweep/orchestrator.log`). Resumable.
- Kill: `pkill -f '[m]ain_pytorch.py'`. Snapshot: `./scripts/snapshot.sh` (per-host bundle).

## Addendum — why only 21 train tasks + rl_eval design
- Source dataset `mzio/aprm-tau2-airline-gpt5m_med-gs4-s0-train` has only **32 unique
  tasks** (unique_data_sample_id 0–31), not tau2 airline's full 50 — a custom seed-0
  subset. `make_split.py` keeps tasks with ≥1 successful rollout (done+reward>0, ≥2
  actions): **31/32 usable** (1 never succeeded). 70/15/15 → **21 train / 4 eval / 6 rl_eval**.
  So the 21 is dataset-limited, not a pipeline limit. (Finance is far bigger: 167 usable → 116 train.)
- **Better rl_eval idea (open):** use tau2 airline tasks NOT in the source dataset as the
  RL hold-out (purer: never-seen; also frees the 6 carved tasks back into train). Blocker:
  our 32 tasks span BOTH canonical tau2 splits (train:30/test:20 — ~14 of each are in the
  data by a noisy user_id match), so the tau2 "test" split is NOT clean-unseen. Needs the
  precise dataset→tau2 map (match on unique reservation-id) to compute the ~11–18 unseen
  complement — the same Stage-3 mapping TODO (cc-airline-2.0). Cleanest long-term: regen the
  source dataset over a known task split (train-only) and reserve tau2 `test` (20) as rl_eval.
  Tau2-specific — finance has no external task pool, so its rl_eval is carved from the dataset.
