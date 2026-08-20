# cc-finance-1.0 — Act-PRM **snorkel_finance** pipeline (state, design, results)

Finance analogue of the tau2 airline/retail work (`cc-airline-1.0`, `cc-1.0`). Runs
the Act-PRM EM → SFT pipeline on the **snorkel_finance** action-only traces and asks:
**do inferred latent thoughts help next-action prediction?** All artifacts are
prefixed `cc-finance-*` / `_split` to avoid clobbering the other domains. Box:
`devvm40015` (2×H100, offline).

## Domain / split
- Dataset: `mzio/aprm-snorkelai_agent_finance_reasoning` (finance-QA agent rollouts
  over real 10-K tables; tools: get_descriptions/get_table_info/sql_query/calculator/
  respond_user). Action = the `<tool_call>…</tool_call>` block or a `Final Answer:`.
- Split (`data/splits/snorkel_finance.json`, seed 0): **116 act_prm_train / 25
  act_prm_eval / 26 rl_eval** (167 usable of 357 trajectories after the success +
  `max_traj_timestep` filter). `rl_eval` is held out for Stage-3 RL eval, never seen
  by act-prm. Env config: `act_prm/snorkel_finance_split`.
- Model: Qwen3-4B-Instruct-2507 + LoRA (r8_a16_linear). (8B arm deferred.)

## Offline setup (this box)
Agent egress can't reach HF's CDN/Xet (`us.aws.cdn.hf.co` not allowlisted for
`agent:claude_code`) — models + dataset were prefetched by the user; everything runs
offline: `HF_HOME=/data/users/mzio/models/hf_cache HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1 UV_FROZEN=1`. Trace pools built offline
from the cached parquet via `scripts/prebuild_pools_offline.py` (116 train / 25 eval,
action-only + a `_expert_thoughts` variant).

## Stages
### Stage 1 — Act-PRM EM thought-gen (`run_actprm_pipeline.sh`, 2 GPUs)
Two runs, one per scorer, uncapped, group 4, **58 batches (= 2 epochs over 116
tasks)**, `--gradient_checkpointing`:
- **policy-scored** (`--no-score_with_base`) and **base-scored** (`--score_with_base`).
- E-step per logged (state, action): sample G=4 thoughts z (`thought_temperature 1.0`,
  `max_thought_tokens 96`), score by length-penalized action likelihood
  `reward = p(x|s,z) − 0.15·(|z|/96)` (`reward_method penalty`); M-step = REINFORCE on
  (thought+action) with group-normalized EM weights (`advantage_mode em`).
- `best_metric = final_reward` (mean held-out `p(x|s,z)`); saves `step_best` +
  `step_last` (rolling, `save_every 20`). **Observation (corrected, live @ batch ~40):**
  the scorers differ — **base plateaus early** (best stuck at step 15, reward 0.5776,
  unchanged ~11h; airline-like) while **policy keeps improving into epoch 2** (best
  climbed 5->35->40, reward 0.5752, still rising). So policy's 2-epoch EM earns its keep;
  base would be fine at 1 epoch. We relabel from BOTH `step_best` and `step_last`.

### Head start: base Act-PRM relabel from the current (plateaued) step_best
Base's `step_best` is stable (batch 15, unchanged ~11h), so we relabel `thoughts_base`
from it EARLY via `scripts/headstart_base_aprm.sh` (base-EM GPU lane, AFTER the head-start
SFT queue, sequential): relabel(best) -> export `data/sft_corpus/.../base` -> SFT
`thoughts_base` {hide, full}. Same run_tags/paths as the pipeline + finishes before the
post-EM relabel, so the pipeline SKIPS it (no double-work/conflict). Only `thoughts_base`
(best) is head-started; `base_last` + `policy`/`policy_last` wait for EM (policy's
`step_best` is still moving, so an early policy relabel would use a weaker checkpoint).

### Stage 1.5 — relabel (best-of-G, TOP-1) + export corpus
For each scorer × {best, last}: `train.sh --no_train --resume_from <ckpt>
--advantage_mode best --group_size 4 --length_penalty 0.15 --save_generations`, then
`export_sft_corpus.py` commits **`thoughts[best] + "\n\n" + action`** (argmax-reward
top-1 — NOT em/top-half/grpo; matches retail/airline). →
`data/sft_corpus/snorkel_finance_split/{policy,base,policy_last,base_last}`.

### Stage 2 — SFT (fresh Qwen3-4B each), 12 runs = 6 variants × {hide-obs, full-ctx}
Plain weighted-CE on the target span (`act_prm_actions_only` generator + `SFTTrainer`),
60 batches, early-stop `--best_metric eval_action_ppl`, `--eval_every 5`.
- **actions_only** (expert action) · **expert_thoughts** (oracle expert reasoning+action,
  offline pool `data/snorkel_finance_split_expert_thoughts`) · **thoughts_policy** and
  **thoughts_base** from the best corpora · **thoughts_policy/base** from the `_last`
  corpora.
- **hide-obs** (`--hide_observations`: system + first user + last obs + all model msgs)
  vs **full-ctx** (`SFT_FULLCTX=1`, keep all obs) — same corpus, hiding at tokenization.

### Stage 3 — RL from SFT (design in `cc-finance-2.0-rl-design.md`)
Finance HAS an interactive gym (`environments/snorkel_finance/`, data at
`src/act_prm/data/snorkel_finance`), but the LLM-judge grader
(`graders/snorkel_finance.SnorkelFinanceGrader`) is **not yet present** — see 2.0.

## The key metric — action sub-span (methodological core)
`eval_action_*` scores the WHOLE (thought+action) span, so verbose thoughts inflate it
and make thought-variants look worse. The trustworthy metric is the **action
sub-span** `eval_actiononly_{ppl,accuracy}` — scored on ONLY the `<tool_call>…` /
`Final Answer:` tokens, isolated by `action_start_token` (a decoded-suffix walk in the
already-tokenized ids — no re-tokenize, no boundary/template misalignment). We made
train ⟷ eval **symmetric**: `compute_loss` now also logs `train/actiononly_{ppl,
accuracy}` (via an `action_mask` ⊆ `label_mask` threaded through `prepare_minibatch` +
the collator), and eval adds `eval_actiononly_loss`. Definitions identical across
train/eval (ppl = exp(mean CE); accuracy = top-1 argmax==gold).

## Bugs fixed on this box (all committed; see git log)
1. **Multi-parquet pool corruption** — the dataset ships 3 parquet configs; globbing
   all let higher-return `aprm_sft_*` rollouts override the expert trace for 19/141
   tasks. Fixed `prebuild_pools_offline.py` to use the canonical `train-*.parquet`.
2. **Pipeline silently skipped Stage-1 EM** — `em(){ local s=$1 … tag="…${s}" }` hit
   `set -u` (same-line `${s}` expanded before assignment) → the backgrounded EM subshell
   died instantly. Split the `local`. Without this, Stage 2 trained on empty corpora.
3. **Action-subspan metric was misaligned** — old code took `tgt[-n_act:]` with a
   standalone-tokenized `n_act`; since `state_action_tokens` ends with `<|im_end|>\n`,
   this dropped the leading `<tool_call>` tokens and scored the template close instead.
   Replaced with the `action_start_token` boundary (verified on tool_call / Final Answer
   / actions_only).

## FINAL results (4B, all 12 variants; pipeline complete 07-29)
Best (min) eval **action-subspan** ppl (↓) / accuracy (↑). All 4 corpora strict-verified
clean (0 sample_id/action mismatches).

| variant | regime | action-ppl ↓ | action-acc ↑ | whole-span ppl |
|---|---|---|---|---|
| **thoughts_policy (Act-PRM)** | full | **2.040** | **0.885** | 2.68 |
| thoughts_base_last | full | 2.043 | 0.885 | 2.71 |
| thoughts_policy_last | full | 2.063 | 0.884 | 2.72 |
| thoughts_base | full | 2.064 | 0.884 | 2.71 |
| expert_thoughts (oracle) | full | 2.222 | 0.877 | 2.55 |
| actions_only | full | 2.441 | 0.871 | 2.44 |
| **thoughts_policy (Act-PRM)** | hide | **2.307** | **0.853** | 3.10 |
| thoughts_base_last | hide | 2.335 | 0.850 | 3.11 |
| thoughts_policy_last | hide | 2.342 | 0.852 | 3.11 |
| thoughts_base | hide | 2.366 | 0.849 | 3.12 |
| expert_thoughts (oracle) | hide | 2.652 | 0.833 | 3.05 |
| actions_only | hide | 2.843 | 0.824 | 2.84 |

**Headline:** in BOTH regimes, **all four Act-PRM variants (policy/base × best/last)
beat the expert-reasoning oracle AND the actions_only baseline** on next-action
prediction — Act-PRM roughly DOUBLES the improvement the expert thoughts give
(hide: actions_only 2.84 → expert 2.65 → Act-PRM ~2.31). `thoughts_policy` (policy-
scored) is best in both regimes; policy≈base, best≈last (tight cluster).

**The action-subspan metric is essential (the flip):** the Act-PRM variants have the
*worst* whole-span ppl (~2.7 full / ~3.1 hide) yet the *best* action-subspan — i.e. the
whole-span metric would rank Act-PRM LAST while it actually predicts actions BEST. The
verbose thought inflates whole-span but helps the action.

## Deliverables
- notes: `cc-finance-1.0-pipeline.md` (this), `cc-finance-2.0-rl-design.md` (Stage 3).
- analysis lib: `notebooks/cc_finance_lib.py` (load logs, plot curves; headless-safe).
- notebooks: `cc-finance-1.1-stage1-em.ipynb`, `cc-finance-1.3-stage2-sft.ipynb`
  (whole-span vs action-subspan train/eval curves per variant×regime; figs in
  `notebooks/figs_finance/`).

## Ops
- Full pipeline (still running): `MODEL=hf_qwen3_4b_instruct nohup
  ./scripts/run_actprm_pipeline.sh act_prm/snorkel_finance_split &` (watch
  `/tmp/aprm/snorkel_finance_split_4b/orchestrator.log`). Resumable.
- No-corpus head-start SFTs (done during EM): `scripts/headstart_sft_seq.sh`.
- Backups: `scripts/backup_results.sh` (dotsynced ~/.claude + laptop-pullable tarball);
  `scripts/periodic_backup.sh` (30-min loop). Kill: `pkill -9 -f '[m]ain_pytorch.py'`.
