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
  `step_last` (rolling, `save_every 20`). **Observation:** eval reward plateaus early
  (policy best≈step 5, base best≈step 10–15) — same early-plateau as airline → we
  relabel from BOTH `step_best` and `step_last` and compare.

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

## Preliminary results (4B, PARTIAL — head-start variants only; live)
Best (min) eval **action-subspan** ppl / accuracy so far:

| variant | regime | action-ppl ↓ | action-acc ↑ | whole-span ppl |
|---|---|---|---|---|
| actions_only | hide | 2.84 | 0.824 | 2.84 (== ; no thought) |
| actions_only | full | 2.45 | 0.871 | 2.45 |
| **expert_thoughts** | hide | **2.65** | **0.833** | 3.05 |

**The airline flip reproduces:** `expert_thoughts` (hide) BEATS `actions_only` (hide)
on action prediction (2.65 < 2.84), even though its whole-span ppl is *worse* (3.05 >
2.84) — the thought inflates whole-span but *helps* the action. This is the whole point
of the action-subspan metric. Act-PRM (thoughts_policy/base) variants pending Stage 1.5.

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
