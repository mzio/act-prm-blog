# cc-1.0 — Act-PRM × tau2 experiment plan & status (retail-first)

PyTorch experiments for the Act-PRM blog. Default policy: **Qwen3-4B-Instruct-2507**
(`hf_qwen3_4b_instruct`), LoRA (`r8_a16_linear`). Envs: **tau2 retail** (this box,
`devvm22863`) and **tau2 airline** (`devvm27322`). Three stages: thought-gen → SFT → env-RL.

Splits (`data/splits/tau2_*.json`) are 3-way: **rl_eval** (held out for Stage-3 only)
+ **act_prm_train** / **act_prm_eval** (used in Stages 1 & 2). Retail = 49/10/12.

## Stage 1 — Thought generation (Act-PRM EM)  ✅ built, retail running
Train a LoRA thought-generator so action-only logs get plausible thoughts. Per logged
action `x` in state `s`: sample G thoughts `z`, reward = length-penalized `p(x|s,z)`,
group-normalize → advantage, PG step on `(thought+action)` tokens.
- Code: `generator/act_prm` (E-step + reward + advantage), `environments/act_prm_traces`
  (loads expert logs, strips to action-only for the E-step), `trainer/pg` (M-step).
- Two EM scorers: **on-policy** (`--no-score_with_base`, current LoRA scores) and
  **base** (`--score_with_base`, LoRA detached → frozen Qwen3-4B scores).
- Runner: `scripts/train.sh --env_config act_prm/tau2_retail --generator_config act_prm ...`
- **Status (2026-07-25):** retail on-policy running (batch ~6/25), base-scored chained
  after it (a `kill -0 <pid>` waiter, PID 488479). Airline both scorers on devvm27322.

### Stage 1.5 — Export the SFT corpus from the BEST Stage-1 ckpt  ⬜ TODO
Per the plan, Stage-2 SFTs on a **fixed** corpus, NOT on-the-fly rollouts. After Stage 1,
run the best ckpt in generate-only mode over **train+eval** to emit one committed
`(state → thought+action)` per logged step; write it as `act_prm_traces` pools so Stage 2
reuses the existing loader.
- Mechanism: `scripts/train.sh ... --resume_from <step_best> --no_train --advantage_mode best
  --group_size G` (samples G, keeps the best-scored thought) → `generations.jsonl`, then a
  small converter `scripts/export_sft_corpus.py` (⬜) turns the committed rows into pools
  `data/sft_corpus/tau2_retail/{policy,base}/{train,eval}.json` (assistant content =
  thought+action). Do this for both scorers → the two aprm SFT datasets.
- SFT sample contract: **1 sample = (state, target)**, CE only on `target` tokens
  (`state_action_tokens[state_len:]`), matching `prepare_minibatch`.

## Stage 2 — SFT four datasets from a FRESH base  ⬜ partly built
Train four fresh Qwen3-4B LoRAs (for comparison + as Stage-3 inits):
1. **expert action-only** — target = action (the split's action-only pools; no gen needed).
2. **aprm thought-action (base-score)** — Stage-1.5 base corpus.
3. **aprm thought-action (policy-score)** — Stage-1.5 policy corpus.
4. **expert thought-action** — target = original expert reasoning+action (keep the thoughts
   we strip in Stage 1). Needs a loader flag `keep_expert_thoughts` in `act_prm_traces/data.py` (⬜).
All with `--hide_observations` (context = system + first user + last obs + all model msgs).
- Trainer: `trainer/sft` (`supervised_fine_tuning: true`, `drop_zero_advantage: true`),
  generator `act_prm_actions_only` (no sampling; target span = assistant content, adv=1).
- Runner today: `scripts/train_sft.sh <env> <variant>` — extend variants to the 4 above
  (currently actions_only/thoughts_policy/thoughts_base; add **expert_thoughts**).
- **Offline SFT metrics** (⬜, add to the SFT eval loop, every `--eval_every` updates over
  act_prm_eval): (a) **action-token perplexity** `exp(mean CE over target tokens)`,
  (b) **action accuracy** = fraction of target tokens whose argmax logit == gold. **Early
  stop / `step_best`** on lowest eval action-PPL (set `best_metric: eval_action_ppl`).

## Stage 3 — Env RL from the best Stage-2 ckpt  ✅ env built, ⬜ eval-split wiring
Warm-start each Stage-2 `step_best` LoRA and RL on the live tau2-gym env.
- Runner: `scripts/train_rl_from_sft.sh retail <sft_step_best>` (`.venv-tau2`, `--extra tau2`,
  local tau2-bench clone). Generator `hf_grpo` (mean-centered adv), trainer `pg`.
- User-sim + judge: **claude_agent_sdk** (ambient Claude Code OAuth on a devserver).
- ⬜ Map aprm `rl_eval` uids → tau2 task indices so eval is strictly on the hold-out
  (today it uses tau2's own index split). Then: train on train+eval, eval every N on rl_eval;
  pick best → final test.

## Deliverables & metrics
- Notes here as `notes/cc-{index}-{name}.md`; analysis notebooks `notebooks/cc-*.ipynb`
  (uv kernel in `./.venv`).
- **Per-step metrics** for train curves: `tinker_cookbook.utils.ml_log` already writes
  `<log_path>/metrics.jsonl` (loss, reward/`p(x|s,z)`, kl, seq-skips) per step for Stages
  1&3; Stage 2 must additionally log eval action-PPL/accuracy per eval. Reconstruct curves
  (train+eval) in `notebooks/cc-1.1-aprm_tau2_retail_results.ipynb` (⬜).

## Build checklist (retail, this box) — do in order as Stage 1 finishes
- [ ] `scripts/export_sft_corpus.py` + Stage-1.5 export runs (policy & base best ckpts).
- [ ] `keep_expert_thoughts` loader flag + `expert_thoughts` SFT variant.
- [ ] Offline SFT metrics (action-PPL + accuracy) + `best_metric: eval_action_ppl` early-stop.
- [ ] Stage-2 SFT ×4 (fresh base, hide-obs) on retail; log per-eval metrics.
- [ ] Stage-3 RL from each Stage-2 `step_best`; wire rl_eval→tau2-index mapping.
- [ ] `notebooks/cc-1.1-aprm_tau2_retail_results.ipynb` — training/eval curves + comparison.
