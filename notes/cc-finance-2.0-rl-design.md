# cc-finance-2.0 — Stage 3 RL from SFT (design + prerequisites)

Goal: warm-start the best Stage-2 SFT LoRA and RL it on the **interactive
snorkel_finance gym**, evaluating strictly on the held-out **rl_eval** tasks (26 uids,
never seen by act-prm). Mirrors the tau2 airline Stage-3 plan (`cc-airline-1.0` §Stage 3)
but finance has its own gym + grader.

## What exists vs. what's missing
| Piece | State |
|---|---|
| Interactive env `environments/snorkel_finance/` (reset/step, 5 tools, SQLite backend) | ✅ present |
| `scripts/train_rl_from_sft.sh <domain> <sft_ckpt>` | ✅ present (generic) |
| `configs/generator/hf_grpo.yaml` (agentic GRPO rollouts) | ✅ present |
| Gym data `src/act_prm/data/snorkel_finance/{raw,_repo}` | ✅ present (user added) |
| **`graders/snorkel_finance.SnorkelFinanceGrader`** (env imports it) | ❌ **MISSING** — env import will fail |
| Benchmark CSVs `src/act_prm/data/snorkel_finance/benchmark/{finqa,finqa_reasoning}.csv` | ❌ not copied (in `_repo/data/benchmark/`) |
| **Online-RL env config** instantiating `SnorkelFinanceEnv` (the two `act_prm/snorkel_finance*.yaml` are `act_prm_traces`, not the gym) | ❌ MISSING |
| **rl_eval uid → benchmark-row map** (strict hold-out eval) | ❌ MISSING (same TODO airline flagged) |
| LLM-judge API for `finqa_reasoning` grading (`ANTHROPIC_API_KEY` or litellm backend) | ❌ needs key |

## Proposed setup (RL loop)
- **Policy:** best Stage-2 SFT LoRA (pick by eval action-subspan ppl from cc-finance-1.3
  — likely a `thoughts_*` variant if the flip holds). Warm-start via `--resume_from <step_best>`.
- **Rollouts:** `hf_grpo` generator — env.reset → (tool_call → env.step)* → respond_user →
  end-of-episode grade; GRPO group-mean-centered advantages; `pg` trainer takes the step
  (no new trainer code).
- **Reward:** `SnorkelFinanceEnv.step` returns [-1,1] from the grader:
  - **finqa** (290 quantitative Qs): 2-decimal-truncation numeric match — deterministic,
    **no API**. Cleanest first RL target.
  - **finqa_reasoning** (79 qualitative Qs; what the aprm traces are): LLM-judge → needs
    `ANTHROPIC_API_KEY` (or the litellm `claude_agent_sdk`/`metagen` backend).
- **Eval:** task success on the 26 `rl_eval` questions only (held out from act-prm).

## Concrete build steps (each is a discrete, testable task)
1. **Copy benchmark CSVs** (local, no network):
   `cp src/act_prm/data/snorkel_finance/_repo/data/benchmark/{finqa,finqa_reasoning}.csv
   src/act_prm/data/snorkel_finance/benchmark/`. *(Done in this pass — see below.)*
2. **Build `src/act_prm/graders/snorkel_finance.py`** with `SnorkelFinanceGrader`
   (+ `graders/__init__.py`): callable `(question, correct_answer, response, …) ->
   (reward01, text)` and an async `call_async`. finqa → truncate-to-2dp compare
   (parse `\boxed{}`/numeric); finqa_reasoning → LLM judge via the litellm custom
   provider used by tau2 (`llm_handlers/litellm_*`). Match the reference grader in
   `snorkel-ai/FinQABenchmark` (`_repo/`). Unit-test on a few CSV rows offline (finqa
   path needs no key).
3. **Online-RL env config** `configs/environments/act_prm/snorkel_finance_gym.yaml`
   (name: the SnorkelFinanceEnv registry key) with `data_path`, `benchmark_csv`,
   `task: finqa` (start numeric), `max_turns`, `eval_all_samples`/split knobs.
4. **rl_eval → benchmark-row map:** the 26 `rl_eval` uids index the aprm trajectory
   dataset (`unique_data_sample_id`), NOT the benchmark CSV rows the gym serves. Build
   the map (uid → company+question → CSV row index) so RL eval runs exactly on the
   hold-out. (Airline used a `cc-airline-2.0` map; write `cc-finance-2.1` similarly.)
5. **Run:** `scripts/train_rl_from_sft.sh finance <sft_step_best>` (adapt: point at the
   gym config + generator `hf_grpo` + `pg` trainer + `--resume_from` the SFT LoRA).

## Why this is NOT auto-run unattended
- The grader must be built + validated (a wrong grader silently corrupts the RL reward).
- `finqa_reasoning` grading needs an API key (not set here); `finqa` numeric is the
  no-key first target but the aprm traces are reasoning-task, so the eval-task mapping
  matters.
- rl_eval→row mapping must be correct or the "hold-out" claim breaks.
These want an attended pass. Stage 1 (thought-gen) + Stage 2 (SFT) + the offline
action-subspan comparison (cc-finance-1.x) are the self-contained result; RL is the
follow-on once the grader + mapping land.

## Status
- Step 1 (copy CSVs) done in this pass. Steps 2–5 pending (attended). The 26 rl_eval
  uids are recorded in `data/splits/snorkel_finance.json` and echoed in cc-finance-1.0.
