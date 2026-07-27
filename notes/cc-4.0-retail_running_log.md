# cc-4.0 — retail running log (autonomous chain + sanity checks)

Chronological log of the autonomous retail pipeline: what ran, what broke, findings.
Sanity-check discipline: verify corpora non-empty, metrics sane, no partial checkpoints
leak into the sweep's skip logic.

## 2026-07-26

### Stage 1 EM (done)
- Both scorers trained; `step_best` **peaked early (batch 5/24)** then held-out `p(x|s,z)`
  *declined* (policy 0.429@5 → 0.394@20; base 0.425@5 → 0.391@20). So the EM reward
  overfits/drifts after ~batch 5. Both `step_best` and `step_last` are saved → we relabel
  from **both** (best = peak reward; last = fully trained) as separate SFT arms.

### BUG (caught by sanity check) — `--no_train` relabel produced train:0 corpora
- The exported `data/sft_corpus/tau2_retail/{policy,base}` had **train: 0, eval: 8** — every
  relabeled trajectory went to eval, so the `thoughts_*` SFT arms would train on nothing.
- Root cause: `rl.py` had `if no_train: continue` **before** the train-set rollout
  generation, so `--no_train` only ever saved the eval pass.
- **Fix:** moved the skip to *after* train rollouts are generated+saved (generations.jsonl)
  and *before* the optimizer step. `run_relabel.sh` regenerates best+last with a **non-empty
  guard**; `run_sft_sweep.sh` now requires a non-empty corpus (won't SFT on train:0).
- Recovery: killed the chain before it could rebuild action-only pools at the corpus path
  (silent mislabel risk), deleted empty corpora + the partial `expert_thoughts` run (kept
  the valid `actions_only`), relaunched the corrected tail.

### Early SFT signal (PRELIMINARY — confounded, do not over-read)
- hide-obs: `actions_only` eval_action_ppl **3.84** (acc .76) vs `expert_thoughts` **4.78**
  (acc .68). BUT `eval_action_ppl` teacher-forces the WHOLE target span, so thought variants
  include (harder) thought tokens → **not same-span** vs actions_only. Need an **action-
  subspan PPL** before concluding thoughts don't help (task #23; airline Claude adding it).
- Eval IS run under the hide-obs context (eval_env == train env). ✓

### Open decisions
- **Qwen3-8B thought generator** (cached: `hf_qwen3_8b` = `Qwen/Qwen3-8B`): if, after the
  corrected corpora + fair action-subspan metric, 4B thoughts still don't lift over
  actions_only, run a **Stage-1 EM with 8B → relabel → SFT (4B policy on 8B thoughts)** arm
  to test whether a stronger thought generator helps. Deferred until the fair 4B read is in.
- `rl_eval`→tau2-index map (Stage-3 eval on the true hold-out) — task #24.

## Current chain (corrected)
`run_retail_tail.sh`: `run_relabel.sh` (best+last, fixed) → `run_sft_sweep.sh` (6 arms ×
{hide,full}, non-empty guarded, skips valid `actions_only` hide) → `run_retail_stage3.sh`
(analysis → RL smoke gate → RL matrix). Logs under `/tmp/aprm/{tail,relabel_tau2_retail,
sft_sweep_tau2_retail,stage3}`.

## KEY RESULT (2026-07-27): action-only metric flips the variant ranking
The fair `eval_actiononly_ppl` (action tokens only, live per-eval) REVERSES the whole-span read.
Preliminary 4B, hide-obs, best-step (4/12 re-runs done):
| variant (hide) | action-only PPL | action-acc | whole-span PPL |
|---|---|---|---|
| actions_only | 3.84 | 0.761 | 3.84 |
| **expert_thoughts** | **2.92** | **0.789** | 4.78 |
| thoughts_policy | 3.18 | 0.782 | 3.68 |
| thoughts_base | 3.17 | 0.782 | 3.71 |
- expert_thoughts = BEST on action-only (was WORST on whole-span — the reasoning-token confound).
- **Act-PRM inferred thoughts (policy/base) beat actions_only** (3.17-3.18 vs 3.84 PPL; .782 vs .761 acc) → thoughts help next-action prediction; inferred recovers much of the expert lift.
- Pending: full-context regime + best/last corpora + 8B (re-run in progress, 4/12) to confirm.
