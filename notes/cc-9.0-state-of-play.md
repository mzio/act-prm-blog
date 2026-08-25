# cc-9.0 — State of play (08-25). What is true, what is retracted, what is running.

Supersedes the status parts of [cc-7.0](cc-7.0-handoff-state.md). Read this first.
Companion notes: [cc-8.0](cc-8.0-rollout-confound.md) (the open rollout problem),
[cc-6.0](cc-6.0-session-report.md) (narrative, rollout sections retracted),
[cc-5.0](cc-5.0-sft-lr-investigation.md) (running numbers, rollout sections retracted).

---

## 1. What is solid

**Stage-2 teacher-forced action-span metrics, all domains.** Computed offline from a fixed
checkpoint over a fixed eval set: no sampling, no user simulator, no judge. Reproducible.

| domain | actions_only | expert_thoughts | thoughts_policy | thoughts_base |
|---|---|---|---|---|
| retail | 2.3422 / .7726 | **2.1071** / .7973 | 2.2059 / .7925 | 2.2003 / .7890 |
| airline | 2.4506 / .7629 | **2.3383** / .7732 | 2.3868 / .7691 | 2.3981 / .7644 |
| finance v3 (clean split) | 1.9282 / .8407 | 1.9031 / .8436 | **1.7919** / .8625 | **1.7954** / .8626 |

Act-PRM's inferred thoughts improve next-action prediction in every domain. On finance's
corrected question-level split they beat the expert arm by ~5.8% PPL; on tau2 the expert arm
is ahead on PPL. Insurance (4th domain) is in flight.

**Supporting results that also hold** (all offline / large-sample, not rollout-based):
- The LR bug: everything before 08-22 trained at 4e-5, ~25-75x too small; adapters were
  numerically near-no-ops. Fixed at 3e-3 (probed: 1e-4 dead, 1e-3 +1.19%, 3e-3 +6.94%).
- The finance leak: v1 split partitioned uids, but ~2.5 uids share a question, so 76% of
  eval and 91% of the RL hold-out leaked from train. v3 splits by QUESTION (40/10).
- Mechanism: the expert reasons on only ~50% of retail turns (46% airline, 31% finance,
  64% insurance), so `expert_thoughts` is trained on a large fraction of bare `<tool_call>`
  targets. Measured over thousands of turns, not 18-42 binary outcomes.

## 2. What is RETRACTED

**Every task-completion (rollout) number.** Not just weakened — currently uninterpretable.
Full evidence in cc-8.0. The three findings that force this:

1. **Seeds do not pin outcomes.** Re-running airline `thoughts_policy` at the *same* seed 42
   gave 55.6% against the original 72.2%. Same adapter (sha `9b64267354d38feca3f29ad9`,
   verified byte-identical), same tasks, same config.
2. **The harness drifts over hours.** `actions_only` scored 53.7% at 09:39 and 33-39% at
   17:50-20:04 the SAME day.
3. **Arm is confounded with time** in every comparison, because both drivers loop arms outer
   and seeds/domains inner, so one arm is always measured before the other.

Consequently: airline's +22.2pp headline was a fluke; the pooled McNemar p=0.077 was a
one-sample-per-task artifact; and at 3 rollouts/task the pooled difference is +0.6pp
(p=0.765). Two opposite conclusions were drawn from this data on 08-24 and BOTH were
contaminated. The direction of the true effect is unknown.

**The base-model control is also affected.** It suggested BC is worse than no SFT
(base 26.7% pooled vs actions_only 23.3%) and `expert_thoughts` worst (18.3%). Those were
single draws at seed 42 and inherit the same problem.

**The fix** (not yet run, ~5.5h): interleave arms within each seed so both sample the same
conditions, plus a fixed canary configuration re-run each session to measure drift directly.

## 3. Insurance (4th domain) — in flight

Data: `mzio/aprm-insurance-gpt5m_med-gs4-s0-r1-train`, 13,131 steps / 1,000 trajectories /
262 questions. Kept the 910 SUCCESSFUL trajectories (91%), covering 261/262 questions.

Design decisions (steered by MZ):
- **QUESTION-level split, three-way disjoint**: 180 train / 40 eval / 41 rollout. Verified
  zero overlap; rollout questions never seen by Stage 1 or 2. (The finance-leak lesson.)
- **One demonstration per question** -> 2,273 train / 540 eval targets, IDENTICAL across all
  four arms. Retail kept two generations per task, which forced a separate volume-matched
  control; this removes the confound by construction.
- **Trajectory filter** = expert solved it AND >=1 reasoning-bearing step. The second
  condition is inert here (220/220 qualify) but would matter on finance (56% fail it).
- **No `expert_thoughts_all` arm** — with every trajectory carrying reasoning, it reduces to
  the same setup.
- **`EM_NB=25` kept** (the other domains' value). NOTE: with 180 train trajectories that is
  25x4 = 100 samples = **0.56 epochs**, vs retail 2.0x and airline 4.8x. Insurance therefore
  has the least-trained EM checkpoint of any domain. If its Act-PRM arms underperform,
  under-training is the first hypothesis, and the cheap test is a warm-start continuation
  (resume from step_best for +20 batches, ~3h/scorer) rather than a full re-run (~30h).

### 3a. Insurance — challenges, what I proposed, and what actually worked

Format: what broke -> hypotheses proposed -> ruled out (with evidence) -> what fixed it.
The failed hypotheses are kept deliberately; they are what makes the surviving explanation
credible, and without them a later reader re-proposes them.

**(i) EM OOMed on the M-step.** 817 MiB free of 95 GiB, allocating 1.11 GiB.
- Proposed: (a) `--gradient_checkpointing`; (b) `batch_size` 4->2; (c) `--obs_max_chars`;
  (d) `group_size` 4->2.
- Diagnosis first: measured the pool. Insurance tool calls return whole SQL tables —
  observations p50=265 but p99=48,995 and max=56,543 chars; trajectories to 99,177 chars
  (~25k tokens). The other domains are nowhere near this.
- **What worked: (c) `obs_max_chars: 4000`**, chosen from the distribution — keeps 93% of
  observations untouched, cuts worst-case context 3.8x to ~6.6k tokens. Set in the ENV
  config, not per-arm, so every arm truncates identically and the comparison stays fair.
- Also applied (a) as belt-and-braces on the pass that actually failed. Did NOT use (b) or
  (d): both would have changed the sample/coverage accounting (b would have forced
  RELABEL_NB from 45 to 90), and neither addressed the root cause.
- Result: 0 OOMs in 21+ batches since.

**(ii) 49 minutes burned before a single training step.** batch-0 eval over the 40 eval
trajectories cost `time/run_evals_eval: 2932s`.
- Proposed: (a) `--no_initial_eval`; (b) fewer eval trajectories; (c) rarer evals.
- **What worked: (a) + (c)** — `--no_initial_eval --eval_every EM_NB`, i.e. exactly one eval
  at the end, which is all that is needed to write `step_best` for Stage 1b to resume from.
- Rejected (b): shrinking the eval set would have made the Stage-2 curves less comparable to
  the other domains for no compute saving in Stage 2 itself.

**(ii-b) LIMITATION accepted: no Stage-1 eval CURVE.** `--eval_every EM_NB` means exactly
one eval point, at the final batch. I made that trade to kill the 49-min batch-0 eval
without flagging that it also destroys the eval curve — that was the wrong call to make
silently.
- The assumption behind it was also wrong: with `obs_max_chars: 4000` the eval STILL cost
  2,893s vs 2,932s uncapped, so eval time is dominated by generating over the 540 eval
  steps, not by context length. Each additional eval point costs ~48 min.
- **What we have:** the TRAIN curve is complete — 25 rows, one per batch, with
  `train/try_0/{final_reward, likelihood, penalized, thought_tokens}`. Reward-over-steps is
  plottable for Stage 1 training. The eval side is a single point (batch 24):
  likelihood 0.3377 (sd 0.26, max 0.9945), penalized 0.2138, thought_tokens 79.3.
- **Decision (MZ, 08-25):** leave the BASE scorer at `eval_every 25` too, so the two scorers
  stay comparable, and record this as a known limitation rather than giving one arm a denser
  curve than the other.
- **If a Stage-1 eval curve is wanted later:** add `--save_every 5` to the EM run so
  intermediate checkpoints exist, then score them offline against the eval pool. Re-running
  with a denser `eval_every` is the expensive way (+3.2h for 5 points) and would break
  comparability with the policy run already finished.
- Note `eval/try_0/correct` and `accuracy` are 0 in these EM logs and that is EXPECTED:
  "correct" means exact action-string match in the Act-PRM generator. The meaningful
  quantities are `likelihood` and `penalized`. Do not read those zeros as a failure.

**(iii) The driver marked a crashed arm as complete and moved on.** After the OOM it logged
`FAILED` and immediately started the base scorer, which would have repeated both mistakes.
- **What worked:** failure counting + conditional ALLDONE in every driver, plus a guard-side
  retraction that removes ALLDONE when a per-arm marker is missing. This pattern has now
  caught three separate stages (finance_v3, finance_rollout, insurance).

**(iv) Design errors caught BEFORE running** (cheap, worth recording):
- I first wrote Stage 1 as a single `--no_train` pass from the base model. That would have
  inferred thoughts from an untrained policy, so `thoughts_policy` would not have been
  policy-scored at all. Caught by comparing against `run_relabel.sh`; corrected to the
  two-pass EM-then-relabel structure the other domains use.
- I passed `--advantage_mode best` to `export_sft_corpus.py`; it is a main_pytorch flag and
  the export would have crashed. Caught by enumerating every flag against the argparse
  definitions rather than assuming.
- I proposed step-level target filtering (only reasoning-bearing turns as targets) to
  equalise the arms, which needed new plumbing because the actions_only pool has reasoning
  stripped and a content-based filter would have left it with ZERO targets. MZ pointed out
  the simpler trajectory-level filter; measurement then showed it is inert here (220/220
  trajectories already qualify), so the right answer was to do nothing.

Measured: EM 9.0 min/batch. uid -> gym `company_task_id` mapping is 261/261, 0 ambiguous,
sets disjoint (`data/splits/snorkel_insurance_uid_to_task.json`).

## 3b. FINDING (08-25): Stage-1 EM has been running at the BROKEN learning rate, in every domain

**The M-step is a numerical no-op and always has been.** The 08-22 LR fix was applied to
Stage 2 (`--learning_rate 3e-3`) and never to Stage 1. `configs/trainer/pg.yaml` still has
`learning_rate: 4e-5`, and neither `train.sh` nor `run_relabel.sh` nor the Stage-1a call in
`run_insurance_pipeline.sh` overrides it. Confirmed on the run configs: retail, finance and
insurance EM runs all record `lr=4e-05`.

**Measured on the weights, not inferred from curves** (insurance policy EM, 25 batches):

| measurement | value |
|---|---|
| max abs lora_B | 5.9e-05 |
| mean abs lora_B | 1.1e-07 |
| max abs B@A (the whole adapter contribution) | **2.8e-06** |
| base weight scale | ~1e-2 |

`lora_B` is zero-initialised, so `B@A` IS the adapter. At 2.8e-06 against 1e-2 the "trained"
EM policy is the base model to ~0.03% at its most-updated position.

**Confirmed across domains (08-25), measured on EXISTING checkpoints — no GPU, no re-run:**

| domain | Stage-1 EM adapter, max abs B@A | verdict |
|---|---|---|
| retail (policy, step_best) | 6.7e-07 | NO-OP |
| retail (base, step_best) | 6.7e-07 | NO-OP |
| retail (step_last variants) | 3.1e-06 | NO-OP |
| finance | 4.1e-07 | NO-OP |
| insurance | 2.8e-06 | NO-OP |
| airline | no EM checkpoint survived the box recovery | unverifiable |

Base weights are order 1e-2, so every one of these is 4-5 orders of magnitude too small.
The flat EM reward curve in every domain has a single measured cause. Worth noting this
check costs nothing: `scripts/report_lora_movement.py` reads the saved adapters directly,
so "did training do anything" never needs a re-run to answer.

**How I got here — three readings, two wrong:**
1. "Insurance's flat EM curve = under-training at 0.56 epochs." WRONG. Refuted by the
   cross-domain table: finance ran 2.00 epochs (58 batches) and its policy curve moved
   **-0.0186**, worse than insurance's +0.0123 at 0.56 epochs.
2. "Flat EM curves are just normal for this method." WRONG, and worse — it rationalised the
   symptom instead of diagnosing it. Every domain being flat should have prompted "what is
   common to every domain?", not "flatness is inherent".
3. **CORRECT, after MZ asked what the learning rate was:** the LR is 4e-5 everywhere in
   Stage 1, the same value proven in cc-5.0/6.0 to leave adapters indistinguishable from
   base. Every symptom follows: flat reward in all domains, `train/loss = -0.0000` batches,
   |delta|/sd < 1 universally, and 2 epochs helping no more than 0.56.

**What this does and does not invalidate:**
- Stage-2 teacher-forced results STAND as measured. Act-PRM corpora do beat baseline (and
  beat expert thoughts on finance). That benefit is real.
- But it comes ENTIRELY from the **E-step** — sample `group_size` thoughts per logged action,
  keep the best by length-penalised likelihood — with zero contribution from EM policy
  improvement. "Act-PRM works" is currently a claim about best-of-4 rejection sampling.
- It explains an oddity noted earlier and left unexplained: `thoughts_policy` and
  `thoughts_base` track each other closely everywhere (retail 2.2059 vs 2.2003, finance
  1.7919 vs 1.7954). If the proposing model is the base model in both, they differ only in
  which model SCORES the candidates — a much smaller difference than intended.

**Decision (autonomous, 08-25 12:30):** let the insurance base-scorer EM finish at 4e-5.
Rationale: parity. Insurance's value is as a fourth COMPARABLE domain, and retail/airline/
finance all ran Stage 1 at 4e-5; switching insurance mid-pipeline would make it the only
domain trained differently, which costs more than the ~3.5h it would save. Flagging rather
than unilaterally re-running everything.

**Recommended next experiment (NOT auto-launched — needs a decision):** re-run Stage 1 for
one domain at `--learning_rate 3e-3`, relabel, and compare the resulting Stage-2 numbers
against the 4e-5 corpus. That is the first real test of whether the M-step adds anything
over best-of-4 sampling. ~9h for EM + relabel, plus ~2.5h for one Stage-2 arm. If it makes
no difference, the honest framing of the method changes (and gets cheaper); if it does, all
four domains' Act-PRM numbers are a lower bound.

## 4. Queue (cron-driven, `sweep_guard.sh` every 5 min)

Gate order, top-down; earlier gates are satisfied and no-op:

| gate | stage | est. |
|---|---|---|
| 2f2 | **insurance** — Stage 1a/1b x2 scorers -> Stage 2 x4 arms -> Stage 3 rollout on 41 held-out | ~28h |
| 2g2 | multi-rollout, remaining `expert_thoughts` x3 (retail+airline) | ~3.5h |
| 2h | best-generation Act-PRM | ~25h |
| 2z | full-context rollout pass | last |

Not queued, but higher value than 2h/2z: **the interleaved rollout fix** (cc-8.0 §5).

## 5. Standing hazards (all have bitten at least once)

- **cron has no proxy.** Any driver calling `main_pytorch.py` directly must export
  `https_proxy`/`http_proxy`/`HF_TOKEN` itself. This killed all 10 finance rollouts on HF DNS.
- **`update_configs` silently drops a CLI flag whose key is not already in the yaml.** This
  made all 10 finance rollouts score the wrong 11 questions. Declare keys as `null`.
- **Drivers touching ALLDONE unconditionally** record wholesale failure as success. All
  drivers now count failures; the guard retracts ALLDONE when a per-arm marker is missing.
- **Re-running the same run tag appends into the finished log dir** (the dir name is
  tag + config hash). Use a distinct tag; `scripts/truncate_restarted_metrics.py` repairs it.
- **The judge defaults to "no" on any unparsed verdict.** 67% of finance verdicts hit that
  default. `parse_verdict` now falls through several shapes;
  `scripts/check_judge_parse_rate.py` gates every insurance rollout.
