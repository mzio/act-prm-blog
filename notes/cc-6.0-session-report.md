# cc-6.0 — Session report: the Stage-2 LR bug, what it invalidated, and the corrected pipeline

Companion to [cc-5.0](cc-5.0-sft-lr-investigation.md), which holds the running numbers.
This note is the narrative: what was done, what it found, what broke, and how each break
was diagnosed and fixed.

---

## 1. The headline

**Every Stage-2 SFT run we had ever reported was trained at a learning rate ~25–75× too
small and barely moved the model.** Measured on the weights, not inferred from curves:
`lora_A` never left its seeded init (two runs on *different corpora* agreed to 1.5e-6),
`lora_B` was barely off its zero-init, and the largest weight delta anywhere in the adapter
was 3.7e-5 against base weights of order 1e-2.

The optimizer was always healthy — adapter movement scales *exactly* linearly with LR
(25× LR → 25× movement, three significant figures). The LR was simply far too small, and
`configs/trainer/{sft,pg}.yaml` had carried `4e-5` untouched since the initial commit.

This invalidated the Stage-2 tables **and** explained the Stage-3 RL null result: all four
retail RLVR arms "warm-started" from adapters numerically indistinguishable from base, so
they *were* the same model. The 5/20, 5/20, 6/20, 5/20 spread was not noise swamping an
effect; there was nothing to detect.

**The tell we already had.** First-vs-last on the old runs would have exposed it
immediately: across all 18 hide-regime arms, held-out PPL moved between −0.12% and +0.14%
and got *worse* in 5 of 18; accuracy moved at most 0.08pp. `analyze_sft.py` reported *best*
PPL, and the minimum of a noisy flat line always looks like a result.

---

## 2. What the corrected runs show

LR chosen by probing a **thought** arm, which mattered — `actions_only` alone would have
picked 1e-3:

| lr | thoughts_base, b29 | verdict |
|---|---|---|
| 1e-4 | −0.05% | dead |
| 1e-3 | +1.19% | trains, slowly |
| **3e-3** | **+6.94%** | ~6× faster, monotonic, stable |

At 3e-3 / 150 batches all 12 hide arms converged. Teacher-forced action-subspan:

| domain | actions_only | thoughts_base | thoughts_policy | expert_thoughts |
|---|---|---|---|---|
| retail | 2.3422 / .7726 | 2.2003 / .7890 | 2.2059 / .7925 | **2.1071 / .7973** |
| airline | 2.4506 / .7629 | 2.3981 / .7644 | 2.3868 / .7691 | **2.3383 / .7732** |
| finance* | 1.8930 / .8398 | 1.8167 / .8565 | **1.7969 / .8596** | 1.8589 / .8455 |

\* finance numbers are **withdrawn** — see §4.

Act-PRM recovers **58% (retail) / 57% (airline)** of the oracle PPL gap — a strikingly
stable fraction across domains whose absolute gaps differ 2×. But the gaps themselves
shrank: the oracle's advantage over a *properly trained* baseline is 10.0% (retail) and
4.6% (airline), versus 24% and 19% under the broken runs. Most of the original advantage
was the untrained baseline being bad.

### The rollout eval inverts the ranking

Teacher-forced PPL measures prediction. Letting each policy *act* in the tau2 gym on
never-in-logs tasks measures behaviour, and the two disagree:

| arm | retail (42 tasks) | airline (18) | PPL rank |
|---|---|---|---|
| actions_only | 11.9% | 50.0% | 4th |
| expert_thoughts | 11.9% (+0.0) | **33.3% (−16.7pp)** | **1st** |
| thoughts_base | 16.7% (+4.8) | 44.4% (−5.6) | 2nd |
| **thoughts_policy** | **21.4% (+9.5pp)** | **72.2% (+22.2pp)** | 3rd |

`thoughts_policy` is 1.8× baseline on retail and 1.44× on airline. `expert_thoughts` has
the **best perplexity in every domain** and delivers zero benefit on retail and a 16.7pp
*penalty* on airline.

**Why the oracle fails.** Its training targets are ~50% bare `<tool_call>` with no
reasoning (retail 318/635; airline 46%; finance 71%) — GPT-5-mini only reasons on about
half its logged actions. So it was taught "usually don't think", and at rollout it reasons
before 0–8% of its tool calls while burning up to 2× the baseline's calls. It gets the
benefit of thought-conditioning at eval without ever learning to produce thoughts. It is
not an oracle; it is a mislabeled arm.

**Statistical honesty.** Paired McNemar on the same tasks: retail p = 0.29, airline
p = 0.29, pooled **p = 0.077**. Not significant. Act-PRM gains 6 and loses 2 tasks in each
domain — real churn, not a clean superset. The cross-domain agreement is what carries this,
not either domain alone.

---

## 3. Challenges, and what fixed them

**Reported numbers from a measurement bug.** I claimed `thoughts_policy` "emits 1.8× the
tokens, consistent with thought generation". That figure counted *talk-only* turns
(customer-facing prose) as thoughts. Corrected: measuring only reasoning that precedes a
tool call, no arm reasons often — retail 21% vs baseline 7%, airline 7% vs 2%. The
defensible claim is that thought-training yields a more *efficient* policy (25–29% fewer
tool calls at higher completion), not a visibly reasoning one.

**A volume confound I failed to check before reporting.** The retail thought corpora carry
1043 supervised steps against the baseline's 635 — the relabel emitted up to two
generations per task and the export kept both. Airline was already matched (221 vs 221)
*and* is where the effect was largest, so the confound cannot explain the headline, but it
threatens the retail half. Controls now run at matched volume (§5).

**Duplicate runs corrupting finished results, twice.** `run_one` checked its `.done` marker
*before* `wait_gpu_free`, so a driver queued behind another could wait hours and then
relaunch an arm that had completed meanwhile, appending a second run's metrics into the
finished log dir (visible as a batch sequence resetting to 0). Fixed by re-checking `.done`
after the wait. A second instance came from the guard's driver-detection regex omitting
`run_matched_control`, so cron could not see a control already in flight; the pattern now
lists every driver. Both times the corrupted rows were truncated and the curve verified
against its pre-corruption values.

**Resume that silently skipped unfinished work.** Completion was keyed on `step_best`,
which is written at the *first* eval — so any interrupted arm looked finished forever.
Now an explicit `.done` touched only on a zero-exit run, plus a back-fill pass for arms
whose metrics reached the final batch.

**A user-sim outage banked as a result.** The full-context rollout pass returned 0/42 and
0/18 for all eight arms. Not a finding: the Claude Agent SDK was down ~01:35–02:09 and every
episode died on its first turn. `check_rollout_valid.py` now gates every rollout on
generate-calls-per-task (healthy 9.6–15.3, outage exactly 1.0). My first version of that
check used `timesteps`, which is 1.00 on healthy runs too and would have rejected them.

**Orchestration that died with my shell.** Backgrounded drivers were repeatedly killed with
their process group, stalling the queue. Moved to cron (`sweep_guard.sh` every 5 min,
`snapshot_results.sh` every 20), which survives disconnection and reboot.

**Everything nearly lost to a dead box.** Airline and finance work existed only on
devservers I cannot reach. Recovered from dotsync git bundles and artifact tarballs in
`~/.claude/act-prm-backups/`, including two branches this box never had. Adapters now
publish to the HF Hub for all three domains every 15 min (github is blocked from here; HF
is not).

---

## 4. The finance data leak

Finance deserves its own section because its numbers are **withdrawn**.

The aprm finance split partitioned `unique_data_sample_id`s, but ~2.5 uids share each
`finqa_reasoning` question, so questions leaked across splits:

| split | questions | leaked from train |
|---|---|---|
| act_prm_eval | 21 | **16 (76%)** |
| rl_eval | 22 | **20 (91%)** |

Every finance Stage-2 curve was scored largely on memorised questions, and the finance RL
hold-out was never held out. Retail and airline are clean (zero train/eval overlap).

A second discovery: the 29 questions with "no usable trajectory" are not merely unused.
They have 123 trajectories and **all are reward=0** — the split's usable filter is
`done + reward>0`, so those are exactly the questions GPT-5-mini *failed*. My initial
proposal to use them as "the cleanest rollout hold-out" was wrong: it is the hardest tail
of the benchmark, selected for difficulty, not comparable to retail's 42 or airline's 18.

**Fix (v3).** Matching pool question text to `finqa_reasoning.csv` resolves 100% of
trajectories, which settles the `uid → CSV row` TODO the finance box left open. The 50
demo-bearing questions are partitioned 40 train / 10 eval **by question**; each arm's pool
is subselected by question. No relabel, no regeneration. Verified zero overlap on all four
arms. Eval is 25 traj / 363 steps — better than v1's contaminated 25 / 340 and far better
than the 7 / 127 an earlier naive filter gave.

Finance rollout runs on **two labelled sets, never pooled**: the 10 fair eval questions
(expert solved them; same questions the SFT curves score) and the 29 expert-failure
questions as an explicit hard test.

---

## 5. What is queued (cron-driven, unattended)

1. **volume-matched control** — retail `thoughts_policy` on one generation per task
   (48 traj / 627 steps vs baseline 635), then rollout. Isolates thoughts from data volume.
2. **expert_thoughts_all** — expert reasoning trained only on turns that *have* reasoning
   (`--require_thought` filters targets, keeping every turn in context). Tests whether the
   oracle recovers once trained to produce thoughts. Caveat: halves supervised steps
   (retail 635→317, airline 221→119, finance 1370→402).
3. **finance v3** — all five arms on the corrected split, full curves.
4. **finance rollout** — fair 10 and hard 29, separately.
5. **best-generation Act-PRM** — one generation per task chosen by mean EM likelihood of
   committed thoughts, volume-matched in every domain (retail 627 vs 635, airline 221 vs
   221, finance 1347 vs 1347), then rollout per environment.
6. **full-context** rollout pass and SFT arms.

---

## 6. What I would not yet claim

- **Significance.** Pooled McNemar p = 0.077. Suggestive, not established. The cheapest
  fix is ≥3 rollouts per task (~6h) — that, not more arms, is what would make the headline
  publishable.
- **Mechanism.** "Act-PRM policies reason before acting" is not supported: at most 21% of
  tool calls carry any preceding reasoning. "Thought-training yields a more efficient
  policy" is supported.
- **Finance anything**, until v3 lands.
- **`thoughts_base`.** Unreliable: +4.8pp retail, −5.6pp airline. Only the policy-scored
  variant is consistently positive, so the claim should name it specifically.
