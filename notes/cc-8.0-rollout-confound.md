# cc-8.0 — The tau2 rollout harness has an arm/time confound (OPEN)

**Status: unresolved, accepted for now (08-24), revisit before any rollout claim is written up.**

Every task-completion number from the tau2 gym is currently uninterpretable as an
arm-vs-arm comparison. This note records what was measured, why the comparison is broken,
and what would fix it, so the finding is not rediscovered from scratch.

## 1. What the numbers do

Same checkpoint throughout (airline `thoughts_policy`, adapter sha256
`9b64267354d38feca3f29ad9`, verified byte-identical across all runs, written 08-21 02:18 and
never rewritten; only one candidate dir matches the driver's glob).

| when | arm | protocol | score |
|---|---|---|---|
| 08-21 22:14 | actions_only | x1 seed42 | 50.0% |
| 08-21 23:29 | thoughts_policy | x1 seed42 | **72.2%** |
| 08-24 09:39 | actions_only | x3 batched | **53.7%** |
| 08-24 11:07 | thoughts_policy | x3 batched | 48.1% |
| 08-24 14:30–17:13 | thoughts_policy | x1 seeds 0/1/7/42 | 50.0 / 44.4 / 44.4 / 55.6 |
| 08-24 17:50–20:04 | actions_only | x1 seeds 0/1/7/42 | 38.9 / 38.9 / 33.3 / **33.3** |

Two facts that matter:

- **Seeds do not pin the outcome.** Seed 42 re-run gave 55.6% against the original 72.2% —
  same seed, same weights, same 18 tasks. `main_pytorch.py` calls `seed_everything(args.seed)`
  (random/numpy/torch/cuda), so this is CUDA-level nondeterminism in generation, not a
  config difference. Labelling a run by its seed does not identify it.
- **The harness drifts over hours.** `actions_only` scored 53.7% at 09:39 and 33–39% at
  17:50–20:04 on the SAME DAY. A 17pp intra-day swing on the baseline arm.

## 2. Why the comparisons are broken

`run_x1_replicate.sh` loops **arms outer, seeds inner**, so all `thoughts_policy` runs
finished before any `actions_only` run began (14:30–17:13 vs 17:50–20:04). `run_multirollout.sh`
does the same (actions_only 09:39, thoughts_policy 11:07). **Arm is therefore confounded with
time in every comparison we have**, and the harness demonstrably moves ~17pp over that
timescale.

This produced two opposite wrong conclusions in one day:

1. "The effect is noise" — from comparing x3 (08-24) against x1 (08-21), across the drift.
2. "The paired delta is real, +14.4pp in 5/5 draws" — from comparing afternoon
   `thoughts_policy` against evening `actions_only`, i.e. the drift itself.

Neither is supported. The direction of the true effect is unknown.

## 3. Hypotheses proposed, and what each one turned out to be

Kept in full, including the dead ends. Each was a real candidate at the time, and the value
of the surviving explanation comes from these having been eliminated with evidence.

| # | hypothesis | test | outcome |
|---|---|---|---|
| 1 | Batch truncation zeroed rewards — `max_input_id_len` sets ALL streams' reward to 0 if ANY exceeds it, which would hit the batched x3 runs harder | grep "Truncating" in every run log | **RULED OUT** — 0 warnings in all runs |
| 2 | Different checkpoint between runs (the driver resolves via `ls -dt \| head -1`, so a newer dir would silently win) | sha256 the adapter each run loaded; check for rival dirs; check file mtime vs run times | **RULED OUT** — one candidate dir, one hash (`9b64267354d38feca3f29ad9`) across all 6 runs, file written 08-21 02:18 and never rewritten |
| 3 | Config drift (task set, turn cap, hide-obs) | diff the run configs | **RULED OUT** — identical but for `seed` and `eval_group_size` |
| 4 | Positional/padding artifact — 3 left-padded streams in one batch, longer thought-arm outputs meaning more padding | compare per-`gen_id` score and tool use within the x3 batch | **RULED OUT** — gen_id 0/1/2 scored 10/7/9 with 7.5/9.2/8.9 tool calls: scatter, no monotonic trend |
| 5 | Batched (x3) and unbatched (x1) measure different things | run x1 at fresh seeds and compare to the x3 mean | **RULED OUT** — `thoughts_policy` x1 mean 48.6% vs x3 48.1% |
| 6 | The original 72.2% was a lucky *seed* | re-run at seeds 0/1/7 | **PARTLY** — new seeds gave 50.0/44.4/44.4, so 72.2% is an outlier; but see 7 |
| 7 | Outcomes are not reproducible even at a FIXED seed | re-run seed 42 itself | **CONFIRMED** — 55.6% vs the original 72.2%, same seed, same weights |
| 8 | The harness drifts over hours/days | compare same-arm runs at different times | **CONFIRMED** — `actions_only` 53.7% at 09:39 vs 33-39% at 17:50-20:04 same day |
| 9 | Arm is confounded with measurement time in every comparison we have | inspect the drivers' loop order | **CONFIRMED** — both loop arms outer, so one arm always precedes the other |

**Two proposed tests that would have proved nothing, caught before running:**
- Re-running x1 unchanged as a "replicate". `main_pytorch.py` calls
  `seed_everything(args.seed)` with default 42, and both the original x1 and the x3 run used
  42 — so this would largely have reproduced 72.2% and been read as "batching is fine".
  MZ caught this; varying the seed is what made the runs independent draws.
- Concluding from the paired seed sweep that the delta is real (+14.4pp in 5/5 draws). That
  sweep ran all `thoughts_policy` (14:30-17:13) then all `actions_only` (17:50-20:04), so
  the "paired" delta is confounded with hypothesis 8. Caught only after computing it — I
  reported it as a real effect first, which was wrong.

## 4. Suspected cause

Not established. The leading candidate is variation in the **LLM user simulator** (a live
Claude Agent SDK service driving every episode) and/or the judge, whose behaviour can differ
between sessions and days. GPU/CUDA nondeterminism is proven present (the seed-42 re-run) but
at ~5pp sd it does not explain a 17pp intra-day move on its own.

## 5. The fix

**Interleave the arms.** Run `actions_only` and `thoughts_policy` back-to-back within each
seed rather than all of one then all of the other, so both sample the same conditions.
4 seeds x 2 arms ≈ 5.5h on airline. Until that is done, no arm-vs-arm completion number
from this harness should be quoted.

A stronger version also logs a fixed **canary** — one reference configuration re-run at the
start of every session — so drift is measured rather than inferred after the fact.

## 6. What is unaffected

Everything teacher-forced. Stage-2 action-span PPL and accuracy are computed offline from a
fixed checkpoint over a fixed eval set with no sampling, no user simulator and no judge, and
are reproducible. The three-domain Stage-2 table stands, as does the insurance pipeline's
Stage-1/Stage-2 output. Only gym task-completion numbers are affected.
