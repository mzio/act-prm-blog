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

## 3. What is NOT the cause (ruled out with evidence)

- **Checkpoint mixup** — sha256 identical across all 6 runs; runtime load lines in each
  `logs.log` confirm the path actually loaded; file mtime predates every run.
- **Batch truncation** — `max_input_id_len` zeroing path never fired (0 "Truncating" warnings).
- **Config drift** — task sets, turn cap, hide-obs regime identical; only `seed` and
  `eval_group_size` differ.
- **Positional padding artifact in the batch** — within the airline x3 batch, gen_id 0/1/2
  scored 10/7/9 with 7.5/9.2/8.9 tool calls: scatter, not a monotonic trend.
- **Batched vs unbatched measuring different things** — for `thoughts_policy` the two agree
  (x3 48.1% vs x1 mean 48.6%). They disagree for `actions_only` (53.7% vs 36.1%), but those
  were run 8 hours apart, so that is the time confound, not the protocol.

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
