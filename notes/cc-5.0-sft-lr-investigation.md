# cc-5.0 — Stage-2 SFT never trained: diagnosis, fix, and the re-run

Running log. The results table near the bottom is refreshed automatically every 20 min
by `scripts/snapshot_results.sh` (cron).

## The finding

Every Stage-2 SFT run we have ever reported was trained at `learning_rate: 4e-5`
(`configs/trainer/{sft,pg}.yaml`, untouched since the initial commit `88d1068`) and
**barely moved the model**. This is measured on the weights, not inferred from curves:

| stage | max &#124;(α/r)·B·A&#124; |
|---|---|
| Stage-1 EM (retail) | 1.3e-06 |
| Stage-1 relabel (retail) | 1.3e-06 |
| Stage-2 SFT (retail) | 4.5e-05 |
| Stage-2 SFT (finance) | 2.4e-06 |
| Stage-3 RLVR (retail) | 2.4e-05 |

Base weights are order 1e-2, so the largest single-layer perturbation was ~0.4% and the
median ~0.01%. `lora_A` was still at its seeded init — two SFT runs trained on
**different corpora** had `A` matrices agreeing to 1.5e-6. `B` is zero-initialised and
`dL/dA ∝ Bᵀ`, so the runs never escaped the LoRA cold start.

It is not an SFT-specific bug and not an advantage-weighting artifact
(`train/advantage ≡ 1.0`, `advantage_mode: uniform`, `loss_fn: cross_entropy`, train loss
a normal ~0.9). The optimizer is healthy: adapter movement scales **exactly linearly**
with LR (25× LR → 25× movement, to three significant figures). The LR was simply ~25×
too small.

### What this invalidates

1. **All 36 shipped SFT arms are ≈ the base model** (retail, airline, finance × both
   context regimes).
2. **It explains the Stage-3 RL null result.** All four retail RLVR arms "warm-started"
   from adapters numerically indistinguishable from base — so they *were* the same model.
   The 5/20, 5/20, 6/20, 5/20 spread was not noise swamping an effect; there was no
   difference to detect.
3. **The Stage-2 comparison measures something real but differently framed.** Since the
   model never changed, the stable 20–35% gaps between arms are *context* effects on the
   base model: having a thought in the prompt makes the action tokens easier to predict.
   That replicates across three datasets and is a legitimate result — but it is not
   "SFT on Act-PRM thoughts yields a better policy."

### The tell we already had

First-vs-last on the old runs would have exposed this immediately. Across all 18
hide-regime arms at 4e-5, held-out action-subspan PPL moved between **−0.12% and +0.14%**
and got *worse* in 5 of 18; accuracy moved at most **0.08pp** (finance `thoughts_policy`
and `thoughts_base`: exactly +0.00pp). `analyze_sft.py` reported *best* PPL, and the min
of a noisy flat line always looks like a result.

## The fix and the re-run

- **lr 1e-3 trains.** Retail `actions_only`, hide: `3.8355 3.8099 3.7839 3.7576 3.7112
  3.6702` over b10..b59 — monotonic, −4.31%, largest drop in the final interval. At 150
  batches the same arm reaches −7.4% by b80 and is still descending.
- **lr 1e-4 is dead**, confirmed with a completed arm: `3.8446 3.8429 3.8395 3.8449`
  (0.14%, non-monotonic), final adapter 9.2e-5. Tier dropped rather than re-confirmed
  across two more datasets (~25 GPU-hours saved).
- **Batch cap raised 60 → 150** with `--early_stop_patience 3`, so arms that plateau stop
  themselves and only arms still learning spend the budget.

### Training vs reporting spans

Training is on the **full thought+action span** — Act-PRM and `expert_thoughts` must
learn to produce the thought. Reporting is **action-only on both splits**:
`eval_actiononly_{ppl,accuracy}` on eval and `train/actiononly_{ppl,accuracy}` on train
(the latter added here; it previously existed only on finance). Both use the same
`action_start_token` boundary, so trained and scored spans cannot drift.
`--train_action_only` remains available as a loss-masking ablation, off by default.

`extract_action` resolves 100% of assistant steps on all three datasets (retail 92/92,
airline 37/37, finance 340/340).

### A caveat on the old cross-domain table

Retail's subspan numbers came from our implementation, airline's and finance's from the
other box's. They differ by exactly 2 tokens on 469/469 steps — ours includes the opening
`<tool_call>\n`, theirs starts after it — so ours reads ~0.3pp optimistic on accuracy and
~1–3% low on PPL. It is a constant offset applied to every arm within a dataset, so
within-dataset rankings are unaffected. The re-run uses one definition throughout.

## The result that reframes everything

The first properly-trained arm — retail `actions_only`, hide, lr 1e-3, 150 batches:

```
PPL  3.8307 -> 3.0719   (-19.81%, monotonic across all 15 eval points, never plateaued)
acc  0.7599 -> 0.7652   (+0.53pp)
```

Against the same arm at 4e-5: **-0.11% PPL, +0.04pp accuracy**.

Now put that next to the *old* (4e-5, i.e. untrained) hide-regime numbers:

| arm | old best PPL @4e-5 | |
|---|---|---|
| actions_only | 3.8378 | ← trains to **3.0719** |
| thoughts_base | 3.1735 | not yet retrained |
| thoughts_policy | 3.1824 | not yet retrained |
| expert_thoughts | 2.9165 | not yet retrained |

A properly trained `actions_only` (3.07) already **beats both Act-PRM thought arms as we
previously reported them** (3.17 / 3.18) and closes most of the gap to the expert-thought
oracle (2.92). Those old thought-arm numbers were never a training result — they were the
base model being scored with a thought in its context. Once the baseline actually trains,
it passes them.

This does **not** say thoughts don't help: the thought arms have not been retrained yet,
and they start from a lower PPL so they may go lower still. What it does say is that
**no previously reported Stage-2 ranking can be carried forward**. The comparison has to
be redone end-to-end at a working LR, which is what the sweep is doing.

Note also the curve never flattened at 150 batches and early stopping never fired, so
150 is still short of convergence — the arms may be step-limited even now.

## LR probe on a thought arm — why 3e-3

`actions_only` alone would have selected 1e-3. Probing `thoughts_base` (retail, hide, 30
batches, held-out action-subspan PPL) shows that is the wrong choice:

| lr | b10 | b20 | b29 | Δ | accuracy |
|---|---|---|---|---|---|
| 1e-4 | 3.1756 | 3.1773 | 3.1772 | **−0.05%** | 0.7819 → 0.7822 |
| 1e-3 | 3.1673 | 3.1515 | 3.1297 | +1.19% | 0.7824 → 0.7819 |
| **3e-3** | 3.1281 | 3.0277 | **2.9112** | **+6.94%** | 0.7824 → **0.7838** |

3e-3 is ~6× faster than 1e-3 over the same 29 batches, monotonic, with no instability, and
it is the first setting to move **accuracy** at a visible rate. By b29 it reaches 2.911 —
already past the old `expert_thoughts` "oracle" number (2.9165), which came from the
untrained regime.

1e-4 is dead on the thought arm too, exactly as on `actions_only`. That is the "just to be
safe" check, now done on both arm types.

The sweep therefore runs at **3e-3**, 150-batch cap, early stopping patience 3.

*(Correction to an earlier read: `thoughts_base` at 1e-3 is not improving at half the rate
of `actions_only` — that came from comparing b27 against b30. At matched b29/b30 they are
+1.19% vs +1.23%.)*

## Reference point: a fully trained baseline

`retail actions_only, hide, lr 3e-3, 150 batches` — complete, 15 eval points:

```
3.780 3.673 3.492 3.297 3.072 2.861 2.696 2.587 2.511 2.459 2.422 2.395 2.371 2.355 2.342
Δ -38.03%   accuracy 0.7607 -> 0.7726 (+1.19pp)
```

The tail deltas shrink (0.027, 0.024, 0.016, 0.013), so this is near convergence — 150
batches is about right at 3e-3, unlike at 1e-3 where the curve was still accelerating at
b149.

Same arm, all LRs:

| lr | batches | final PPL | Δ | Δ accuracy |
|---|---|---|---|---|
| 4e-5 (shipped) | 60 | 3.8455 | −0.11% | +0.04pp |
| 1e-4 | 60 | 3.8449 | −0.14% (non-monotonic) | +0.00pp |
| 1e-3 | 150 | 3.0719 | +19.81% | +0.53pp |
| **3e-3** | 150 | **2.3418** | **+38.03%** | **+1.19pp** |

The trained baseline at 2.342 sits far below every arm in the original table
(`expert_thoughts` 2.9165, `thoughts_base` 3.1735, `thoughts_policy` 3.1824). Whatever
the thought arms do at 3e-3, this is the number they have to beat — and the old
"thoughts win" ordering is not a valid prior for it.

## First valid thoughts-vs-no-thoughts comparison (retail, hide, 3e-3, 150 batches)

Both arms trained to convergence at a working LR:

| arm | PPL curve (b10..b149) | final PPL | Δ | final acc |
|---|---|---|---|---|
| actions_only | 3.780 … 2.355 2.342 | **2.3422** | −38.03% | 0.7726 |
| expert_thoughts | 2.886 … 2.126 2.115 | **2.1149** | −26.71% | **0.7964** |

**Thoughts still win** — 9.7% lower action-token PPL and +2.4pp accuracy.

But the margin is roughly **half** what the broken runs implied:

| | actions_only | expert_thoughts | gap |
|---|---|---|---|
| old (4e-5, untrained) | 3.8378 | 2.9165 | **24.0%** PPL, +2.9pp |
| new (3e-3, converged) | 2.3422 | 2.1149 | **9.7%** PPL, +2.4pp |

The baseline gained far more from proper training (−38.0%) than the thought arm did
(−26.7%), so most of the apparent advantage in the original table was the untrained
baseline being bad, not the thoughts being good. The effect is real and survives; it is
just considerably smaller than reported.

The accuracy advantage is the more stable of the two (2.9pp → 2.4pp), which matters
because accuracy is the quantity closest to "does the agent pick the right action".

Still to come: `thoughts_policy` and `thoughts_base` — the actual Act-PRM arms. The
question they answer is how much of this 9.7% / 2.4pp oracle gap the *inferred* thoughts
recover.

## HEADLINE: retail hide, all four arms converged (lr 3e-3, 150 batches)

| arm | PPL | vs base | accuracy | vs base | PPL gap recovered | acc gap recovered |
|---|---|---|---|---|---|---|
| actions_only (baseline) | 2.3422 | — | 0.7726 | — | — | — |
| thoughts_base (Act-PRM, base-scored) | 2.2003 | 6.06% | 0.7890 | +1.64pp | **60%** | **66%** |
| thoughts_policy (Act-PRM, policy-scored) | 2.2059 | 5.82% | 0.7925 | +1.99pp | **58%** | **81%** |
| expert_thoughts (oracle) | 2.1071 | 10.04% | 0.7973 | +2.47pp | — | — |

Oracle gap: 0.2351 PPL / 2.47pp accuracy.

Three things this establishes on retail:

1. **The ordering holds**: baseline < Act-PRM < oracle, monotone on both metrics.
2. **Act-PRM recovers ~60% of the oracle PPL gap and 66–81% of the accuracy gap.**
3. **The two Act-PRM variants agree to 0.0056 PPL and 0.35pp accuracy** despite being
   scored by different models — so the effect does not hinge on the scorer choice. (They
   swap rank between the two metrics, which is a good reminder that a 0.006 PPL
   difference is not a result.)

Against the old untrained numbers the recovery fractions were 71% PPL / 76% accuracy — so
the *qualitative* Act-PRM claim survives essentially intact. What does not survive is the
magnitude: the oracle's advantage over the baseline fell from 24.0% to 10.0% PPL, because
a properly trained baseline is far better than the broken one (2.34 vs 3.84).

Figures: `notebooks/figs_sft/sft_curves_subspan_hide_lr3e3.png`.

## Open questions

- **Accuracy does not move.** At 1e-3, PPL improves 4.31% while held-out accuracy goes
  0.7602 → 0.7609 and the whole four-LR panel spans 0.13pp. The model gets better
  calibrated on action tokens without changing its argmax. If next-action accuracy is the
  quantity the Act-PRM story rests on, nothing has moved it yet.
- **Train loss was uninformative at 4e-5 but IS informative at 3e-3.** At 4e-5 the train
  curves are pixel-identical across a 25× LR range (same spikes at b20/b36/b47) — with the
  model barely moving, the loss just reads out per-batch difficulty. At 3e-3 the train
  action-only PPL trends down by nearly as much as eval: `actions_only` first-half 3.135 ->
  second-half 2.007 (−36.0%, slope −0.0107/batch) against eval's −38.0%; `expert_thoughts`
  −19.1% against eval's −27.0%. The trend is invisible point-to-point because per-batch
  noise is huge (stdev 2.69 on a mean of ~2.5, range 1.0–22.4, ~8× the mean) — it needs
  half-means or a regression to see. Eval by comparison has stdev 0.495 and is monotonic.
- **Step- or LR-limited?** 1e-3 was still descending at b150. The `thoughts_base` probe
  (1e-4 / 1e-3 / 3e-3, 30 batches) is intended to separate these.

## Live results — 1e-3 arms

<!--RESULTS-->
```
arm                                                  reg     b   eval ppl first->last      Δ%     acc  train ao ppl
----------------------------------------------------------------------------------------------------------------------
retail/actions_only_lr1e_3                           hide   59     3.8355 ->   3.6702   4.31%  0.7609        1.6094
retail/actions_only_lr1e_3_nb150                     hide  149     3.8307 ->   3.0719  19.81%  0.7652        2.1719

2/2 arms complete   (Δ% = held-out action-subspan PPL improvement, higher is better)
```
_last refreshed: 2026-08-20 22:40_
<!--/RESULTS-->

## Infrastructure

Two cron entries keep this moving without supervision (agent-launched background shells
kept being killed with their process group):
- `*/5 sweep_guard.sh` — runs the `thoughts_base` probe once, then keeps the LR matrix
  advancing. Idempotent, `flock`-guarded, no-ops while a trainer is live.
- `*/20 snapshot_results.sh` — refreshes notes/CSVs/figures, updates the table above,
  backs `metrics.jsonl` into dotsync, and commits.

Resume correctness: completion is an explicit `.done` marker, **not** `step_best` —
`step_best` is written at the first eval, so an interrupted arm used to look finished and
be skipped forever. `NUM_BATCHES` rides in the run tag so a budget change re-runs.
