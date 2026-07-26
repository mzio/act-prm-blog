# cc-2.0 — Training modes & SFT variants (reference)

Two orthogonal knobs. **Advantage mode** = how Stage-1 EM / relabel *weights the sampled
thoughts* (the generator). **SFT variant** = *which target corpus* Stage-2 trains on.

## A. Advantage modes (generator: `--advantage_mode`, `configs/generator/act_prm.yaml`)
For each logged action `x` in state `s`, the E-step samples G candidate thoughts `z` and
computes a length-penalized reward `r = p(x|s,z) - λ·len_frac` (`reward_method: penalty`;
`lift` is the alternative per-token-likelihood variant). The mode maps those G rewards →
per-thought advantages the PG/M-step trains on:

| mode        | what each of the G thoughts gets                                   | use |
|-------------|--------------------------------------------------------------------|-----|
| `em`        | clamped, group-normalized **EM weights** (soft posterior over z)   | **Act-PRM EM default** (Stage 1 training) |
| `best`      | 1.0 on the single highest-reward thought, 0 on the rest (MLE)      | **Stage-1.5 relabel/export** (pick one thought per step) |
| `top_half`  | 1.0 on the better half by reward, 0 on the rest                    | coarser EM alternative |
| `uniform`   | 1.0 on every thought                                               | ablation: treat all samples as positive |
| `grpo`      | mean-centered reward `r - mean(r)` (÷ std if `grpo_normalize`)     | GRPO-style; can be negative |

`grpo_normalize: true` divides the mean-centered reward by the group std. The EM-weight
contract (`advantage_is_computed=True`) means the trainer uses these as-is (no re-centering).

Separately, **Stage-3 agentic RL** uses a different generator (`hf_grpo`, `mean_center: true`)
on the live tau2 env — advantage = env reward − group mean (GRPO), not thought scoring.

## B. Stage-2 SFT variants (`scripts/train_sft.sh <env> <variant>`)
All four: fresh Qwen3-4B-Instruct + LoRA, `SFTTrainer` (advantage-weighted CE),
`--hide_observations`, 60 batches, early-stop on `eval_action_ppl` (split A: 49 train / 10
held-out eval). They differ only in the **supervised target span**:

| variant           | target = supervised tokens                                  | source |
|-------------------|-------------------------------------------------------------|--------|
| `actions_only`    | the logged **action** only (no thoughts) — baseline         | expert data, action span |
| `expert_thoughts` | the **original expert reasoning + action** (oracle upper-bound) | expert data, `--keep_expert_thoughts` |
| `thoughts_policy` | Act-PRM **inferred thought + action**, best thought scored by the **policy** | `data/sft_corpus/tau2_retail/policy` |
| `thoughts_base`   | Act-PRM **inferred thought + action**, best thought scored by the **base** model | `data/sft_corpus/tau2_retail/base`   |

The two `thoughts_*` corpora come from Stage-1.5: relabel the expert data with the best
Stage-1 checkpoint (`--advantage_mode best`), then `export_sft_corpus.py`. The comparison
of interest: does SFT-ing on **inferred** thoughts (policy vs base scored) recover the
lift that **expert** thoughts give over the actions-only baseline?

## Where this lives in code
- Advantage modes: `configs/generator/act_prm.yaml` (23–29) → `generator/act_prm/base.py::_advantages`.
- SFT variants: `scripts/train_sft.sh` (header + case block).
- Pipeline overview: `notes/cc-1.0-aprm_tau2_retail_plan.md`, `CLAUDE.md`.
