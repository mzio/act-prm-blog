# On Learning to Think with Action Process Reward Models (Act-PRMs)

Companion site + notebooks + experiments for the ICML 2026 RLxF workshop paper
**[On Learning to Think with Action Process Reward Models](https://openreview.net/forum?id=2zsteCP2wy)**
(Michael Zhang & Madison Ho).

**tldr:** given easy-to-collect, *action-only* demonstration logs, treat the missing
thoughts as latent variables — sample candidate thoughts $z$ from the LLM itself, reward each by the
likelihood it induces on the *observed next action*, $\tilde r(z) = p(x \mid s, z)$, and train with
policy gradient. The EM derivation, interactive demos, and results live in the blog post
(`index.html`, deployed via GitHub Pages).

## Repo layout

```
index.html                     the blog post (single page, interactive)
assets/css/style.css           typography + layout + components
assets/js/hero.js              animated hero banner (canvas: s → sampled z's → x chain)
assets/js/charts.js            SVG bar charts (paper Tables 1 & 2) + signal-strip diagrams
assets/js/demo.js              interactive thought-sampling playground (E-step walkthrough)
assets/js/main.js              KaTeX + hoverable equation terms + quick-nav + bibtex copy
assets/img/                    plots, reward curves, memes/GIFs extracted from the slides
notebooks/act_prm_transformers.ipynb   didactic HF Transformers walkthrough (Colab-friendly)
notebooks/act_prm_tinker.ipynb         same loop via the Tinker training API
scripts/act_prm_length_penalty.py      train + demo script: length-penalized thought reward
runs/                          JSON logs + checkpoint URLs from script runs
.nojekyll                      serve as-is on GitHub Pages
```

## Quickstart: the site

```bash
# preview locally
python3 -m http.server 8000    # → http://localhost:8000

# deploy: push to GitHub, then Settings → Pages → "Deploy from a branch" → main / (root)
```

Math renders via the KaTeX CDN; everything else is self-contained static files. Asset links carry
`?v=N` cache-busters — bump them when editing CSS/JS.

## Setup: notebooks + training scripts

Everything runs through [`uv`](https://docs.astral.sh/uv/) with ephemeral environments — no venv to
manage. You need one secret: a [Tinker](https://thinkingmachines.ai/blog/announcing-tinker/) API key
in a `.env` file at the repo root (gitignored):

```
TINKER_API_KEY=sk-...
```

- **`notebooks/act_prm_transformers.ipynb`** — the whole algorithm in plain HF Transformers + PyTorch
  on Qwen3-0.6B: prompt reversal, sampling $G$ thoughts, computing $p(x \mid s, z)$ from raw logits,
  a REINFORCE M-step. Runs on a free Colab T4.
- **`notebooks/act_prm_tinker.ipynb`** — the same loop against the Tinker API, the way the paper's
  experiments ran: `sample_async` (E-step), `compute_logprobs_async` (reward),
  `forward_backward_async(loss_fn="importance_sampling")` (M-step).

## Experiment: what happens if thoughts must be *short* as well as *predictive*?

`scripts/act_prm_length_penalty.py` trains an Act-PRM whose reward subtracts a length penalty:

$$r_\text{pen}(z) \;=\; \underbrace{p(x \mid s, z)}_{\text{action likelihood}} \;-\; \lambda \cdot \underbrace{|z| / L_\text{max}}_{\text{fraction of thought budget used}}$$

Both terms are $O(1)$-scale, so with $\lambda = 0.15$ a thought that maxes out the token budget must
buy ~0.15 of extra action-likelihood over a concise alternative to win its group. Group weights are
the normalized clamped rewards (falling back to likelihood-normalization if the penalty pushes a
whole group negative), and the *committed* thought is the penalized-reward argmax — so the shortest
thought that still explains the logged action carries the trajectory forward.

**Data**: [`mzio/aprm-snorkelai_agent_finance_reasoning`](https://huggingface.co/datasets/mzio/aprm-snorkelai_agent_finance_reasoning)
— successful multi-turn financial tool-calling rollouts. The loader keeps only successful
trajectories and **cuts the narration out of every logged action**, so assistant turns are bare
`<tool_call>…</tool_call>` blocks (the action-only condition). Long middle observations are elided
codebase-style.

```bash
# full training run (the one reported below)
uv run --with tinker --with datasets --with python-dotenv \
       --with "transformers>=4.51" --with numpy --with jinja2 \
  python -u scripts/act_prm_length_penalty.py train \
    --model Qwen/Qwen3-8B --lora-rank 8 --group-size 8 \
    --length-penalty 0.15 --max-thought-tokens 200 \
    --num-trajectories 8 --eval-trajectories 2 --max-steps-per-traj 6 \
    --em-iterations 16 --checkpoint-every 4 \
    --log-path runs/length_penalty_qwen3_8b_full.json

# resample from any saved checkpoint (paths printed during training + stored in the run log)
uv run ... python scripts/act_prm_length_penalty.py demo --sampler-path "tinker://..."
```

The run log (`runs/*.json`, updated after every iteration) records per-candidate thoughts,
likelihoods, penalties, and weights for every step — plus `tinker://` sampler URLs for the initial,
periodic, and final checkpoints, so everything below can be regenerated post-hoc.

### Results

Run: `Qwen/Qwen3-8B`, LoRA r=8, G=8 thoughts/state, λ=0.15, L_max=200 — 8 train + 2 held-out
trajectories × 6 steps, up to 100 EM iterations with early stopping (patience 12 on held-out
likelihood). **Early-stopped at iteration 59** (best held-out p=0.888 @ iteration 43). One external
kill at iteration 48 was recovered losslessly via `--resume-state` from the checkpoint's saved
trainer state. Full logs: `runs/length_penalty_qwen3_8b_100.json` (main),
`runs/length_penalty_qwen3_8b_full.json` (earlier 16-iter run),
`runs/length_penalty_qwen3_8b_100_killed_at_25.json` (25-iter partial that motivated resumability).

#### The two-phase dynamic

Optimizing *likelihood minus length* produced a clean two-phase trajectory, visible in the curves:

![Action likelihood during training](assets/img/lenpen-likelihood.png)

![Thought length during training](assets/img/lenpen-thought-length.png)

1. **Phase 1 (iterations ~0–35): likelihood buys length.** Mean action likelihood climbs
   0.47 → 0.89 (held-out), and thoughts *grow* — from ~63 to a peak of ~85 tokens — because early
   on, extra explanatory detail buys more likelihood than the λ·|z|/L_max tax costs. The penalty only
   shapes *selection* at this stage.
2. **Phase 2 (iterations ~38–59): the length tax takes over.** Once likelihood saturates
   (marginal gains < marginal length cost), thoughts compress hard: ~85 → ~45 mean tokens, with
   selected thoughts down to ~37, at **no cost to training likelihood** (holds ~0.95). Held-out
   likelihood plateaus at ~0.87–0.89, which is what triggers early stopping.

#### Checkpoint generations: the same probe step over training

Held-out probe: *"How significant is the company's international operations footprint in terms of
asset allocation and income generation as of 2022–2024?"* → logged action
`get_descriptions(company_name="meta")`. Best-of-G thought at each checkpoint:

| iter | p(x\|s,z) | \|z\| | best-of-G thought |
|---|---|---|---|
| 0 | 0.0018 | 28 | "I need information about meta's international operations. First, I should identify the relevant tables that describe their operations and financial data." |
| 8 | 0.113 | 31 | "I need to examine the international operations of Meta. First, I should determine which tables and data points exist in the database for this company." |
| 16 | 0.275 | 44 | "I need to understand how to get the data I need for acme. I'll first list the available table descriptions for the company meta…" *(note the few-shot's "acme" leaking in)* |
| 24 | 0.478 | 46 | "First, I need an understanding of the tables available for Meta to answer questions about their international footprint in terms of assets and income…" |
| 32 | 0.763 | 51 | "To understand the context of Meta's operations, I'll start by retrieving the descriptions of available tables to determine the relevant data sources…" |
| 40 | 0.956 | 56 | "I need details about international operations for Meta. first step is to fetch the available tables and their descriptions, which will guide me to the correct data sources…" |
| 48 | 0.994 | 48 | "I must analyze meta's operations globally for this report. First I'll check its available tables to understand where the data might be stored…" |
| 56 | 0.997 | 31 | "To get insights into international operations for meta, I must first examine available tables and structures by retrieving descriptions to understand location and income relationships effectively." |
| final (stop @ 59) | 0.996 | 38 | — |

By the end, **every one of the 8 sampled thoughts scores p≈0.997** on a step whose initial best was
p=0.0018 (a ~550× improvement) — so the length penalty becomes the sole differentiator, and the
group is ranked purely by brevity.

#### Amusing failure modes

- **Token-saving typos.** Under length pressure the model started inventing contractions —
  one selected thought opens *"I'am examining Meta's operations…"* (a token cheaper than "I am").
- **Confident confabulation.** Late-training thoughts casually cite plausible-but-invented table
  names (`fact_revenue`, `dimension_region`, `meta_position_breakdown_this_year`) — the action
  likelihood is indifferent to whether the *reasoning's props* are real, only whether the right
  action follows.
- **Few-shot bleed-through.** The mid-training checkpoint at iteration 16 briefly addresses "acme",
  the company from the built-in few-shot example.

#### Checkpoints (Tinker sampler weights)

All resumable: each periodic checkpoint also saved full trainer state (`state_path` in the run log).

| iter | sampler path |
|---|---|
| 0 | `tinker://fa604ca3-…:train:0/sampler_weights/lenpen-iter-0` |
| 8–48 | `…/lenpen-iter-{8,16,24,32,40,48}` (see run log for full URLs + state paths) |
| 56, final | `tinker://c637f323-…:train:0/sampler_weights/lenpen-iter-{56,100}` |
| best eval (iter 43) | `…/lenpen-best-43` (recorded in run log) |

---

<details>
<summary>Original project planning notes (pre-build)</summary>

## Todo:

### Static-site Blog

We want to make a clean, minimal, yet interactive blog post / demo site that I can upload to
github pages. This might roughly be a motivating aesthetic:
- https://michaelzhang.xyz/distillation-is-not-a-heist/
- https://yoonholee.com/meta-harness/

Using the notes at `docs/1.0-notes-act_prm.ipynb`, write up a blog post that combines the prose there
(basically word-for-word, if not with edits + revisions by first reading over an thinking about as a
Technical Editor for content that is scientific yet didacts and meant for the masses; Nathan Lambert
or Michael Levine ah), with our results at the following sources:

- docs/slides.pptx
- docs/paper_act_prm-10.pdf

We want the results + experiment descriptions to be standard, i.e., Tables, Plots, Reward Curves. We
may want to back-up the initial claims in the notes about just trying to predict future actions from
past ones as not having enough context via the motivating results that we have in the slides.

Additional sources for references may be found at:

- Paper repository: /Users/michael/Documents/projects/papers/paper-act-prm
- Codebase: /Users/michael/Documents/projects/act-prm-tinker (training with Tinker)
- Codebase 2: /Users/michael/Documents/projects/act-prm

#### Requirements

We want to have nice interactive visualizations of the idea and our method. This should all be
present in a static-site that can be served and deployed to GitHub pages, but can make use of
technologies like HTML5, CSS, Canvas, or Three.js (if the latter can be served)

Some inspiration seeds:
1. Given an initial prompt (s), and a logged action (x), spaced apart, we can sample various thoughts
   (z), and compute rewards via p(x | s, z), and bold or select the thought with the highest reward.
   * Each prompt, action, thought should be text, where the thoughts get generated
     character-by-character as an animation
   * The thought that gets selected can also be an animation
2. We can have figures showing the latent variable modeling at play
3. We can have the equation derivation in MathTex or LaTex, with different parts being hoverable and
   glowing in different colors.

The overall site should be a white background (or off-white). We want to use clean simple fonts
(Roboto, Calibri, Arial, Helvetica Neue; thin-styling or regular styling)

We want to have a hero banner or image that is animated and visualizes the method's core conepts at
large.

We want to have the memes and gifs present in the slides when they make sense.

Include a bibtex for the site itself as a blog post and for the ICML RLxF paper.

In the appendix (or as part of illustrating the method), and also in the notebooks, we want to setup
what the prompts and context are like for our thought-generation — how we reverse the prompts first
as (state, action, thought) for offline traces of (state, thought, ___), seeding this with a few
few-shot examples, and then scoring the generated samples via the same model as
p(action | state, thought).

We want the derivation to be didactic and make the EM connections very easy to follow and natural.

### Code / Jupyter notebook

It'd also be good to include a standalone jupyter-notebook using HuggingFace Transformers and Tinker
for walking through how one would sample and train the models via RL. This should be quite didactic
and make it very easy for someone to follow implementations of the core concepts. I have Tinker keys
in a .env file, though you may need to test on Google Colab for the HuggingFace transformers route.

</details>
