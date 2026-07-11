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

$$r_\text{pen}(z) = \underbrace{p(x \mid s, z)}_{\text{action likelihood}} - \lambda \cdot \underbrace{|z| / L_\text{max}}_{\text{fraction of thought budget used}}$$

Both terms are $O(1)$-scale, so with $\lambda = 0.15$ a thought that maxes out the token budget must
buy ~0.15 of extra action-likelihood over a concise alternative to win its group. The *committed*
thought is the penalized-reward argmax — the shortest thought that still explains the logged action
carries the trajectory forward. Standalone writeup with full details:
**[LENGTH_PENALTY.md](LENGTH_PENALTY.md)**.

```bash
# reproduce (needs TINKER_API_KEY in .env)
uv run --with tinker --with datasets --with python-dotenv \
       --with "transformers>=4.51" --with numpy --with jinja2 \
  python -u scripts/act_prm_length_penalty.py train \
    --model Qwen/Qwen3-8B --lora-rank 8 --group-size 8 \
    --length-penalty 0.15 --max-thought-tokens 200 \
    --num-trajectories 8 --eval-trajectories 2 --max-steps-per-traj 6 \
    --em-iterations 100 --checkpoint-every 8 --patience 12 \
    --log-path runs/length_penalty_qwen3_8b_100.json
```

### Results

Run: `Qwen/Qwen3-8B`, LoRA r=8, G=8 thoughts/state, λ=0.15, L_max=200 — 8 train + 2 held-out
trajectories × 6 steps, up to 100 EM iterations with early stopping (patience 12 on held-out
likelihood). **Early-stopped at iteration 59** (best held-out p=0.888 @ iteration 43). One external
kill at iteration 48 was recovered losslessly via `--resume-state` from the checkpoint's saved
trainer state. Full log: `runs/length_penalty_qwen3_8b_100.json` (every candidate thought, reward,
and checkpoint URL, regenerable via `scripts/plot_length_penalty_run.py` and
`scripts/make_eval_tables.py`).

#### The two-phase dynamic

![Action likelihood during training](assets/img/lenpen-likelihood.png)

![Thought length during training](assets/img/lenpen-thought-length.png)

1. **Phase 1 (iterations ~0–35): likelihood buys length.** Held-out action likelihood climbs
   0.47 → 0.89, and thoughts *grow* — from ~63 to a peak of ~85 tokens — because early on, extra
   explanatory detail buys more likelihood than the λ·|z|/L_max tax costs. The penalty only shapes
   *selection* at this stage.
2. **Phase 2 (iterations ~38–59): the length tax takes over.** Once likelihood saturates (marginal
   gains < marginal length cost), thoughts compress hard: ~85 → ~45 mean tokens (selected ẑ down to
   ~37), at no cost to training likelihood (holds ~0.95). Held-out likelihood plateaus at ~0.87–0.89,
   which triggers early stopping.

#### Watching the same eval steps across training

Best-of-G thought for the held-out probe step at each checkpoint — the probe question is
*"How significant is the company's international operations footprint in terms of asset allocation
and income generation as of 2022–2024?"* and the ground-truth tool call being explained is:

```
<tool_call>
{"name": "get_descriptions", "arguments": {"company_name": "meta"}}
</tool_call>
```

| iter | p(x\|s,z) | \|z\| | best-of-G thought |
|---|---|---|---|
| 0 | 0.0018 | 28 | "I need information about meta's international operations. First, I should identify the relevant tables that describe their operations and financial data." |
| 8 | 0.113 | 31 | "I need to examine the international operations of Meta. First, I should determine which tables and data points exist in the database for this company." |
| 16 | 0.275 | 44 | "I need to understand how to get the data I need for acme. I'll first list the available table descriptions for the company meta…" *(the few-shot's "acme" leaking in)* |
| 24 | 0.478 | 46 | "First, I need an understanding of the tables available for Meta to answer questions about their international footprint in terms of assets and income…" |
| 32 | 0.763 | 51 | "To understand the context of Meta's operations, I'll start by retrieving the descriptions of available tables to determine the relevant data sources…" |
| 40 | 0.956 | 56 | "I need details about international operations for Meta. first step is to fetch the available tables and their descriptions, which will guide me to the correct data sources…" |
| 48 | 0.994 | 48 | "I must analyze meta's operations globally for this report. First I'll check its available tables to understand where the data might be stored…" |
| 56 | 0.997 | 31 | "To get insights into international operations for meta, I must first examine available tables and structures by retrieving descriptions to understand location and income relationships effectively." |
| 59 (stop) | 0.996 | 38 | "To understand asset allocation and income generation across regions, I should first list out all tables related to meta to make sure I have all relevant data structures ready for the analysis query." |

By the end, **every one of the 8 sampled thoughts scores p≈0.997** on a step whose initial best was
p=0.0018 (~550×) — the length penalty becomes the sole differentiator, and the group is ranked
purely by brevity.

The same effect across **different held-out tasks** (step-1 thought for each eval question, at
every 10th iteration — watch the English erode into telegraphic noun-stacks by iteration 59). This
run held out 2 tasks; the λ-sweep runs below hold out 5:

<details>
<summary><b>Per-task thought-evolution tables (click to expand)</b></summary>

_Held-out tasks, every 10 EM iterations; "selected ẑ" is the penalized-reward argmax of G=8 sampled thoughts (the one committed to context)._


#### Task 1

**Question:** Here is the question : How significant is the company's international operations footprint in terms of asset allocation and income generation as of 2022-2024?, Here are the companies name in the database to query for : meta

**Logged first action:**

```
<tool_call>
{"name": "get_descriptions", "arguments": {"company_name": "meta"}}
</tool_call>
```

| EM iter | p(x\|s,ẑ) | \|ẑ\| tokens | selected thought ẑ |
|---:|---:|---:|---|
| 0 | 0.0018 | 29 | I need data on meta's international operations. First I should identify which tables are available for this company to gather the necessary details. |
| 10 | 0.1356 | 30 | I need details about meta's international operations. First, I should identify which tables exist for this company to understand what data is available. |
| 20 | 0.3919 | 49 | I need to check which tables and columns are available for the company "meta". I will execute a query to get the descriptions of all tables and columns related to this company so that I can understand what data I can work with. |
| 30 | 0.6758 | 50 | I need a comprehensive view on meta's international activities, as it spans asset allocation and income. To prepare, I'll first check the tables related to meta for table descriptions so I understand the possible structures and fields available for this query. |
| 40 | 0.9624 | 43 | To understand how impactful Meta's international operations are within these years, I should first identify the relevant tables. This helps ensure I'm gathering the correct data for assessing asset allocation and income participation across countries. |
| 50 | 0.9945 | 45 | To calculate the international operations' significance, I must identify relevant tables for Meta that store detailed financial metrics, particularly those covering tables with "international" keywords. This will help align the necessary data structure for comparison. |
| 59 | 0.9964 | 37 | Since I need detailed selections about asset and income allocations for meta across multiple years, I should first understand available tables to accurately select relevant columns for comprehensive queries addressing this question. |

#### Task 2

**Question:** Here is the question : How well-positioned is the company to meet short-term liquidity needs given its current investment portfolio composition?, Here are the companies name in the database to query for : microsoft

**Logged first action:**

```
<tool_call>
{"name": "get_descriptions", "arguments": {"company_name": "microsoft"}}
</tool_call>
```

| EM iter | p(x\|s,ẑ) | \|ẑ\| tokens | selected thought ẑ |
|---:|---:|---:|---|
| 0 | 0.0034 | 42 | I need to assess how well-positioned the company is to meet short-term liquidity needs based on its investment portfolio. First, I should understand the available data structures for Microsoft to guide my analysis. |
| 10 | 0.1872 | 37 | I need to analyze Microsoft's liquidity and investment portfolio. To do that, I will start by checking which tables and descriptions are available for Microsoft to understand the data structure. |
| 20 | 0.3859 | 40 | To assess short-term liquidity needs, I need to understand the company's investment portfolio. First, I'll collect all table descriptions related to the company to identify the right datasets for analysis. |
| 30 | 0.7163 | 73 | Beginning with the Microsoft company, to analyze whether it's ready for short-term liquidity needs, I need to understand which aspects of its investment portfolio matter. I should first identify which tables and fields describe this portfolio. By retrieving the descriptions, I can map out relevant data points like liq… |
| 40 | 0.9666 | 37 | The question focuses on semi-liquid investments in the short term. I should check the available table descriptions for Microsoft to identify relevant tables for investment portfolio mixes and liquidity assessments. |
| 50 | 0.9967 | 40 | I need to analyze how Microsoft is positioned for short-term needs based on its investment portfolio. I'll check table structures to identify relevant categories like investments or financial status to model this analysis. |
| 59 | 0.9965 | 34 | Getting table descriptions for microsoft helps simplify structuring queries for accessing investment data about liquidity requirements and portfolio assessments. Comprehensive table structure existence understanding ensures appropriate metrics retrieval. |

</details>

#### λ sweep: push harder, collapse sooner

Rerunning with **λ = 0.4** and **λ = 1.0** for the full 100 iterations (early stopping disabled)
adds a third act to the story:

![Held-out likelihood across length penalties](assets/img/lenpen-sweep-likelihood.png)

![Held-out thought length across length penalties](assets/img/lenpen-sweep-length.png)

Likelihood climbs identically for every λ — but at matched likelihood, thoughts are shorter with
higher λ (at iteration 20: ~62 → 50 → 33 selected tokens for λ = 0.15 / 0.4 / 1.0). Both aggressive
runs converge to terse ~33-token thoughts at p≈0.99 by iteration ~70… and then **collapse**: past
iteration ~78 (λ=1.0) / ~84 (λ=0.4) the policy falls into a degenerate attractor — thoughts blow up
to the 200-token cap as literal *"percentage percentage percentage…"* repetition, which still
reward-hacks p≈0.85 on the logged action while held-out likelihood craters 0.91 → 0.70. Stronger
penalty, earlier collapse — and the λ=0.15 run's early stopping (iteration 59) exited safely at the
plateau. Full analysis in [LENGTH_PENALTY.md](LENGTH_PENALTY.md).

**Fourth variant — lift-per-token + lexicographic selection + KL anchor** (violet in the plots):
rewarding thoughts by likelihood-*lift over no-thought* per token, committing the shortest
near-best candidate, and anchoring to the base model with a small KL (0.05) gives the **fastest
climb, highest peak (~0.92 @ iter 42), and earliest compression with no length bump** — but the
small KL only delays the babble collapse (~iteration 77; this run's filler: *"category category
insights…"*). Details in [LENGTH_PENALTY.md](LENGTH_PENALTY.md).

<details>
<summary><b>Cross-λ thought-evolution tables (same tasks, same checkpoints — click to expand)</b></summary>

_Selected thought ẑ (penalized-reward argmax of G=8) for the **first step** of shared held-out tasks, under different length penalties λ. “—” = run already finished (λ=0.15 early-stopped at iter 59)._


### Task 1

**Prompt:** Here is the question : How significant is the company's international operations footprint in terms of asset allocation and income generation as of 2022-2024?, Here are the companies name in the database to query for : meta

**Logged action:**

```
<tool_call>
{"name": "get_descriptions", "arguments": {"company_name": "meta"}}
</tool_call>
```

| iter | λ=0.15 | λ=0.4 | λ=1.0 |
|---:|---|---|---|
| 0 | p=0.0018, 29 tok — “I need data on meta's international operations. First I should identify which tables are available for this company to gather the necessary details.” | p=0.0022, 32 tok — “I need to understand the international operations footprint of meta. First, I should check the available tables to see what data is available for this company.” | p=0.0008, 30 tok — “I need to analyze meta's international operations. First, I should check which tables are available for this company to understand the data structure.” |
| 10 | p=0.14, 30 tok — “I need details about meta's international operations. First, I should identify which tables exist for this company to understand what data is available.” | p=0.17, 35 tok — “I need to assess the international operations of meta. The first step is to identify the relevant tables for this company, so I will list the available table descriptions.” | p=0.14, 37 tok — “I need information about Meta's international operations footprint for 2022-2024, so I should first identify which tables are available for this company.” |
| 20 | p=0.39, 49 tok — “I need to check which tables and columns are available for the company "meta". I will execute a query to get the descriptions of all tables and columns related to this company so that I can understand what data I can wo…” | p=0.38, 37 tok — “I need data on Meta's international operations. I'll start by checking the description of the Meta tables to understand available data fields, which might include international assets and income.” | p=0.37, 43 tok — “To answer how significant Meta's international operations are, I need to understand what tables are available for the Meta company to analyze their global operations. I'll list the available table descriptions to identi…” |
| 30 | p=0.68, 50 tok — “I need a comprehensive view on meta's international activities, as it spans asset allocation and income. To prepare, I'll first check the tables related to meta for table descriptions so I understand the possible struct…” | p=0.70, 41 tok — “I need data for Meta. First discover the databases describing Meta to see if information on their international operations exists. I'll query for available table descriptions for Meta to prepare for answering the questi…” | p=0.69, 36 tok — “I should check the descriptions for tables related to meta to understand the structure of the data. This will help me determine which tables to query for international operations footprint details.” |
| 40 | p=0.96, 43 tok — “To understand how impactful Meta's international operations are within these years, I should first identify the relevant tables. This helps ensure I'm gathering the correct data for assessing asset allocation and income…” | p=0.95, 37 tok — “Looking for specifics about Meta's international operations. I need to know the available tables to understand their data structure and find which ones discuss international activities, assets, or income.” | p=0.93, 31 tok — “I need data about meta's international operations. Starting by listing available tables for meta helps me understand table structure and valid queries to conduct next.” |
| 50 | p=0.99, 45 tok — “To calculate the international operations' significance, I must identify relevant tables for Meta that store detailed financial metrics, particularly those covering tables with "international" keywords. This will help a…” | p=0.99, 38 tok — “To explore Meta's international footprint, I'll review their available tables first. This gives insights into how their international operations compare in asset allocation and income generation over the specified timef…” | p=0.99, 34 tok — “I need tables related to meta for the international operations question. First, I'll explore available tables and descriptions to understand which information is available for this company.” |
| 59 | p=1.00, 37 tok — “Since I need detailed selections about asset and income allocations for meta across multiple years, I should first understand available tables to accurately select relevant columns for comprehensive queries addressing t…” | p=1.00, 30 tok — “To understand international operations, I need table structures for meta. This will guide me to extract asset allocation and income details effectively for analysis.” | p=1.00, 32 tok — “To understand the structure of tables related to meta, I should first list available descriptions so I'm prepared for detailed queries and create queries where necessary.” |
| 60 | — | p=1.00, 33 tok — “To address how impactful international operations are for meta, I should examine their descriptions and table structures, which can guide me on what queries might work next.” | p=1.00, 30 tok — “International evaluations for meta. I need to analyze structure and available descriptions to understand data relationships and calculate company performance across various regions and years.” |
| 70 | — | p=1.00, 37 tok — “Since I'm analyzing international operations for meta, starting by retrieving table descriptions will provide me with the structures available for this company, which spans asset allocation and income generation insight…” | p=1.00, 31 tok — “For meta's international analysis, I'll start by retrieving its table descriptions to identify relevant schema structures for asset allocation and income comparisons across years.” |
| 80 | — | p=1.00, 29 tok — “I should analyze client division table descriptions for meta to understand their asset portfolios and revenue categorizations across regions for comprehensive comparative insights analysis.” | p=0.99, 25 tok — “I should explore table structures for meta before calculating percentage growth rates over revenue asset period comparisons needed for this task.” |
| 90 | — | p=0.71, 200 tok — “The user needs percentage change percentage category percentage growth category percentage percentage calculation percentage category percentage percentage percentage category percentage percentage percentage percentage…” | p=0.72, 73 tok — “There's percentage growth percentage percentage percentage percentage percentage percentage percentage percentage percentage percentage percentage percentage percentage percentage percentage percentage percentage percen…” |
| 99 | — | p=0.84, 105 tok — “I need percentage percentage percentage percentage percentage percentage percentage percentage percentage percentage percentage category percentage percentage percentage percentage percentage percentage percentage perce…” | p=0.85, 34 tok — “Percentage percentage percentage percentage percentage percentage percentage percentage percentage percentage percentage percentage percentage percentage percentage percentage percentage percentage percentage percentage…” |

### Task 2

**Prompt:** Here is the question : How well-positioned is the company to meet short-term liquidity needs given its current investment portfolio composition?, Here are the companies name in the database to query for : microsoft

**Logged action:**

```
<tool_call>
{"name": "get_descriptions", "arguments": {"company_name": "microsoft"}}
</tool_call>
```

| iter | λ=0.15 | λ=0.4 | λ=1.0 |
|---:|---|---|---|
| 0 | p=0.0034, 42 tok — “I need to assess how well-positioned the company is to meet short-term liquidity needs based on its investment portfolio. First, I should understand the available data structures for Microsoft to guide my analysis.” | p=0.0024, 46 tok — “I need to assess how well-positioned the company is to meet short-term liquidity needs based on its investment portfolio. First, I will list the available table descriptions for the company to understand the data I can …” | p=0.0064, 38 tok — “I need to assess microsoft's liquidity position based on its investment portfolio. First, I should see which tables exist for this company, so I'll list the available table descriptions.” |
| 10 | p=0.19, 37 tok — “I need to analyze Microsoft's liquidity and investment portfolio. To do that, I will start by checking which tables and descriptions are available for Microsoft to understand the data structure.” | p=0.20, 38 tok — “I need to assess the short-term liquidity of microsoft based on its investment portfolio. First, I will query for the available table descriptions that can provide details on its investment portfolio.” | p=0.16, 27 tok — “I need to assess short-term liquidity of microsoft. First I will determine which tables are available for this company to examine.” |
| 20 | p=0.39, 40 tok — “To assess short-term liquidity needs, I need to understand the company's investment portfolio. First, I'll collect all table descriptions related to the company to identify the right datasets for analysis.” | p=0.38, 43 tok — “To evaluate how well-positioned the company is for short-term liquidity, I need information about the current investment portfolio composition. I'll first check the available table descriptions to determine the proper t…” | p=0.39, 33 tok — “I need to understand which tables are relevant to Microsoft's investment portfolio and liquidity. First, I should check the available table descriptions to find relevant data.” |
| 30 | p=0.72, 73 tok — “Beginning with the Microsoft company, to analyze whether it's ready for short-term liquidity needs, I need to understand which aspects of its investment portfolio matter. I should first identify which tables and fields …” | p=0.71, 47 tok — “To evaluate Microsoft's liquidity, I need to understand its financial context. Initially, I'll retrieve table descriptions to uncover relevant datastreams, which will guide my next steps toward analyzing cash reserves, …” | p=0.72, 47 tok — “For the question about liquidity, I need to understand what tables exist for Microsoft that might relate to investment portfolios. I'll start by retrieving the list of available table descriptions to find relevant data …” |
| 40 | p=0.97, 37 tok — “The question focuses on semi-liquid investments in the short term. I should check the available table descriptions for Microsoft to identify relevant tables for investment portfolio mixes and liquidity assessments.” | p=0.97, 37 tok — “To assess liquidity needs, I need details about Microsoft's investment portfolio and related liquidity tables first. I'll retrieve the descriptions to identify the relevant tables I should continue with.” | p=0.95, 33 tok — “To analyze short-term liquidity, I need tables related to investments and liquidity indicators for microsoft. I will check table descriptions to understand the relevant data structures.” |
| 50 | p=1.00, 40 tok — “I need to analyze how Microsoft is positioned for short-term needs based on its investment portfolio. I'll check table structures to identify relevant categories like investments or financial status to model this analys…” | p=0.99, 34 tok — “I need to analyze liquidity for microsoft. To understand what tables and data are available, I'll list table descriptions for microsoft to explore its investment portfolio details.” | p=0.99, 28 tok — “I need to assess investments for microsoft to evaluate short-term liquidity. First I should list all available table descriptions for visualization insights.” |
| 59 | p=1.00, 34 tok — “Getting table descriptions for microsoft helps simplify structuring queries for accessing investment data about liquidity requirements and portfolio assessments. Comprehensive table structure existence understanding ens…” | p=0.99, 29 tok — “Determine the structure of the table for microsoft to understand investment details, which will help analyze liquidity needs based on their portfolio composition.” | p=1.00, 29 tok — “To analyze liquidity, I first need to understand what tables are available for microsoft. We can assess their investment portfolio with this information.” |
| 60 | — | p=0.99, 31 tok — “To evaluate financially how Microsoft can meet liquidity needs with their investments, I must first understand their table structure and the available data for this company.” | p=1.00, 32 tok — “To determine short-term liquidity, I need table information for microsoft's available tables to evaluate suitable queries for portfolio assessment and cash flow reallocation potential.” |
| 70 | — | p=1.00, 26 tok — “I should obtain descriptions for microsoft tables to analyze investment performance data necessary for assessing liquidity position based on portfolio composition queries.” | p=1.00, 29 tok — “To determine liquidity positioning assessment for microsoft, I need table descriptions to understand available data structures for relevant inventory and cash flow management insights.” |
| 80 | — | p=1.00, 28 tok — “To assess liquidity needs, I should check table descriptions for microsoft to understand structure and available investment data fields relevant to liquidity analysis.” | p=1.00, 30 tok — “To evaluate financial health through investment compositions, I should first discover table structures for microsoft holdings to plan appropriate revenue growth query retrieval strategies accurately.” |
| 90 | — | p=1.00, 41 tok — “This table description retrieval for microsoft investments helps structure category percentage percentage growth percentage category percentage percentage percentage growth percentage percentage category percentage perc…” | p=0.70, 200 tok — “For revenue percentage growth percentage percentage percentage percentage percentage percentage percentage percentage percentage percentage percentage percentage percentage percentage percentage percentage percentage pe…” |
| 99 | — | p=0.98, 73 tok — “I need percentage category percentage information percentage percentage percentage percentage percentage percentage percentage percentage percentage percentage percentage percentage percentage percentage percentage perc…” | p=0.90, 83 tok — “Percentage percentage percentage percentage percentage percentage percentage percentage percentage percentage percentage percentage percentage percentage percentage percentage percentage percentage percentage percentage…” |

</details>

#### Amusing failure modes

- **Token-saving typos.** Under length pressure the model started inventing contractions —
  one selected thought opens *"I'am examining Meta's operations…"* (a token cheaper than "I am").
- **Telegraphic jargon, not caveman.** Late thoughts stay fluent-ish but stack nouns and drop
  articles: *"I need asset table structures table details for geographic areas to ensure precise
  selections analyzing regional exposure accurately by asset types across years."*
- **Confident confabulation.** Late thoughts casually cite plausible-but-invented table names
  (`fact_revenue`, `dimension_region`, `meta_position_breakdown_this_year`) — the action likelihood
  is indifferent to whether the reasoning's *props* are real, only whether the right action follows.

#### Checkpoints (Tinker sampler weights)

All resumable — each periodic checkpoint also saved full trainer state (`state_path` in the run log).

| iter | sampler path |
|---|---|
| 0–48 | `tinker://fa604ca3-…:train:0/sampler_weights/lenpen-iter-{0,8,16,24,32,40,48}` |
| 48 (resumed), 56, final | `tinker://c637f323-…:train:0/sampler_weights/lenpen-iter-{48,56,100}` |
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
