# Action Process Reward Models (Act-PRMs)

On learning to think with Action Process Reward Models (Act-PRMs).

Companion-site + blog for paper at ICML 2026 Workshop RLxF

## Todo:

### Static-site Blog

We want to make a clean, minimal, yet interactive blog post / demo site that I can upload to 
github pages. This might roughly be a motivating aesthetic:  
- https://michaelzhang.xyz/distillation-is-not-a-heist/
- https://yoonholee.com/meta-harness/

Using the notes at `/Users/michael/Documents/projects/act-prm-blog/docs/1.0-notes-act_prm.ipynb`, write up a blog post that combines the prose there (basically word-for-word, if not with edits + revisions by first reading over an thinking about as a Technical Editor for content that is scientific yet didacts and meant for the masses; Nathan Lambert or Michael Levine ah), with our results at the following sources:

- /Users/michael/Documents/projects/act-prm-blog/docs/slides.pptx
- /Users/michael/Documents/projects/act-prm-blog/docs/paper_act_prm-10.pdf


We want the results + experiment descriptions to be standard, i.e., Tables, Plots, Reward Curves. We may want to back-up the initial claims in the notes about just trying to predict future actions from past ones as not having enough context via the motivating results that we have in the slides.

Additional sources for references may be found at: 

- Paper repository: /Users/michael/Documents/projects/papers/paper-act-prm 
- Codebase: /Users/michael/Documents/projects/act-prm-tinker (training with Tinker)
- Codebase 2: /Users/michael/Documents/projects/act-prm


#### Requirements

We want to have nice interactive visualizations of the idea and our method. This should all be present in a static-site that can be served and deployed to GitHub pages, but can make use of technologies like HTML5, CSS, Canvas, or Three.js (if the latter can be served)

Some inspiration seeds:  
1. Given an initial prompt (s), and a logged action (x), spaced apart, we can sample various thoughts (z), and compute rewards via p(x | s, z), and bold or select the thought with the highest reward. 
  * Each prompt, action, thought should be text, where the thoughts get generated character-by-character as an animation  
  * The thought that gets selected can also be an animation

2. We can have figures showing the latent variable modeling at play  

3. We can have the equation derivation in MathTex or LaTex, with different parts being hoverable and glowing in different colors. 

The overall site should be a white background (or off-white). We want to use clean simple fonts (Roboto, Calibri, Arial, Helvetica Neue; thin-styling or regular styling)

We want to have a hero banner or image that is animated and visualizes the method's core conepts at large.

We want to have the memes and gifs present in the slides when they make sense.

Include a bibtex
- For the site itself as a blog post
- For the ICML RLxF paper:
@inproceedings{
anonymous2026on,
title={On Learning to Think with Action Process Reward Models},
author={Michael Zhang and Madison Ho},
booktitle={ICML 2026 Workshop on RL from World Feedback},
year={2026},
url={https://openreview.net/forum?id=2zsteCP2wy}
}


In the appendix (or as part of illustrating the method), and also in the notebooks (next section), we want to setup what the prompts and context are like for our thought-generation.  

You can reference these in the companion codebases, but have an illustrative example (e.g., how we reverse the prompts first as (state, action, thought) for offline traces of (state, thought, ___), seeding this with a few few-shot examples.  

And then scoring the generated samples via the same model as p(action | state, action).

We want the derivation to be didactic and make the EM connections very easy to follow and natural. 

### Code / Jupyter notebook

It'd also be good to include a standalone jupyter-notebook using HuggingFace Transformers and Tinker for walking through how one would sample and train the models via RL. 

This should be quite didactic and make it very easy for someone to follow implementations of the core concepts.

You can use the companion repositories above as Code Base references.  

I have Tinker keys in a .env file, though you may need to test on Google Colab for the HuggingFace transformers route. 

---

## Repo layout (site + notebooks)

```
index.html                     the blog post (single page)
assets/css/style.css           typography + layout + components
assets/js/hero.js              animated hero banner (canvas: s → sampled z's → x chain)
assets/js/charts.js            SVG bar charts (Tables 1 & 2 data) + signal-strip diagrams
assets/js/demo.js              interactive thought-sampling playground (E-step walkthrough)
assets/js/main.js              KaTeX rendering + hoverable equation terms + bibtex copy
assets/img/                    plots, reward curves, memes/GIFs extracted from the slides
notebooks/act_prm_transformers.ipynb   didactic HF Transformers walkthrough (Colab-friendly)
notebooks/act_prm_tinker.ipynb         same loop via the Tinker training API
scripts/act_prm_length_penalty.py      train + demo script: length-penalized thought reward,
                                       real Snorkel Agent Finance data, runs on Tinker
runs/                          JSON logs from script runs
.nojekyll                      serve as-is on GitHub Pages
```

**Length-penalty experiment**: `scripts/act_prm_length_penalty.py` rewards each sampled thought with
`r_pen(z) = p(x | s, z) − λ·|z|/L_max` (likelihood minus the fraction of the thought-token budget
consumed, λ configurable via `--length-penalty`), group-normalizes the clamped rewards for the EM
weights, and picks the committed thought by penalized reward — so the shortest thought that still
explains the logged action wins. Data: `mzio/aprm-snorkelai_agent_finance_reasoning` (successful
rollouts, narration cut out of the logged actions so assistant turns are bare `<tool_call>`s).

```bash
# train (small run; needs TINKER_API_KEY in .env)
uv run --with tinker --with datasets --with python-dotenv \
       --with "transformers>=4.51" --with numpy --with jinja2 \
  python scripts/act_prm_length_penalty.py train --model Qwen/Qwen3-8B

# sample thoughts with the trained checkpoint (path printed by train / stored in runs/*.json)
uv run ... python scripts/act_prm_length_penalty.py demo --sampler-path "tinker://..."
```

**Preview locally**: `python3 -m http.server 8000` then open http://localhost:8000.

**Deploy**: push to GitHub, then Settings → Pages → deploy from branch (`main`, `/ (root)`).
Math renders via the KaTeX CDN; everything else is self-contained static files.

Note: the BibTeX "howpublished" URL in `index.html` is a placeholder
(`https://michaelzhang.xyz/act-prm-blog/`) — update it once the Pages URL is final.





