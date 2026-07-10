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