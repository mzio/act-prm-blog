"""
act-prm — Action Process Reward Models (Act-PRM), PyTorch / HuggingFace edition.

An offline EM over logged action-only demonstrations: sample candidate thoughts
``z`` behind each logged action ``x`` in state ``s``, reward each by the
(length-penalized) action likelihood ``p(x | s, z)``, and take a group-normalized
policy-gradient / REINFORCE step on the (thought + action) tokens.

The training stack (generator harness, PG trainer, LoRA, replay-buffer types) is a
lean fork of ``show-don't-tell-rl`` (import package ``strl``); the Act-PRM E-step
lives in :mod:`act_prm.generator.act_prm` and reads logged traces from
:mod:`act_prm.environments.act_prm_traces`.
"""

__version__ = "0.1.0"
