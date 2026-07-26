"""
``act_prm_traces`` — logged, action-only demonstration trajectories.

An offline "environment": it does not step interactively. It loads successful
agent trajectories (Snorkel Agent Finance reasoning traces by default), strips
each assistant turn down to its explicit action (the ``<tool_call>`` block or a
``Final Answer:`` suffix), and hands whole trajectories to the Act-PRM generator,
which infers the latent thought behind each logged action.
"""

from .env import ActPrmTracesEnv

__all__ = ["ActPrmTracesEnv"]
