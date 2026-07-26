"""
Act-PRM generator — the EM E-step harness for inferring latent thoughts behind
logged actions, built on the HuggingFace generation + logprob-scoring machinery.
"""

from .base import ActPrmGenerator

__all__ = ["ActPrmGenerator"]
