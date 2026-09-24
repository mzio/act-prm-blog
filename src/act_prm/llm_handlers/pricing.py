"""
Per-model token pricing (USD per 1M tokens).

Used to compute rollout cost from cumulative token counts when the upstream
SDK doesn't directly surface a dollar figure. Keep this table current — when
you add a new model config under `configs/model/`, add the matching pricing
entry here so the summary doesn't silently show $0.00.

Sources: Anthropic and OpenAI public pricing pages (USD/1M tokens). Update
when published rates change.
"""

from typing import TypedDict


class ModelPricing(TypedDict, total=False):
    input: float          # USD per 1M input tokens
    output: float         # USD per 1M output tokens
    cache_read: float     # USD per 1M cached input tokens (optional)
    cache_write: float    # USD per 1M cache-write tokens (optional)


PRICING: dict[str, ModelPricing] = {
    # --- Anthropic Claude ----------------------------------------------------
    "claude-haiku-4-5":  {"input": 1.00,  "output": 5.00,  "cache_read": 0.10,  "cache_write": 1.25},
    "claude-sonnet-4-6": {"input": 3.00,  "output": 15.00, "cache_read": 0.30,  "cache_write": 3.75},
    # Opus 4.5+ dropped to 5.00 / 25.00 (1M context at standard pricing)
    "claude-opus-4-7":   {"input": 5.00,  "output": 25.00, "cache_read": 0.50,  "cache_write": 6.25},
    "claude-opus-4-8":   {"input": 5.00,  "output": 25.00, "cache_read": 0.50,  "cache_write": 6.25},

    # --- OpenAI -------------------------------------------------------------
    "gpt-5":             {"input": 1.25,  "output": 10.00, "cache_read": 0.125},
    "gpt-5-mini":        {"input": 0.25,  "output": 2.00,  "cache_read": 0.025},
    "gpt-5-nano":        {"input": 0.05,  "output": 0.40,  "cache_read": 0.005},
    "gpt-4o":            {"input": 2.50,  "output": 10.00},
    "gpt-4o-mini":       {"input": 0.15,  "output": 0.60},
}


def compute_cost_usd(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    cache_read_tokens: int = 0,
    cache_creation_tokens: int = 0,
) -> float:
    """
    Compute total cost (USD) for a model from cumulative token counts.

    Anthropic reports ``input_tokens`` EXCLUSIVE of cached tokens, with
    ``cache_read_input_tokens`` / ``cache_creation_input_tokens`` reported
    separately — so the four buckets are additive (no double counting) and each
    is priced at its own rate (cache_read / cache_write).

    Returns 0.0 if the model isn't in PRICING — callers should still surface
    the token counts so the user can spot the gap.
    """
    rates = PRICING.get(model)
    if rates is None:
        return 0.0
    return (
        (prompt_tokens / 1_000_000) * rates.get("input", 0.0)
        + (completion_tokens / 1_000_000) * rates.get("output", 0.0)
        + (cache_read_tokens / 1_000_000) * rates.get("cache_read", 0.0)
        + (cache_creation_tokens / 1_000_000) * rates.get("cache_write", 0.0)
    )
