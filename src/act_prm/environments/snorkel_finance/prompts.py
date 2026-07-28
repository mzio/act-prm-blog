"""
Prompts for Snorkel Agent Finance Reasoning environment.

Two task types with different system prompts:
- finqa: Quantitative financial QA (boxed numeric answers)
- finqa_reasoning: Qualitative financial reasoning (paragraph answers)
"""

# From: https://github.com/snorkel-ai/FinQABenchmark/blob/main/src/prompts/finqa.txt
FINQA_SYSTEM_PROMPT = "Only execute one tool call at a time"

# From: https://github.com/snorkel-ai/FinQABenchmark/blob/main/src/prompts/finqa_reasoning.txt
FINQA_REASONING_SYSTEM_PROMPT = (
    "Only execute one tool call at a time. Think and reason step by step. "
    "Feel free to iteratively gather data as much as you need to answer the question. "
    "Although there is a limit of {max_turns} turns, you can still gather data as much as you "
    "need to answer the question. When you have gathered all the data, answer the question.\n\n"
    "Once you have gathered all the data, then you can gradually start forming an answer. "
    "Once you have formed an answer, generate a 1 paragraph summary of it with all "
    "the relevant figures summarized."
)

SYSTEM_PROMPTS = {
    "finqa": FINQA_SYSTEM_PROMPT,
    "finqa_reasoning": FINQA_REASONING_SYSTEM_PROMPT,
}


def render_prompt(
    user_query: str,
    company: str,
) -> str:
    """Render the user prompt for a snorkel finance task.

    Matches the reference format from FinQABenchmark/src/func-calling-sim.py
    """
    return f"For company `{company}`, here is the question : {user_query}"
