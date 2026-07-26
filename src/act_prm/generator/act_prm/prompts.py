"""
Prompt formats for Act-PRM "prompt reversal".

The core trick: we normally sample a thought then an action (state -> thought ->
action). To infer the thought *behind* a logged action, we reverse the prompt —
show the action first, then continue from an open ``<thought>`` tag — seeded with
one worked few-shot example of the reversed (action-then-thought) format.
"""

THOUGHT_BOS, THOUGHT_EOS = "<thought>", "</thought>"

ACT_PRM_SYSTEM_PROMPT = (
    "You are a helpful assistant that infers reasoning thoughts behind your own observed actions."
)

# One compact worked example that seeds the action-then-thought (reversal) format.
FEWSHOT: list[dict[str, str]] = [
    {
        "role": "user",
        "content": (
            "Here is the question : What was the company's total revenue growth in 2024?, "
            "Here are the companies name in the database to query for : acme"
        ),
    },
    {
        "role": "assistant",
        "content": (
            '<tool_call>\n{"name": "get_descriptions", "arguments": {"company_name": "acme"}}\n</tool_call>\n\n'
            f"{THOUGHT_BOS}\nI need revenue figures for acme. First I should see which tables exist "
            f"for this company, so I'll list the available table descriptions.\n{THOUGHT_EOS}"
        ),
    },
]

FEWSHOT_TRANSITION = "Great! Now do the same for the next task:\n\n## Next Task:\n\n"


def build_thought_prompt_messages(
    state_messages: list[dict[str, str]],
    target_action: str,
    committed: list[str],
    use_fewshot: bool = True,
) -> list[dict[str, str]]:
    """Reversal prompt (as a message list): few-shot seed + prior turns rendered
    as action-then-thought (using the already-committed thoughts) + the target
    action shown first, with the final assistant turn left open right after
    ``<thought>\\n`` so the model continues into the thought.

    Tokenize with ``continue_final_message=True``.
    """
    msgs: list[dict[str, str]] = [{"role": "system", "content": ACT_PRM_SYSTEM_PROMPT}]
    if use_fewshot:
        msgs += [dict(m) for m in FEWSHOT]
    body = [dict(m) for m in state_messages if m["role"] != "system"]
    a_i = 0
    for m in body:
        if m["role"] == "assistant" and a_i < len(committed):
            m["content"] = f"{m['content']}\n\n{THOUGHT_BOS}\n{committed[a_i]}\n{THOUGHT_EOS}"
            a_i += 1
    if body and use_fewshot:
        body[0]["content"] = FEWSHOT_TRANSITION + body[0]["content"]
    msgs += body
    msgs.append({"role": "assistant", "content": f"{target_action}\n\n{THOUGHT_BOS}\n"})
    return msgs


def build_scoring_messages(
    system_prompt: str,
    state_messages: list[dict[str, str]],
    thought: str,
    target_action: str,
) -> list[dict[str, str]]:
    """Natural-order scoring context: state, thought, then action — under the
    task's *original* system prompt (not the reversal system prompt). Used to
    score p(x | s, z)."""
    return (
        [{"role": "system", "content": system_prompt}]
        + [m for m in state_messages if m["role"] != "system"]
        + [{"role": "assistant", "content": f"{thought}\n\n{target_action}"}]
    )


def build_thought_prefix_messages(
    system_prompt: str,
    state_messages: list[dict[str, str]],
    thought: str,
) -> list[dict[str, str]]:
    """The (state + thought) prefix, without the action — used to locate the
    action-token boundary when computing p(x | s, z)."""
    return (
        [{"role": "system", "content": system_prompt}]
        + [m for m in state_messages if m["role"] != "system"]
        + [{"role": "assistant", "content": thought}]
    )
