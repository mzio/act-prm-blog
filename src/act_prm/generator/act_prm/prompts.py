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

# ---------------------------------------------------------------------------
# Per-domain few-shots.
#
# WHY PER-DOMAIN: a single shared example poisons any domain that shares the
# example's TASK MODALITY but not its subject matter. The original seed was the
# finance one below (a SQL/table-exploration task about `acme`'s revenue), and it
# was used everywhere. Measured damage on the insurance corpora, which are also
# SQL/table-exploration tasks:
#     insurance EM base   -- 18.8% of thoughts share a 4-gram with the seed,
#                            8.4% open "First I should", 308 verbatim "I need
#                            revenue figures for acme", 57.5% duplicate prefixes
#     insurance EM policy -- 77.7% carry finance vocabulary on insurance tasks
# Sample from the insurance corpus, on an appetite question about a Maine B&B:
#     "I need to find revenue growth in 2024. Looking at revenue growth data could
#      involve checking a revenue growth table." -> get_table_data_dictionary(
#      small_business_insurance_appetite)
# The thought contradicts the action it precedes.
#
# retail/airline were UNHARMED (0.0% shared 4- and 5-grams) because they are
# CRUD/API tasks with a disjoint tool vocabulary -- the seed offered nothing to
# copy, so the model had to ground in the actual episode. Style transferred
# everywhere regardless (the "I need ..." opener runs 78-82% in all domains),
# which is the few-shot doing its job: it should teach FORMAT, not content.
#
# So each example below uses its own domain's real tool names and entities, and
# each thought references something only that episode could supply. Keep them
# short -- the generator caps thoughts at max_thought_tokens (default 96).
# ---------------------------------------------------------------------------

FEWSHOT_FINANCE: list[dict[str, str]] = [
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

FEWSHOT_INSURANCE: list[dict[str, str]] = [
    {
        "role": "user",
        "content": (
            "Here is the company information:\n\n"
            "Company: Ridgeline Auto Body\n"
            "NAICS Code: 811121\n"
            "Annual Revenue: 2400000\n"
            "Number of Employees: 12\n"
            "State: Colorado\n"
            "Line of Business (LOB): general liability\n\n"
            "Question: Is this one inside our appetite?"
        ),
    },
    {
        "role": "assistant",
        "content": (
            '<tool_call>\n{"name": "list_tables", "arguments": {}}\n</tool_call>\n\n'
            f"{THOUGHT_BOS}\nTo judge general liability appetite for Ridgeline Auto Body I need the "
            f"appetite rules keyed on NAICS 811121 and Colorado. I don't know the table names yet, "
            f"so I'll list what the database holds before querying.\n{THOUGHT_EOS}"
        ),
    },
]

FEWSHOT_RETAIL: list[dict[str, str]] = [
    {
        "role": "user",
        "content": "Hi, I'd like to return a pair of headphones I bought last month.",
    },
    {
        "role": "assistant",
        "content": (
            '<tool_call>\n{"name": "find_user_id_by_email", "arguments": '
            '{"email": "d.patel@example.com"}}\n</tool_call>\n\n'
            f"{THOUGHT_BOS}\nA return needs the delivered order, and every order lookup is keyed on "
            f"user ID. The customer gave an email, so I'll resolve that to a user ID before pulling "
            f"their order history to find the headphones.\n{THOUGHT_EOS}"
        ),
    },
]

FEWSHOT_AIRLINE: list[dict[str, str]] = [
    {
        "role": "user",
        "content": "Hi, I need to move my Denver flight to a day earlier if that's possible.",
    },
    {
        "role": "assistant",
        "content": (
            '<tool_call>\n{"name": "get_reservation_details", "arguments": '
            '{"reservation_id": "HX7K2Q"}}\n</tool_call>\n\n'
            f"{THOUGHT_BOS}\nBefore I can rebook I need the current itinerary for HX7K2Q -- the cabin, "
            f"the fare class, and which segment touches Denver -- since change eligibility depends on "
            f"those. I'll read the reservation first.\n{THOUGHT_EOS}"
        ),
    },
]

# domain key -> example. Keys are matched as substrings of the env config name.
FEWSHOTS: dict[str, list[dict[str, str]]] = {
    "insurance": FEWSHOT_INSURANCE,
    "retail": FEWSHOT_RETAIL,
    "airline": FEWSHOT_AIRLINE,
    "finance": FEWSHOT_FINANCE,
}

# Back-compat: the historical name, and the fallback when no domain is resolved.
FEWSHOT: list[dict[str, str]] = FEWSHOT_FINANCE

FEWSHOT_TRANSITION = "Great! Now do the same for the next task:\n\n## Next Task:\n\n"


def resolve_fewshot(domain: str | None) -> list[dict[str, str]]:
    """Pick the few-shot for ``domain`` (an explicit key, or an env-config name such
    as ``act_prm/snorkel_insurance`` / ``act_prm/tau2_airline``).

    Falls back to the finance example -- the historical shared seed -- so an
    unrecognised domain reproduces the old behaviour rather than losing its
    few-shot. Longest key first so a name containing two keys resolves to the more
    specific one.
    """
    if not domain:
        return FEWSHOT
    d = domain.lower()
    for key in sorted(FEWSHOTS, key=len, reverse=True):
        if key in d:
            return FEWSHOTS[key]
    return FEWSHOT


def build_thought_prompt_messages(
    state_messages: list[dict[str, str]],
    target_action: str,
    committed: list[str],
    use_fewshot: bool = True,
    fewshot: list[dict[str, str]] | None = None,
) -> list[dict[str, str]]:
    """Reversal prompt (as a message list): few-shot seed + prior turns rendered
    as action-then-thought (using the already-committed thoughts) + the target
    action shown first, with the final assistant turn left open right after
    ``<thought>\\n`` so the model continues into the thought.

    ``fewshot`` is the domain's worked example (see ``resolve_fewshot``); None
    keeps the historical shared finance seed.

    Tokenize with ``continue_final_message=True``.
    """
    msgs: list[dict[str, str]] = [{"role": "system", "content": ACT_PRM_SYSTEM_PROMPT}]
    if use_fewshot:
        msgs += [dict(m) for m in (fewshot or FEWSHOT)]
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
