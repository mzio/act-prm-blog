"""
Default prompts for the Claude Agent SDK Generators

We handle all tool calls on our end (i.e., not in the Claude Agent SDK loop).
So we instruct Claude to only respond with text for the tools to call.

LENGTH IS DELIBERATELY UNSPECIFIED. This asked for "a SHORT reasoning line" until
2026-09-23, and Claude complied literally: an insurance collection of 1374 thoughts came
back at a 60-char median (vs 447 for the logged GPT-5-mini corpus), 65% unique, with
"Let me check the schema first." appearing 87 times. Those carry no information the tool
call does not already imply, which defeats the point of collecting thoughts at all.
The fix is to remove the length cue, NOT to replace it with "detailed"/"thorough" -- we
want the teacher's natural reasoning, not a length we talked it into. Any future edit
here should keep that neutrality.

ITERATION LOG (insurance, claude-sonnet-4-6, thinking disabled):
  v1 "A SHORT reasoning line describing what you observed, ..."
       median 60 chars, 55.7% unique, 99.1% of tool calls carried a thought
  v2 "Your thoughts before the tool call: what you observed, ..."  (length cue removed)
       median 75, 74.1% unique, 86.2% coverage -- barely longer, and coverage got WORSE:
       with no cue at all Claude sometimes skips the narration rather than expanding it.
  v3 (current) anchors the narration to the LATEST observation instead of describing
       length. In an agentic loop the model narrates intent ("Let me check the schema
       first.") because its actual reasoning happens internally; pointing it at what the
       last tool message *told* it asks for the inference, not the plan. Still says
       nothing about length.
"""

def get_system_prompt_template(name: str) -> str:
    """
    Get the system prompt template for the given name.
    """
    if name == "chat":
        return CHAT_CLAUDE_SYSTEM_PROMPT_TEMPLATE
    # if name == "handoff":
    #     return CLAUDE_SYSTEM_PROMPT_TEMPLATE_HANDOFF
    raise ValueError(f"Sorry, '{name}' prompt template not implemented yet.")


# Note: uses `name` and `arguments` function-calling format (Qwen2.5 - Qwen3 style)
CHAT_CLAUDE_SYSTEM_PROMPT_TEMPLATE = """
You are helping the user complete a task.

The user has instructions and a set of tools they can call. These are provided below.

You DO NOT have direct access to any of the tools listed above. You must NOT attempt to
invoke any tools yourself (no function calls, no built-in tools, no shell commands). 
The user — i.e., the environment — will execute tool calls on your behalf and return 
the result on the next turn.

## Response Format
Each turn, respond ONLY with plain text in the following shape:

- Narrate your reasoning behind the next tool call, based on what the latest user or tool message told you.
- Then EXACTLY ONE `<tool_call>...</tool_call>` block describing the action
  the user should perform next. The block body must be a valid JSON object
  with `name` and `arguments` fields.

i.e., the response should be like this:
'''
{{reasoning-before-the-tool-call}}

<tool_call>
{{"name": "<tool_name>", "arguments": {{"<arg>": "<value>"}}}}
</tool_call>
'''

# User Instructions
'''
{system_prompt}
'''

# User Tools
'''
{tools}
'''
""".strip()
