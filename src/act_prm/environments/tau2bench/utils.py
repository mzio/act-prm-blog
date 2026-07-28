"""
Utilities for the tau2-bench online environment.

Handles conversion of tau2's tool formats to our flat format,
and parsing of tau2 observation strings.
"""

from typing import Any

# Tool for direct assistant-to-user messages (reused from action_lm/env_utils/tau_bench.py)
RESPOND_USER_TOOL: dict[str, Any] = {
    "type": "function",
    "name": "respond_user",
    "description": "Respond or message the user.",
    "parameters": {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "The text content to respond or message the user.",
            }
        },
        "required": ["text"],
    },
}


def select_tasks_by_ids(
    all_tasks: list[Any],
    requested_ids: list[Any],
    domain: str = "",
) -> list[Any]:
    """
    Select tau2 tasks by an explicit list of task ids.

    Matching is robust: each requested id is first matched against each task's
    ``id`` field (compared as strings, since tau2 task ids are strings like
    "0".."N-1"); if a task exposes no usable ``id``, we fall back to positional
    index. tau2 retail/airline task ids are contiguous "0".."N-1" == list index,
    so an id list maps directly, but the id-first match keeps this correct even
    if a domain's ids are ever non-contiguous.

    Args:
        all_tasks: Full task list for the domain (from registry.get_tasks_loader).
        requested_ids: Task ids to select (ints or strings).
        domain: Domain name, for clearer error messages.

    Returns:
        Tasks in the same order as ``requested_ids``.

    Raises:
        ValueError: if any requested id can't be resolved (out of range / absent),
            or if the number of selected tasks != len(requested_ids).
    """
    id_to_task: dict[str, Any] = {}
    for task in all_tasks:
        tid = getattr(task, "id", None)
        if tid is not None:
            id_to_task[str(tid)] = task

    selected: list[Any] = []
    missing: list[Any] = []
    for rid in requested_ids:
        key = str(rid)
        if key in id_to_task:
            selected.append(id_to_task[key])
            continue
        # Fall back to positional index when the id isn't a known task.id.
        try:
            idx = int(rid)
        except (TypeError, ValueError):
            idx = None
        if idx is not None and 0 <= idx < len(all_tasks):
            selected.append(all_tasks[idx])
        else:
            missing.append(rid)

    _tag = f" [{domain}]" if domain else ""
    if missing:
        raise ValueError(
            f"tau2bench{_tag}: {len(missing)} requested task id(s) not found "
            f"(out of range or absent from the task list): {missing}. "
            f"Domain has {len(all_tasks)} tasks (ids 0..{len(all_tasks) - 1})."
        )
    if len(selected) != len(requested_ids):
        raise ValueError(
            f"tau2bench{_tag}: selected {len(selected)} tasks but "
            f"{len(requested_ids)} ids were requested."
        )
    return selected


def convert_tau2_tools(tau2_tools: list[Any]) -> list[dict[str, Any]]:
    """
    Convert tau2 Tool objects to our flat tool description format.

    tau2 exposes tools via `tool.openai_schema` which returns the nested OpenAI format:
        {"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}

    We convert to our flat format:
        {"type": "function", "name": ..., "description": ..., "parameters": ...}

    Also appends the `respond_user` tool so the agent always produces tool calls.

    Args:
        tau2_tools: List of tau2 Tool objects (from info["tools"] after env.reset()).

    Returns:
        List of flat tool description dicts, with `respond_user` appended.
    """
    flat_tools: list[dict[str, Any]] = []
    for tool in tau2_tools:
        schema = tool.openai_schema
        fn = schema.get("function", {})
        flat_tool = {
            "type": "function",
            "name": fn.get("name", ""),
            "description": fn.get("description", ""),
            "parameters": fn.get("parameters", {}),
        }
        flat_tools.append(flat_tool)
    # Append respond_user so the agent always produces structured tool calls
    flat_tools.append(RESPOND_USER_TOOL)
    return flat_tools


def parse_observation(obs_str: str) -> str:
    """
    Parse a tau2 observation string to extract the meaningful content.

    tau2 formats observations as "role: content" lines (e.g. "user: Hello").
    This strips known role prefixes so we get just the content.

    NOTE: This strips role prefixes greedily — if the actual content itself starts
    with one of the known prefixes (e.g. "user: user: please help" or a line like
    "system: reboot required" in a tool output), the prefix will be incorrectly
    stripped. In practice this is rare since tau2 observations are structured, but
    be aware of this limitation if debugging unexpected content truncation.

    Args:
        obs_str: Raw observation string from tau2's step() or reset().

    Returns:
        Cleaned observation content with role prefixes removed.
    """
    if not obs_str:
        return ""

    lines = obs_str.strip().split("\n")
    cleaned_lines: list[str] = []
    for line in lines:
        # Strip known role prefixes
        for prefix in ["user: ", "tool: ", "assistant: ", "system: "]:
            if line.startswith(prefix):
                line = line[len(prefix) :]
                break
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines)
