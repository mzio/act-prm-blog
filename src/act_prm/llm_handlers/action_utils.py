"""
Helper functions for processing actions from HuggingFace Transformer and Tinker LLM handlers
"""

import json
import re
from json import JSONDecodeError
from typing import Any

from rich import print as rich_print

from .types import ActionFromLLM

# Qwen3.5/3.6 XML tool-call syntax. Mirrors tinker_cookbook/renderers/qwen3_5.py
# (_FUNCTION_BLOCK_RE / _PARAM_BLOCK_RE) so we stay in lockstep with the renderer
# the trainer already uses to encode messages going *into* the model.
_QWEN_XML_FUNCTION_RE = re.compile(
    r"^\s*<function=(?P<name>[^>\n]+)>\s*(?P<body>.*?)\s*</function>\s*$",
    re.DOTALL,
)
_QWEN_XML_PARAM_RE = re.compile(
    r"<parameter=(?P<name>[^>\n]+)>\s*(?P<value>.*?)\s*</parameter>",
    re.DOTALL,
)


def _parse_qwen3_xml_body(body: str) -> dict[str, Any] | None:
    """Parse the inside of a <tool_call>...</tool_call> block when it
    contains Qwen3.5/3.6 XML rather than JSON.

    Returns the canonical {"name": str, "arguments": {...}} dict on
    success, or None if the body doesn't match (caller should then fall
    back to the JSON path / invalid_tool_call branch).
    """
    match = _QWEN_XML_FUNCTION_RE.match(body)
    if not match:
        return None
    name = match.group("name").strip()
    if not name:
        return None
    inner = match.group("body")
    arguments: dict[str, Any] = {}
    pos = 0
    for param in _QWEN_XML_PARAM_RE.finditer(inner):
        # Reject stray text between <parameter=...> blocks: this matches
        # tinker_cookbook's strictness so a malformed call falls through
        # to invalid_tool_call rather than silently dropping content.
        if inner[pos : param.start()].strip():
            return None
        pname = param.group("name").strip()
        if not pname:
            return None
        pvalue_text = param.group("value").strip("\n")
        # Each <parameter> body is a JSON-encoded scalar/array/object, but
        # Qwen3.5 sometimes emits raw strings without quoting. Try JSON
        # first, fall back to raw string.
        try:
            pvalue: Any = json.loads(pvalue_text)
        except JSONDecodeError:
            pvalue = pvalue_text
        arguments[pname] = pvalue
        pos = param.end()
    if inner[pos:].strip():
        return None
    return {"name": name, "arguments": arguments}


def get_actions(
    response: list[dict[str, Any]],
    tool_call_argname: str = "arguments",
    **tool_call_parse_kwargs: Any,
) -> list[ActionFromLLM]:
    """
    Parse chat response into list of actions, where
    response is a (singleton) list: [{"role": "assistant", "content": <response_text>}].
    """
    action_list = []
    # Split thoughts and tool_calls into different messages
    try:
        response = get_messages_from_text(
            response[0]["content"],
            tool_call_argname=tool_call_argname,
            **tool_call_parse_kwargs,
        )
    except Exception as e:
        rich_print(f"[red]Error in get_messages_from_text: {e}[/red]")
        breakpoint()

    for message in response:
        if message.get("tool_calls", None) is not None:
            for tool_call in message["tool_calls"]:
                output = tool_call["function"]
                # name = output.get("name", "invalid_tool_call")
                name = output.get("name", None)
                arguments = output.get(tool_call_argname, {})

                # Error and edge-case handling
                if name is None:
                    name = json.dumps(output)
                elif not isinstance(name, str):
                    # name = "invalid_tool_call"
                    name = str(name)
                if not isinstance(arguments, dict):
                    arguments = {"arguments": json.dumps(arguments)}
                # Carry the parse diagnosis onto the action (it's otherwise only in
                # `text`) so the env can surface WHY a tool call was invalid instead of
                # routing the `invalid_tool_call` sentinel through tool-availability checks.
                if output.get("parse_error") is not None:
                    arguments = {**arguments, "parse_error": output["parse_error"]}
                text_repr = json.dumps(output)
                action_list.append(
                    ActionFromLLM(
                        role="assistant",
                        type="function_call",
                        text=text_repr,
                        call_id=None,
                        name=name,
                        arguments=arguments,
                    )
                )
        else:
            # Parse as regular message
            action_list.append(
                ActionFromLLM(
                    role="assistant",
                    type="message",
                    text=message["content"],
                    call_id=None,
                    name=None,
                    arguments=None,
                )
            )
    return action_list


def get_messages_from_text(
    text: str,
    tool_call_bos: str = "<tool_call>",
    tool_call_eos: str = "</tool_call>",
    tool_call_argname: str = "arguments",
) -> list[dict[str, Any]]:
    """
    Convert text to LLM chat messages
    """
    messages = []
    parse_error: str | None = None
    tool_call: Any = None  # bound only on the valid path; guarded by valid_tool_call below
    try:
        tool_call_str = text.split(tool_call_bos)[-1].split(tool_call_eos)[0]
        # Qwen3.5/3.6 XML format: <function=name><parameter=p>v</parameter></function>
        # Auto-detect by sniffing for the <function= sentinel; the wrapper
        # boundary tokens are identical to Qwen3, so only the inner body shape
        # disambiguates. If XML parses cleanly we synthesize the canonical
        # {"name", "arguments"} dict; if not we fall through to the JSON
        # path (and ultimately invalid_tool_call) like before.
        _stripped = tool_call_str.strip()
        if _stripped.startswith("<function="):
            xml_parsed = _parse_qwen3_xml_body(_stripped)
            if xml_parsed is None:
                raise JSONDecodeError("Malformed Qwen3.5 XML tool call", _stripped, 0)
            tool_call = xml_parsed
        else:
            # strict=False permits literal control characters (newlines,
            # tabs) inside JSON string values. Models routinely emit
            # multi-sentence args -- e.g. a `done` answer with a raw newline
            # -- which strict JSON rejects; without this they'd parse as
            # invalid_tool_call and a terminal `done` would never register,
            # looping the episode to max_turns.
            try:
                tool_call = json.loads(tool_call_str, strict=False)
            except JSONDecodeError:
                # Recover a COMPLETE JSON object followed by trailing junk: models
                # often close a tool call with the wrong tag (e.g. </invoke>,
                # </function_results> from the Claude/Qwen function-call syntax) instead
                # of </tool_call>, leaving valid JSON + stray text in the body. raw_decode
                # parses just the leading object and ignores the rest; genuine truncation
                # (incomplete JSON) still raises and falls through to invalid_tool_call.
                _brace = tool_call_str.find("{")
                if _brace < 0:
                    raise
                tool_call, _ = json.JSONDecoder(strict=False).raw_decode(tool_call_str[_brace:])
        valid_tool_call = True
    except JSONDecodeError as e:
        valid_tool_call = False
        # Most useful diagnosis first: if there's an opening <tool_call>
        # but no closing </tool_call>, the model almost certainly hit
        # max_tokens mid-write. Tell the model that directly so it can
        # adapt (be more concise / split the call).
        if tool_call_bos in text and tool_call_eos not in text:
            parse_error = (
                "tool call body appears truncated (missing closing "
                f"`{tool_call_eos}`). The model likely hit its max_tokens "
                "budget mid-write. Underlying parser error: "
                f"JSONDecodeError: {e}"
            )
        else:
            parse_error = f"JSONDecodeError: {e}"

    if valid_tool_call:
        if isinstance(tool_call, str):
            valid_tool_call = False
        else:
            try:
                assert tool_call.get("name", None) is not None
                assert tool_call.get(tool_call_argname, None) is not None
            except AssertionError:
                if tool_call.get("arguments", None) is not None:
                    tool_call_argname = "arguments"
                else:
                    valid_tool_call = False

    # Convert any text before first tool call to regular message
    message = text.split(tool_call_bos)[0].strip()
    if len(message) > 0 or tool_call_bos not in text:  # 2nd case handles empty text
        messages.append({"role": "assistant", "content": message})

    if valid_tool_call:
        messages.append(
            {
                "role": "assistant",
                "tool_calls": [{"type": "function", "function": tool_call}],
            }
        )
    elif tool_call_bos in text:  # Invalid tool call
        try:
            _invalid_tool_call_text = text.split(tool_call_bos)[-1].strip()
            assert _invalid_tool_call_text != "", "Invalid tool call"
            try:
                _invalid_tool_call_text = _invalid_tool_call_text.split(tool_call_eos)[0].strip()
                assert _invalid_tool_call_text != "", "Invalid tool call"
            except AssertionError:
                pass
        except AssertionError:
            _invalid_tool_call_text = text.strip()

        _invalid_tool_call = {
            "name": "invalid_tool_call",
            "arguments": _invalid_tool_call_text,
            "parse_error": parse_error or "Tool call body could not be parsed",
        }
        messages.append(
            {
                "role": "assistant",
                "tool_calls": [{"type": "function", "function": _invalid_tool_call}],
            }
        )
    return messages
