"""
litellm custom provider that routes ``claude_agent_sdk/<model>`` model strings
through this project's Claude Agent SDK handler (``ClaudeQueryLLM``) instead of
litellm's built-in Anthropic-API path.

tau2's user simulator / NL-assertions judge call ``litellm.completion(model=...)``
(see ``tau2/utils/llm_utils.py``). By registering this provider and setting
``user_llm: "claude_agent_sdk/claude-haiku-4-5"``, those calls run through the
SAME Claude Agent SDK auth as the agent (Claude Code subscription / the SDK's
own credentials) — no separate ``ANTHROPIC_API_KEY`` needed.

The bridge mirrors how ``ami_bench`` drives its user model: build a
system_prompt + conversation, call ``ClaudeQueryLLM(model=...).sample(...)``,
then ``get_actions(resp)`` for the reply text.

Everything is lazy: ``litellm`` (a tau2 dependency) and the SDK handler are
imported inside ``register()`` / the call path, so importing this module does
not require either to be installed.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
from typing import Any

logger = logging.getLogger(__name__)

PROVIDER = "claude_agent_sdk"

# Agent-loop turn budget for the user-sim / NL-assertion-judge calls. See _get_handler.
_JUDGE_MAX_TURNS = 4


class ClaudeAgentSDKError(RuntimeError):
    """The Claude Agent SDK call failed and produced no text.

    Raised instead of silently returning "" so that a transport/turn-limit failure is
    distinguishable from a judge that genuinely answered with an empty string. tau2
    catches it the same way it catches a JSON error, but the message now names the
    real cause and callers can retry.
    """

# Cache one handler per concrete model name (handler construction is cheap but
# this avoids re-creating it on every turn).
_HANDLERS: dict[str, Any] = {}
_REGISTERED = False


def _get_handler(model: str) -> Any:
    if model not in _HANDLERS:
        from act_prm.llm_handlers.claude_agent_sdk import ClaudeQueryLLM

        # max_turns MUST be > 1. At max_turns=1 the SDK raises "Reached maximum
        # number of turns (1)" whenever the model's first turn is not a final text
        # answer (e.g. it emits a tool_use). ClaudeQueryLLM._query swallows that,
        # returns None -> the reply text is "" -> tau2's json.loads("") raises
        # JSONDecodeError("Expecting value: line 1 column 1") -> the tau2 evaluator
        # is downgraded to reward=0.0. Net effect: a judge that never ran scored
        # identically to a failed episode, on ~10% of airline episodes (concentrated
        # on gift-card tasks 14/23, which ask the judge to verify a numeric total).
        # Extra turns are free on the happy path: verified identical output shape
        # and length at max_turns 1 vs 4.
        _HANDLERS[model] = ClaudeQueryLLM(model=model, max_turns=_JUDGE_MAX_TURNS)
    return _HANDLERS[model]


def _reply_text(handler: Any, response: Any) -> str:
    """Concatenate the message-text from a ClaudeAgentResponse (mirrors ami_bench's
    get_actions(...)[-1].text, but joins all message actions and ignores reasoning)."""
    actions = handler.get_actions(response) if response is not None else []
    texts = [a.text for a in actions if getattr(a, "type", "") == "message" and getattr(a, "text", None)]
    if texts:
        return "\n".join(texts).strip()
    return (getattr(actions[-1], "text", "") if actions else "").strip()


def _complete(model: str, messages: list[dict[str, Any]], max_new_tokens: int | None) -> tuple[str, dict[str, int]]:
    """Run one Claude Agent SDK completion for litellm-style ``messages``.

    Splits system messages into the system_prompt and passes the rest as the
    conversation. Runs the (sync, asyncio.run-based) handler in a worker thread
    when called from inside a running event loop, so it never trips
    'asyncio.run() cannot be called from a running event loop'.
    """
    real_model = model.split("/", 1)[-1]  # strip the "claude_agent_sdk/" prefix if present
    system = "\n\n".join(
        m["content"] for m in messages
        if m.get("role") == "system" and isinstance(m.get("content"), str)
    )
    convo = [
        {"role": m["role"], "content": m["content"]}
        for m in messages
        if m.get("role") != "system" and isinstance(m.get("content"), str)
    ]
    handler = _get_handler(real_model)

    def _do() -> tuple[str, dict[str, int]]:
        responses = handler.sample(
            system_prompt=system or None,
            messages=convo,
            tools=None,
            max_new_tokens=max_new_tokens,
            num_return_sequences=1,
        )
        resp = responses[0] if responses else None
        # ClaudeQueryLLM._query returns None on ANY exception (it prints
        # "ClaudeQueryLLM query error: ..." and swallows it). Previously that became
        # an empty reply string, which the NL-assertion judge turned into a silent
        # reward=0.0. Surface it instead so the caller can retry / mark unscored.
        if resp is None:
            raise ClaudeAgentSDKError(
                f"Claude Agent SDK returned no response for model={real_model!r} "
                "(see the preceding 'ClaudeQueryLLM query error' line for the cause)"
            )
        return _reply_text(handler, resp), (getattr(resp, "usage", None) or {})

    try:
        asyncio.get_running_loop()
        running = True
    except RuntimeError:
        running = False
    if running:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            return ex.submit(_do).result()
    return _do()


def register() -> None:
    """Register the ``claude_agent_sdk`` litellm custom provider (idempotent).

    Call before any tau2 LLM call that uses a ``claude_agent_sdk/...`` model.
    """
    global _REGISTERED
    if _REGISTERED:
        return

    import litellm
    from litellm import CustomLLM

    class ClaudeAgentSDKLLM(CustomLLM):
        def completion(self, *args: Any, **kwargs: Any) -> Any:
            model = kwargs.get("model") or (args[0] if args else "")
            messages = kwargs.get("messages") or []
            model_response = kwargs.get("model_response")
            optional_params = kwargs.get("optional_params") or {}
            max_new_tokens = optional_params.get("max_tokens")
            text, usage = _complete(model, messages, max_new_tokens)
            model_response.choices[0].message.content = text
            model_response.model = model
            # litellm has no pricing map for this custom provider, so tau2's
            # get_response_cost() logs a per-turn "model isn't mapped" ERROR.
            # Register a 0-priced entry (both the bare + provider-qualified name)
            # so the lookup resolves quietly; the agent's own cost is tracked
            # separately by the generator.
            try:
                _zero = {
                    "input_cost_per_token": 0.0,
                    "output_cost_per_token": 0.0,
                    "litellm_provider": PROVIDER,
                    "mode": "chat",
                }
                litellm.register_model({model.split("/", 1)[-1]: _zero, model: _zero})
            except Exception:  # noqa: BLE001 — cost registration is best-effort
                pass
            try:
                from litellm.types.utils import Usage

                inp = int(usage.get("input_tokens", 0) or 0)
                out = int(usage.get("output_tokens", 0) or 0)
                model_response.usage = Usage(prompt_tokens=inp, completion_tokens=out, total_tokens=inp + out)
            except Exception:  # noqa: BLE001 — usage is best-effort
                pass
            return model_response

        async def acompletion(self, *args: Any, **kwargs: Any) -> Any:
            # tau2's user sim uses the sync path; provide async for completeness.
            return await asyncio.to_thread(lambda: self.completion(*args, **kwargs))

    handler = ClaudeAgentSDKLLM()
    provider_map = list(getattr(litellm, "custom_provider_map", None) or [])
    if not any(entry.get("provider") == PROVIDER for entry in provider_map):
        provider_map.append({"provider": PROVIDER, "custom_handler": handler})
    litellm.custom_provider_map = provider_map
    _REGISTERED = True
    logger.info("Registered litellm custom provider '%s/' -> Claude Agent SDK", PROVIDER)
