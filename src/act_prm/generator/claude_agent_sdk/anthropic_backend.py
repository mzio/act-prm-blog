"""
Raw Anthropic SDK backend for the Claude generators.

Used instead of the bundled Claude Code CLI (``claude_agent_sdk``) when talking to an
Anthropic-Messages endpoint whose model id Claude Code's client-side model gate
rejects -- e.g. Meta's Llama API Anthropic passthrough with a ``...-genai`` model
name (Claude Code: "the selected model ... may not exist or you may not have access").

The raw Anthropic SDK sends the model straight to ``/v1/messages`` (exactly like a
plain ``curl``), so the passthrough's ``claude-4-6-sonnet-genai`` works. The Anthropic
API is stateless, so this maps cleanly onto the ``query()`` path (each call sends the
full flattened prompt) -- which is the full Show-Don't-Tell implementation
(retrieval + reflection + judge). The persistent-client optimization does not apply.

Selection: active when ``STRL_CLAUDE_BACKEND=anthropic_api`` (auto-set by
``utils/claude_auth.load_dotenv_auth()`` when a ``LLAMA_API_KEY`` / passthrough base
URL is configured; can also be set explicitly in ``.env``). Auth + base URL come from
the environment (``ANTHROPIC_BASE_URL`` + ``ANTHROPIC_AUTH_TOKEN`` / ``ANTHROPIC_API_KEY``),
the same vars ``load_dotenv_auth()`` sets.
"""

from __future__ import annotations

import logging
import os
from types import SimpleNamespace

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # import only for annotations -- see _get_client() below
    from anthropic import AsyncAnthropic

from .types import ClaudeAgentResponse

logger = logging.getLogger(__name__)

BACKEND_ENV_VAR = "STRL_CLAUDE_BACKEND"
ANTHROPIC_BACKEND = "anthropic_api"
# Passthrough streaming is incompatible with the Bedrock/Vertex backends behind it;
# keep requests non-streaming. Modest default cap when the caller passes no max_tokens.
_DEFAULT_MAX_TOKENS = 8192
# Default auxiliary (judge / compactor) model on the raw Anthropic passthrough, where a
# standard id like claude-haiku-4-5 404s. Override per-run with STRL_GRADER_DEFAULT.
_PASSTHROUGH_AUX_DEFAULT = "claude-4-6-sonnet-genai"

# Lazily constructed. `anthropic` is an OPTIONAL dependency: it is needed only for this
# raw-SDK backend (the Llama passthrough, whose model ids the bundled Claude Code CLI
# rejects). The default OAuth path drives the CLI and never touches it, so importing it
# at module scope would make the whole claude_agent_sdk generator unimportable without it.
_client: "AsyncAnthropic | None" = None


def use_anthropic_backend() -> bool:
    """True when the raw Anthropic SDK backend is selected (vs. the Claude Code CLI)."""
    return os.environ.get(BACKEND_ENV_VAR) == ANTHROPIC_BACKEND


def resolve_backend_model(model: str, *, override_env: str = "STRL_GRADER_MODEL") -> str:
    """
    Resolve an auxiliary (non-policy) Claude model for the active backend.

    Auxiliary Claude calls -- LLM-judge graders (ClawBench, WebJudge), the context
    compactor, etc. -- go through the same ``sample_query_response`` seam as the policy, so
    on the raw Anthropic backend (``use_anthropic_backend()`` -- e.g. Meta's Llama
    passthrough) their model must be a passthrough ``*-genai`` id; a standard id like
    ``claude-haiku-4-5`` would 404.

    ``override_env`` (default ``STRL_GRADER_MODEL``) overrides the configured model -- set it
    in ``.env`` alongside ``LLAMA_API_KEY`` to point all auxiliary calls at a model your
    entitlement serves. Otherwise, when the raw Anthropic backend is active and the configured
    model is a non-``*-genai`` Claude id (which would 404 on the passthrough), we auto-substitute
    a passthrough default (``STRL_GRADER_DEFAULT``, else ``claude-4-6-sonnet-genai``). On a
    personal ``ANTHROPIC_API_KEY`` (no passthrough) the configured model is returned unchanged.
    """
    override = os.environ.get(override_env)
    if override:
        return override
    if use_anthropic_backend() and str(model).startswith("claude") and "-genai" not in str(model):
        fallback = os.environ.get("STRL_GRADER_DEFAULT", _PASSTHROUGH_AUX_DEFAULT)
        logger.info(
            "Auxiliary Claude model %r isn't a passthrough '*-genai' id and the raw Anthropic "
            "backend is active; using %r instead (override with %s or STRL_GRADER_DEFAULT).",
            model,
            fallback,
            override_env,
        )
        return fallback
    return model


def _get_client() -> "AsyncAnthropic":
    """Cached AsyncAnthropic; reads base URL + auth from the env (set by load_dotenv_auth)."""
    global _client
    if _client is None:
        try:
            from anthropic import AsyncAnthropic
        except ImportError as e:  # pragma: no cover - depends on the install extra
            raise ImportError(
                "The raw Anthropic backend needs the optional `anthropic` package "
                f"(STRL_CLAUDE_BACKEND={ANTHROPIC_BACKEND}). Install it, or unset "
                "STRL_CLAUDE_BACKEND / ANTHROPIC_BASE_URL to use the Claude Code CLI path."
            ) from e
        _client = AsyncAnthropic()
    return _client


async def anthropic_messages_response(
    prompt_text: str,
    *,
    model: str,
    system_prompt: str | None = None,
    max_tokens: int | None = None,
    timeout: float | None = None,
) -> ClaudeAgentResponse:
    """
    One-shot Anthropic Messages call returning a ``ClaudeAgentResponse``.

    ``ClaudeAgentResponse`` consumers only touch ``.assistant_messages[*].content``
    (a list of content blocks); Anthropic's blocks already satisfy the duck-typed
    block checks in ``types.py`` (``.text`` / ``.thinking`` / ``.name``+``.input``+``.id``),
    so we wrap the message content directly.
    """
    if not model:
        raise ValueError("anthropic_messages_response requires a non-empty `model`.")

    client = _get_client()
    if timeout is not None:
        client = client.with_options(timeout=timeout)

    create_kwargs: dict = {
        "model": model,
        "max_tokens": max_tokens or _DEFAULT_MAX_TOKENS,
        "messages": [{"role": "user", "content": prompt_text}],
    }
    if system_prompt:
        create_kwargs["system"] = system_prompt

    msg = await client.messages.create(**create_kwargs)
    return _response_from_message(msg)


def _response_from_message(msg) -> ClaudeAgentResponse:
    """Wrap an Anthropic ``Message`` into a ``ClaudeAgentResponse`` (duck-typed blocks)."""
    usage = {
        "input_tokens": int(getattr(msg.usage, "input_tokens", 0) or 0),
        "output_tokens": int(getattr(msg.usage, "output_tokens", 0) or 0),
    }
    return ClaudeAgentResponse(
        assistant_messages=[SimpleNamespace(content=list(msg.content))],
        result=None,
        usage=usage,
        cost=0.0,
    )


def to_anthropic_messages(messages: list[dict]) -> tuple[str | None, list[dict]]:
    """
    Convert STRL ``[{"role", "content"}, ...]`` messages into ``(system, messages)`` for
    a structured Anthropic Messages call.

    Produces a valid Anthropic array: ``system`` extracted; roles mapped (``assistant``
    stays, everything else -- user / tool / function_call_output -- becomes ``user``);
    starts with a user turn; and never ends on an assistant turn -- a trailing assistant
    message (e.g. an injected reflection) would be treated as an assistant *prefill* and
    rejected by current models, so it is re-tagged to ``user``. (Consecutive same-role
    messages are allowed; the API merges them.)
    """
    system_parts: list[str] = []
    out: list[dict] = []
    for m in messages:
        content = m.get("content", "")
        if not content:
            continue
        role = m.get("role")
        if role == "system":
            system_parts.append(content if isinstance(content, str) else str(content))
            continue
        out.append({"role": "assistant" if role == "assistant" else "user", "content": content})

    if out and out[0]["role"] == "assistant":
        out.insert(0, {"role": "user", "content": "(continuing the task)"})
    if out and out[-1]["role"] == "assistant":
        out[-1] = {"role": "user", "content": out[-1]["content"]}
    if not out:
        out = [{"role": "user", "content": (system_parts and "\n\n".join(system_parts)) or "Continue."}]
    return ("\n\n".join(system_parts) or None), out


async def anthropic_structured_response(
    messages: list[dict],
    *,
    model: str,
    system_prompt: str | None = None,
    max_tokens: int | None = None,
    timeout: float | None = None,
) -> ClaudeAgentResponse:
    """
    Structured multi-turn Anthropic Messages call returning a ``ClaudeAgentResponse``.

    Unlike :func:`anthropic_messages_response` (one flattened user message), this sends
    the conversation as a proper Anthropic ``messages`` array with ``system`` separate --
    the idiomatic shape for the API (better role attribution + cacheable prefix).
    """
    if not model:
        raise ValueError("anthropic_structured_response requires a non-empty `model`.")
    system, anthropic_messages = to_anthropic_messages(messages)
    if system_prompt:  # an explicit system_prompt prepends to any extracted system text
        system = f"{system_prompt}\n\n{system}" if system else system_prompt

    client = _get_client()
    if timeout is not None:
        client = client.with_options(timeout=timeout)
    create_kwargs: dict = {
        "model": model,
        "max_tokens": max_tokens or _DEFAULT_MAX_TOKENS,
        "messages": anthropic_messages,
    }
    if system:
        create_kwargs["system"] = system
    msg = await client.messages.create(**create_kwargs)
    return _response_from_message(msg)
