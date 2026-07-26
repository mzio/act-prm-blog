"""
litellm custom provider that routes ``metagen/<model>`` model strings through the
Meta **Llama-API passthrough** (OpenAI-compatible) instead of litellm's built-in
providers.

tau2's user simulator / NL-assertions judge call ``litellm.completion(model=...)``
(see ``tau2/utils/llm_utils.py``). By registering this provider and setting
``user_llm: "metagen/openai-gpt-5-5-responses"``, those calls go to
``https://api.llama.com/experimental/passthrough/openai/v1/`` with an
``LLM|<id>|<secret>`` Llama-API key from ``LLAMA_API_KEY`` — no litellm provider
config, no fbcode/CIF client, works from the uv venv.

Model-string convention (after stripping the ``metagen/`` prefix):
- ``*-responses`` (e.g. ``openai-gpt-5-5-responses``) -> the Responses API
  (``client.responses.create(input=...)`` -> ``output_text``).
- anything else -> Chat Completions (``client.chat.completions.create``).

Everything is lazy: ``openai`` / ``litellm`` (both tau2 deps) are imported inside
``register()`` / the call path, so importing this module needs neither installed.

Auth note: the passthrough expects an ``LLM|<id>|<secret>`` key (mint at
metagen-llm-api-keys.nest.x2p.facebook.net, linked to an ``mg-api-...`` entitlement
that allowlists the target model). A bare ``mg-api-...`` key returns HTTP 401.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any

logger = logging.getLogger(__name__)

PROVIDER = "metagen"
BASE_URL = "https://api.llama.com/experimental/passthrough/openai/v1/"
API_KEY_ENV_VAR = "LLAMA_API_KEY"

# The passthrough throttles under concurrency by returning 401 (and 429/5xx
# transiently), which the openai SDK won't retry — retry in place.
_RETRY_STATUSES = {401, 429, 500, 502, 503, 504}
_MAX_RETRIES = 4
_RETRY_BACKOFF_S = 1.5

_CLIENT: Any = None
_REGISTERED = False

# Load .env / .env.local so LLAMA_API_KEY is available without exporting it.
try:  # pragma: no cover - optional dependency
    from dotenv import load_dotenv

    load_dotenv(".env.local")
    load_dotenv()
except ImportError:
    pass


def _client() -> Any:
    global _CLIENT
    if _CLIENT is not None:
        return _CLIENT
    api_key = os.environ.get(API_KEY_ENV_VAR)
    if not api_key:
        raise RuntimeError(
            f"No API key for the metagen passthrough: set {API_KEY_ENV_VAR} "
            "(an 'LLM|<id>|<secret>' Llama-API key linked to an mg-api-... entitlement)."
        )
    from openai import OpenAI

    _CLIENT = OpenAI(base_url=BASE_URL, api_key=api_key, timeout=120.0)
    return _CLIENT


def _with_retry(fn: Any) -> Any:
    last_exc: Exception | None = None
    for attempt in range(_MAX_RETRIES):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            status = getattr(e, "status_code", None) or getattr(getattr(e, "response", None), "status_code", None)
            if status not in _RETRY_STATUSES or attempt == _MAX_RETRIES - 1:
                raise
            last_exc = e
            logger.warning("metagen passthrough transient %s; retry %d/%d", status, attempt + 1, _MAX_RETRIES - 1)
            time.sleep(_RETRY_BACKOFF_S * (attempt + 1))
    assert last_exc is not None
    raise last_exc


def _complete(model: str, messages: list[dict[str, Any]], optional_params: dict[str, Any]) -> tuple[str, dict[str, int]]:
    """Run one passthrough completion for litellm-style ``messages``."""
    real_model = model.split("/", 1)[-1]  # strip the "metagen/" prefix
    client = _client()
    max_tokens = optional_params.get("max_tokens")
    temperature = optional_params.get("temperature")

    if real_model.endswith("-responses"):
        convo = [
            {"role": m["role"], "content": m["content"]}
            for m in messages
            if m.get("role") in ("system", "user", "assistant") and isinstance(m.get("content"), str)
        ]
        payload: dict[str, Any] = {"model": real_model, "input": convo}
        if max_tokens is not None:
            payload["max_output_tokens"] = max_tokens
        if temperature is not None:
            payload["temperature"] = temperature
        resp = _with_retry(lambda: client.responses.create(**payload))
        text = getattr(resp, "output_text", "") or ""
    else:
        payload = {"model": real_model, "messages": messages}
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if temperature is not None:
            payload["temperature"] = temperature
        resp = _with_retry(lambda: client.chat.completions.create(**payload))
        text = (resp.choices[0].message.content or "") if resp.choices else ""

    usage = getattr(resp, "usage", None)
    tok = {
        "input_tokens": int(getattr(usage, "prompt_tokens", 0) or getattr(usage, "input_tokens", 0) or 0),
        "output_tokens": int(getattr(usage, "completion_tokens", 0) or getattr(usage, "output_tokens", 0) or 0),
    }
    return text.strip(), tok


def register() -> None:
    """Register the ``metagen`` litellm custom provider (idempotent)."""
    global _REGISTERED
    if _REGISTERED:
        return

    import litellm
    from litellm import CustomLLM

    class MetaGenPassthroughLLM(CustomLLM):
        def completion(self, *args: Any, **kwargs: Any) -> Any:
            model = kwargs.get("model") or (args[0] if args else "")
            messages = kwargs.get("messages") or []
            model_response = kwargs.get("model_response")
            optional_params = kwargs.get("optional_params") or {}
            text, usage = _complete(model, messages, optional_params)
            model_response.choices[0].message.content = text
            model_response.model = model
            # 0-priced entry so tau2's get_response_cost() doesn't log "unmapped model".
            try:
                _zero = {
                    "input_cost_per_token": 0.0,
                    "output_cost_per_token": 0.0,
                    "litellm_provider": PROVIDER,
                    "mode": "chat",
                }
                litellm.register_model({model.split("/", 1)[-1]: _zero, model: _zero})
            except Exception:  # noqa: BLE001
                pass
            try:
                from litellm.types.utils import Usage

                inp, out = usage["input_tokens"], usage["output_tokens"]
                model_response.usage = Usage(prompt_tokens=inp, completion_tokens=out, total_tokens=inp + out)
            except Exception:  # noqa: BLE001
                pass
            return model_response

        async def acompletion(self, *args: Any, **kwargs: Any) -> Any:
            return await asyncio.to_thread(lambda: self.completion(*args, **kwargs))

    handler = MetaGenPassthroughLLM()
    provider_map = list(getattr(litellm, "custom_provider_map", None) or [])
    if not any(entry.get("provider") == PROVIDER for entry in provider_map):
        provider_map.append({"provider": PROVIDER, "custom_handler": handler})
    litellm.custom_provider_map = provider_map
    _REGISTERED = True
    logger.info("Registered litellm custom provider '%s/' -> Llama-API passthrough", PROVIDER)
