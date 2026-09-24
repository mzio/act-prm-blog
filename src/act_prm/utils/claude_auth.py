"""
Self-contained Claude Agent SDK auth from ``.env``.

The Claude Agent SDK spawns the bundled Claude Code CLI as a subprocess whose
environment is ``{**os.environ, **ClaudeAgentOptions.env}`` (options.env wins; see
``claude_agent_sdk/_internal/transport/subprocess_cli.py``). When the parent process
inherits a managed / launcher environment -- e.g. Meta's Claude Code launcher exports
``CLAUDE_CODE_USE_VERTEX=1`` + ``ANTHROPIC_VERTEX_*`` + ``ANTHROPIC_CUSTOM_HEADERS`` --
those vars get inherited by the subprocess and hijack its auth routing, so any key you
set in ``.env`` is ignored and you get ``authentication_failed`` ("Invalid API key").

``load_dotenv_auth()`` makes auth come ONLY from ``.env``: it loads ``.env`` (overriding
inherited values), scrubs the inherited gateway / launcher vars from ``os.environ``
(unless you deliberately set them in ``.env``), and returns an explicit auth dict to hand
to ``ClaudeAgentOptions(env=...)``.

Put ONE of these in ``.env`` (see ``.env.example``):
  - ``ANTHROPIC_API_KEY=sk-ant-...``                                personal Anthropic account
  - ``LLAMA_API_KEY=LLM|...`` (+ optional ``METAGEN_KEY=mg-api-``)  Meta Llama Anthropic passthrough,
        auto-wired to ANTHROPIC_AUTH_TOKEN + the passthrough base URL (the LLM|... token must be
        linked to the mg-api-... entitlement in the Llama API keys portal -- a one-time portal step)
  - ``ANTHROPIC_BASE_URL=...`` + ``ANTHROPIC_AUTH_TOKEN=LLM|...``   any token-based gateway (Bearer), explicit
  - ``CLAUDE_CODE_OAUTH_TOKEN=...``                                 personal Claude Code login (``claude setup-token``)
"""

from __future__ import annotations

import logging
import os

from dotenv import dotenv_values, load_dotenv

logger = logging.getLogger(__name__)

# Launcher / managed Claude Code vars that hijack the subprocess's auth routing if
# inherited. Scrubbed from os.environ unless explicitly set in .env (so you can still
# opt into the gateway via .env if you really want to).
GATEWAY_ENV_VARS: tuple[str, ...] = (
    "CLAUDE_CODE_USE_VERTEX",
    "CLAUDE_CODE_USE_BEDROCK",
    "CLAUDE_CODE_SKIP_VERTEX_AUTH",
    "ANTHROPIC_VERTEX_BASE_URL",
    "ANTHROPIC_VERTEX_PROJECT_ID",
    "ANTHROPIC_CUSTOM_HEADERS",
    "CLAUDE_CODE_OAUTH_TOKEN",
)

# Auth vars read from .env and forwarded explicitly to the SDK subprocess.
# CLAUDE_CODE_OAUTH_TOKEN is dual-listed (also in GATEWAY_ENV_VARS): an *inherited*
# one is scrubbed, but one you set in .env is kept, forwarded, and counted as auth.
AUTH_ENV_VARS: tuple[str, ...] = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_BASE_URL",
    "CLAUDE_CODE_OAUTH_TOKEN",  # personal Claude Code login (`claude setup-token`)
)

# Meta's Llama API Anthropic-passthrough base URL. When a LLAMA_API_KEY (LLM|...) is set
# and no explicit Anthropic auth is, we auto-wire "Path B" against this endpoint.
LLAMA_PASSTHROUGH_BASE_URL = "https://api.llama.com/experimental/passthrough/anthropic"

_cached_auth_env: dict[str, str] | None = None


def load_dotenv_auth(dotenv_path: str = ".env", *, force: bool = False) -> dict[str, str]:
    """
    Load ``.env`` and return an explicit auth env dict for ``ClaudeAgentOptions(env=...)``.

    Side effect: scrubs inherited gateway / launcher vars from ``os.environ`` so the SDK
    subprocess authenticates ONLY from ``.env``. Idempotent and memoized after the first
    call (pass ``force=True`` to re-read the file and re-scrub).

    Returns an empty dict when ``.env`` sets no ``ANTHROPIC_*`` auth -- in that case the
    SDK falls back to its own credential resolution (managed apiKeyHelper / OAuth login).
    """
    global _cached_auth_env
    if _cached_auth_env is not None and not force:
        return dict(_cached_auth_env)

    file_keys = set(dotenv_values(dotenv_path)) if os.path.exists(dotenv_path) else set()
    load_dotenv(dotenv_path, override=True)

    # If .env supplies NO auth of its own, scrubbing would strip the inherited launcher
    # credentials and leave the subprocess with nothing -- on a managed devserver
    # (CLAUDE_CODE_USE_VERTEX + ANTHROPIC_VERTEX_* + an apiKeyHelper) those vars ARE the
    # working auth. Measured 2026-09-23 on this box: a one-line query returns "PONG" with
    # the vars inherited and HANGS until timeout once they are scrubbed -- a stall, not a
    # clean 401, so a collection run would just wedge. The scrub exists to stop launcher
    # vars from overriding auth you put in .env; with no .env auth there is nothing to
    # protect, so leave the environment alone and let the SDK resolve credentials itself.
    env_provides_auth = bool(file_keys & set(AUTH_ENV_VARS)) or "LLAMA_API_KEY" in file_keys
    if not env_provides_auth:
        logger.info(
            "Claude auth: %s defines no ANTHROPIC_*/LLAMA_API_KEY auth -- leaving the "
            "inherited environment intact so the SDK can use the ambient login.",
            dotenv_path,
        )
        _cached_auth_env = {}
        return {}

    # Scrub inherited gateway/launcher vars AND any auth vars not defined in .env, so
    # the subprocess sees auth ONLY from .env -- no stale shell ANTHROPIC_* leaking in
    # via the inherited environment, and no api-key/auth-token conflicts.
    scrubbed: list[str] = []
    for var in dict.fromkeys((*GATEWAY_ENV_VARS, *AUTH_ENV_VARS)):  # dedupe, keep order
        if var not in file_keys and var in os.environ:
            os.environ.pop(var, None)
            scrubbed.append(var)

    # Convenience (Path B): a Llama API key (LLM|...) is the wire credential for Meta's
    # Llama API Anthropic passthrough. If one is set and no explicit Anthropic auth is,
    # auto-wire it -- use it as the Bearer token and default the base URL to the passthrough.
    # The mg-api MetaGen key (METAGEN_KEY) is the entitlement the token is linked to in the
    # portal; it is NOT sent on the wire, so it isn't forwarded here.
    llama_token = os.environ.get("LLAMA_API_KEY")
    if llama_token and not os.environ.get("ANTHROPIC_AUTH_TOKEN") and not os.environ.get("ANTHROPIC_API_KEY"):
        os.environ["ANTHROPIC_AUTH_TOKEN"] = llama_token
        os.environ.setdefault("ANTHROPIC_BASE_URL", LLAMA_PASSTHROUGH_BASE_URL)
        # Drop a coexisting Claude Code OAuth login token so the passthrough Bearer is the
        # one, unambiguous credential (otherwise the bundled CLI may prefer the OAuth token
        # and 401 against the passthrough).
        os.environ.pop("CLAUDE_CODE_OAUTH_TOKEN", None)
        logger.info(
            "Claude auth: wired Path B from LLAMA_API_KEY (Bearer + %s); dropped any OAuth token.",
            os.environ["ANTHROPIC_BASE_URL"],
        )

    # Endpoints whose model ids the bundled Claude Code CLI rejects (e.g. the Llama
    # passthrough, which only serves `...-genai` names) must use the raw Anthropic SDK
    # backend. Auto-select it when ANTHROPIC_BASE_URL points at the passthrough, unless
    # STRL_CLAUDE_BACKEND was pinned explicitly in .env. See generator/.../anthropic_backend.py.
    if "STRL_CLAUDE_BACKEND" not in file_keys and "api.llama.com" in os.environ.get("ANTHROPIC_BASE_URL", ""):
        os.environ["STRL_CLAUDE_BACKEND"] = "anthropic_api"
        logger.info("Claude auth: selected raw Anthropic SDK backend (STRL_CLAUDE_BACKEND=anthropic_api).")

    auth_env = {k: os.environ[k] for k in AUTH_ENV_VARS if os.environ.get(k)}

    if auth_env:
        logger.info(
            "Claude auth: using .env keys %s%s",
            sorted(auth_env),
            f"; scrubbed inherited gateway vars {scrubbed}" if scrubbed else "",
        )
    else:
        logger.warning(
            "Claude auth: no ANTHROPIC_* auth keys in %s -- the SDK will fall back to its "
            "default credential resolution (managed apiKeyHelper / OAuth login).%s",
            dotenv_path,
            f" Scrubbed inherited gateway vars {scrubbed}." if scrubbed else "",
        )

    _cached_auth_env = auth_env
    return dict(auth_env)
