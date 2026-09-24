"""
Adapter that lets the Claude Agent SDK generator run under the normal act-prm trainer.

Collects FULL thought+action expert trajectories ourselves, as an alternative to the
logged GPT-5-mini corpora. Those carry a thought on only 50-64% of turns (insurance 64.1%,
retail 49.9%, airline 53.8%), which is why `--require_thought` has to drop the rest and
why `expert_thoughts_all` ends up trained on a biased half of the data. A Claude teacher
emits ThinkingBlocks alongside ToolUseBlocks, so both halves are captured at the source
instead of inferred by the Act-PRM E-step.

Three impedance mismatches with `trainer/train.py`, all handled here so the files copied
from `strl` stay byte-comparable to upstream and can be re-synced:

1. ``train.py:169`` constructs every generator with ``llm=`` and ``enable_thinking=``.
   The Claude generator has no local policy and no chat-template thinking toggle, so it
   would TypeError. Both are accepted and ignored (``llm`` is kept only so the attribute
   exists for anything that introspects it).
2. ``train.py:209`` calls ``do_group_rollout`` SYNCHRONOUSLY, but the Claude version is a
   coroutine. Driven here on a dedicated, persistent event loop -- persistent because the
   SDK's subprocess transport keeps state across calls, and ``asyncio.run()`` would tear
   that down (and its executor) after every task.
3. ``num_return_sequences`` is the group size. For teacher collection this is "how many
   independent attempts at this task", which is the knob for getting >=1 SUCCESSFUL
   trajectory per sample -- see ``scripts/collect_claude_trajectories.sh``.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from .query import QueryClaudeAgentSDKGenerator

logger = logging.getLogger(__name__)


class ClaudeTeacherGenerator(QueryClaudeAgentSDKGenerator):
    """Sync-callable Claude Agent SDK generator, wired for the act-prm trainer."""

    # ClaudeAgentOptions fields that the base ctor does not expose as named parameters.
    # Listed explicitly (rather than swallowing **kwargs) so a typo in a yaml still raises
    # instead of being silently dropped.
    PASSTHROUGH_OPTIONS = ("thinking", "system_prompt", "setting_sources", "cwd")

    def __init__(
        self,
        *args: Any,
        llm: Any = None,
        enable_thinking: bool = False,
        **kwargs: Any,
    ) -> None:
        # The trainer owns these; this generator has no local policy to apply them to.
        self.llm = llm
        self.enable_thinking = enable_thinking
        # Pull passthrough options out before super(): the base ctor takes only
        # model/permission_mode/effort/max_agent_turns and TypeErrors on anything else, so
        # e.g. `thinking: {type: disabled}` in the yaml would kill the run at startup.
        passthrough = {k: kwargs.pop(k) for k in self.PASSTHROUGH_OPTIONS if k in kwargs}
        super().__init__(*args, **kwargs)
        for k, v in passthrough.items():
            # OmegaConf hands us DictConfig/ListConfig; the SDK needs plain containers.
            try:
                from omegaconf import OmegaConf

                if OmegaConf.is_config(v):
                    v = OmegaConf.to_container(v, resolve=True)
            except ImportError:
                pass
            self.claude_agent_option_kwargs[k] = v
        if passthrough:
            logger.info("ClaudeAgentOptions passthrough: %s", sorted(passthrough))
        self._loop: asyncio.AbstractEventLoop | None = None

    def _get_loop(self) -> asyncio.AbstractEventLoop:
        """Persistent loop shared by every rollout in this process."""
        if self._loop is None or self._loop.is_closed():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
        return self._loop

    def _filter_rollout_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Keep only what ``do_single_rollout`` accepts.

        ``train.py:209`` passes ``pbar_position`` (for the HF generators' nested progress
        bars), which this generator's ``do_single_rollout`` does not take and which has no
        ``**kwargs`` to absorb it -- every rollout TypeErrors. Filtering here rather than
        editing the copied ``query.py`` keeps that file re-syncable with upstream, and
        keeps this robust if train.py grows another display-only kwarg.
        """
        import inspect

        params = inspect.signature(self.do_single_rollout).parameters
        if any(p.kind == p.VAR_KEYWORD for p in params.values()):
            return kwargs
        kept = {k: v for k, v in kwargs.items() if k in params}
        dropped = set(kwargs) - set(kept) - {"num_return_sequences"}
        if dropped:
            logger.debug("dropping kwargs unsupported by do_single_rollout: %s", sorted(dropped))
        # hf_tokenizer defaults to None but the rollout needs it for prompt rendering.
        kept.setdefault("hf_tokenizer", self.hf_tokenizer)
        # temperature stays None when unset: sampling temperature belongs to a local
        # policy and the Claude teacher has none, so EpisodeStep.temperature is nullable
        # (replay_buffer/types.py) rather than defaulted to an invented 1.0.
        return kept

    def do_group_rollout(self, **kwargs: Any) -> dict[str, list[Any]]:  # type: ignore[override]
        """Sync wrapper over the async group rollout.

        Returns the same ``{"policy": [TrajectoryGroup, ...]}`` shape as the HF
        generators. A failure here would otherwise abort the whole collection run, so a
        raising task degrades to an empty group -- the caller's per-sample loop moves on
        and the sample simply contributes no trajectories.
        """
        n = kwargs.pop("num_return_sequences", 1)
        coro = super().do_group_rollout(
            num_return_sequences=n, **self._filter_rollout_kwargs(kwargs)
        )
        try:
            return self._get_loop().run_until_complete(coro)
        except Exception as e:  # noqa: BLE001 - one bad task must not kill the run
            logger.warning(
                "Claude group rollout failed for sample_id=%s: %s: %s",
                kwargs.get("sample_id"), type(e).__name__, e,
            )
            return {"policy": []}
