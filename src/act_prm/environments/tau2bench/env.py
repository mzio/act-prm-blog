"""
Online tau2-bench environment for step-by-step rollouts.

Wraps tau2's AgentGymEnv to provide a live interactive environment where an
LLM agent interacts with a simulated user and domain-specific tools.
This enables rollout evaluation and RL training on tau2-bench tasks.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os as _os
import sys as _sys
from copy import copy
from typing import Annotated, Any
from unittest.mock import MagicMock as _MagicMock

import numpy as np
from pydantic import ConfigDict, Field, SkipValidation

from act_prm.llm_handlers import ActionFromLLM

from ..base import Environment
from ..types import EnvironmentState, EnvironmentStepResult
from .utils import convert_tau2_tools, parse_observation, select_tasks_by_ids

# --- tau2 load-time setup --------------------------------------------------
# Runs at import time, before the LAZY ``import tau2…`` calls inside the methods
# below (tau2 is never imported at module level, nor by .utils or the strl
# imports above, so this ordering is sufficient).
#
# Stub voice-related modules with MagicMock so tau2.voice (eager-loaded by
# tau2.run / tau2.agent.base.streaming on v1.0.0+) imports without any
# voice-specific packages installed (pyaudio needs system portaudio.h;
# elevenlabs/deepgram/google-cloud are heavy + unused here). MagicMock
# auto-creates any attribute on access, so every ``from <pkg> import <symbol>``
# in the voice chain resolves to a placeholder. We never CALL voice features.
for _stub_name in (
    "pyaudio",
    "elevenlabs", "elevenlabs.client",
    "deepgram",
    "jiwer",
    "pydub",
    "websockets",
    "google.cloud", "google.genai",
    "boto3",
    "rank_bm25",
):
    if _stub_name not in _sys.modules:
        _sys.modules[_stub_name] = _MagicMock()

# tau2 resolves data paths at import time -> set TAU2_DATA_DIR before tau2 imports.
if not _os.getenv("TAU2_DATA_DIR"):
    for _candidate in [
        _os.path.join("data", "tau2", "_repo", "data"),
        _os.path.join(_os.path.dirname(__file__), "..", "..", "..", "..", "data", "tau2", "_repo", "data"),
    ]:
        if _os.path.exists(_candidate):
            _os.environ["TAU2_DATA_DIR"] = _os.path.abspath(_candidate)
            break

logger = logging.getLogger(__name__)


class Tau2BenchState(EnvironmentState):
    """
    State for a tau2-bench episode.

    Holds a reference to the live AgentGymEnv instance (SkipValidation because
    it's not a Pydantic-serializable object), along with task metadata and
    the latest info dict from tau2.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Live tau2 gym environment for this episode (excluded from serialization)
    tau2_gym_env: Annotated[Any, SkipValidation] = Field(exclude=True, repr=False)
    # Current task identifier
    task_id: str
    # Info dict from the last tau2 gym step (contains tools, policy, reward_info)
    tau2_info: Annotated[dict[str, Any], SkipValidation] = Field(default_factory=dict, exclude=True, repr=False)


class Tau2BenchStepResult(EnvironmentStepResult):
    """Result of a step in the tau2-bench environment."""

    state: Tau2BenchState
    reward: float
    done: bool
    truncated: bool
    info: dict[str, Any] | None = None


class Tau2BenchEnv(Environment):
    """
    Online tau2-bench environment.

    Creates a fresh AgentGymEnv per episode, routes parsed LLM actions through
    tau2's orchestrator (which handles user simulation, tool execution, and
    evaluation), and maps results back to our Environment interface.

    Args:
        domain: tau2 domain name (e.g. "airline", "retail", "telecom").
        user_llm: Model name for the simulated user (e.g. "gpt-4.1-2025-04-14").
        user_llm_args: LLM arguments for user simulator (e.g. {"temperature": 0.0}).
        task_split: Optional tau2 task split name ("train", "test", or None for all).
        num_train_tasks: Number of tasks allocated to the train split.
        num_val_tasks: Number of tasks for the validation/eval split (None = use test).
        num_test_tasks: Number of tasks allocated to the test split.
        train_task_ids: Explicit list of tau2 task ids for the train split. When
            provided (non-null), the train split is EXACTLY these ids, overriding the
            count-based split. None (default) keeps count-based behavior.
        eval_task_ids: Explicit list of tau2 task ids for the eval/test split. Same
            semantics as train_task_ids. Used for Stage-3 RL to train on the logged
            tasks and eval on the never-in-logs hold-out.
        max_steps: Maximum steps per episode in tau2's orchestrator.
        max_turns: Maximum LLM turns before we truncate (our-level truncation).
        seed: Random seed.
        split: Active data split ("train", "eval", or "test").
        system_prompt: Base system prompt for the agent.
    """

    # Raw text of the most recent judge response that failed to parse, stashed by the
    # tolerant-JSON shim so _patched_get_reward can persist it to the audit log.
    _last_bad_judge_output: str | None = None

    # How many times to re-run tau2's evaluator when it raises. Judge failures are
    # stochastic (observed ~4-6 of a task's 8 rollouts, not all 8), so a retry
    # recovers most of them instead of manufacturing a false reward=0.0.
    JUDGE_MAX_ATTEMPTS: int = 3

    def __init__(
        self,
        data_path: str | None = None,
        domain: str = "airline",
        user_llm: str = "gpt-4.1-2025-04-14",
        user_llm_args: dict[str, Any] | None = None,
        # Independent model for tau2's NL-assertions judge (defaults to a
        # cheap, separate-quota model so concurrent rollouts don't pile up
        # against the user_llm's TPM ceiling). Pass None to fall back to
        # user_llm when it's non-OpenAI, else "gpt-5-mini".
        nl_assertions_llm: str | None = None,
        env_interface_llm: str | None = None,
        task_split: str | None = None,
        num_train_tasks: int = 80,
        num_val_tasks: int | None = None,
        num_test_tasks: int = 50,
        # Explicit task-id selection (Stage-3 RL). When both lists are non-null,
        # those splits are EXACTLY the given tau2 task ids, overriding the count-based
        # split below: train on the tasks present in the Act-PRM logs, eval on the
        # never-seen complement. Either pass the lists directly, or a task-map json
        # (scripts/map_dataset_to_tau2.py) whose covered_tau2_ids -> train and
        # unseen_tau2_ids -> eval. All null (default) => unchanged count-based split.
        train_task_ids: list[int] | None = None,
        eval_task_ids: list[int] | None = None,
        task_id_map_file: str | None = None,
        max_steps: int = 50,
        # Inherited arguments
        max_turns: int = 30,
        num_tries: int = 1,
        eval_num_tries: int = 1,
        seed: int = 0,
        split: str = "train",
        system_prompt: str = "You are a helpful customer service agent.",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            max_turns=max_turns,
            num_tries=num_tries,
            eval_num_tries=eval_num_tries,
            seed=seed,
            split=split,
            **kwargs,
        )
        # Override TAU2_DATA_DIR from config if provided
        import os

        if data_path and os.path.exists(data_path):
            os.environ["TAU2_DATA_DIR"] = os.path.abspath(data_path)
            logger.info("Set TAU2_DATA_DIR=%s (from config)", os.environ["TAU2_DATA_DIR"])

        self.domain = domain
        self.user_llm = user_llm
        self.user_llm_args = user_llm_args or {"temperature": 0.0}
        # Resolve the NL-assertions model (independent of user_llm so we
        # can offload concurrency pressure from gpt-4.1 onto a separate
        # quota / cheaper model). Fallback chain:
        #   1. explicit nl_assertions_llm kwarg / yaml field
        #   2. user_llm if non-OpenAI (preserves haiku45 yaml behavior)
        #   3. "gpt-5-mini" (cheap default, separate quota from gpt-4.1)
        if nl_assertions_llm is None:
            if not (user_llm.startswith("gpt-") or user_llm.startswith("openai/")):
                nl_assertions_llm = user_llm
            else:
                nl_assertions_llm = "gpt-5-mini"
        if env_interface_llm is None:
            env_interface_llm = nl_assertions_llm
        self.nl_assertions_llm = nl_assertions_llm
        self.env_interface_llm = env_interface_llm

        # Route user-simulator / NL-assertions-judge / env-interface models through a
        # litellm custom provider (tau2 calls litellm.completion). Two backends:
        #   metagen/<model>            -> Llama-API passthrough (needs LLAMA_API_KEY)
        #   claude_agent_sdk/<model>   -> Claude Agent SDK (Claude Code OAuth; no API key)
        # Register whichever prefix is in use. Lazy imports (need litellm / the SDK).
        _sim_models = (self.user_llm, self.nl_assertions_llm, self.env_interface_llm)
        if any(str(_m).startswith("metagen/") for _m in _sim_models):
            from .litellm_metagen import register as _register_metagen

            _register_metagen()
        if any(str(_m).startswith("claude_agent_sdk/") for _m in _sim_models):
            from .litellm_claude_agent_sdk import register as _register_claude

            _register_claude()

        # Tau2 hard-codes DEFAULT_LLM_NL_ASSERTIONS / _ENV_INTERFACE to gpt-4.1.
        # Always monkey-patch with the resolved values above so the judge
        # never silently lands on the gpt-4.1 default.
        if True:  # always-on guard kept for diff-stability with prior block
            import tau2.config as _tau2_cfg
            _tau2_cfg.DEFAULT_LLM_NL_ASSERTIONS = nl_assertions_llm
            _tau2_cfg.DEFAULT_LLM_ENV_INTERFACE = env_interface_llm
            # tau2.evaluator.evaluator_nl_assertions did  (early bind); patch the imported ref too.
            class _SkipShim(Exception):
                pass
            try:
                import json as _json_mod
                import re as _re_mod

                import tau2.evaluator.evaluator_nl_assertions as _tau2_nl
                _tau2_nl.DEFAULT_LLM_NL_ASSERTIONS = nl_assertions_llm
                # Claude often wraps JSON in markdown fences (```json ... ```)
                # or adds a preamble. Replace the module's json.loads with a
                # fence-tolerant variant ONLY when the NL judge is a Claude
                # model -- OpenAI models return strict JSON and don't need it.
                # Apply tolerant JSON shim universally. Earlier we gated this on
                # "is the judge a Claude model" because Claude routinely wraps JSON
                # in markdown fences; in practice gpt-5-mini and gemini both emit
                # malformed JSON on a fraction of long-trajectory eval prompts too,
                # killing entire training runs. The shim is a strict superset of
                # json.loads (passes valid JSON through verbatim) so there is no
                # downside to enabling it for every judge model.
                _orig_loads = _json_mod.loads

                def _tolerant_loads(s, *args, **kwargs):
                    if not isinstance(s, str):
                        return _orig_loads(s, *args, **kwargs)
                    s2 = s.strip()
                    fence = _re_mod.search(
                        r"```(?:json)?\s*(.*?)\s*```", s2, _re_mod.DOTALL
                    )
                    if fence:
                        s2 = fence.group(1)
                    if not s2.lstrip().startswith(("{", "[")):
                        # Pull the first JSON object/array out of any preamble.
                        m = _re_mod.search(r"[{\[].*[}\]]", s2, _re_mod.DOTALL)
                        if m:
                            s2 = m.group(0)
                    try:
                        return _orig_loads(s2, *args, **kwargs)
                    except Exception:
                        # Stash the raw judge output so the reward path can persist it.
                        # The audit log previously recorded only the exception type,
                        # which made it impossible to tell an empty response (SDK
                        # failure) from genuinely malformed JSON without re-deriving it.
                        Tau2BenchEnv._last_bad_judge_output = s
                        raise

                # IMPORTANT: replace the module's `json` reference with a small
                # shim, NOT json.loads itself. Mutating json.loads on the shared
                # module breaks every other json.loads in the process (e.g. HF
                # tokenizer loader, tinker telemetry, etc.).
                class _JsonShim:
                    loads = staticmethod(_tolerant_loads)
                    def __getattr__(self, name):
                        return getattr(_json_mod, name)
                _tau2_nl.json = _JsonShim()
            except Exception:  # noqa: BLE001
                pass
        # task_split not used by tau2 v0.2.0 API (no built-in splits)
        self.num_train_tasks = num_train_tasks
        self.num_val_tasks = num_val_tasks
        self.num_test_tasks = num_test_tasks
        # Normalize explicit id lists to plain python lists (OmegaConf passes
        # ListConfig objects); keep None when not provided.
        self.train_task_ids = list(train_task_ids) if train_task_ids is not None else None
        self.eval_task_ids = list(eval_task_ids) if eval_task_ids is not None else None
        self.task_id_map_file = task_id_map_file
        self.max_steps = max_steps
        self.system_prompt = system_prompt
        self.eval_num_tries = eval_num_tries

        # Track the current tau2 env for cleanup
        self._current_tau2_env = None

        # Set TAU2_DATA_DIR before any tau2 imports (resolves paths at import time)
        import os as _os

        if not _os.getenv("TAU2_DATA_DIR"):
            _candidate = _os.path.join("data", "tau2", "_repo", "data")
            if _os.path.exists(_candidate):
                _os.environ["TAU2_DATA_DIR"] = _os.path.abspath(_candidate)
                logger.info("Set TAU2_DATA_DIR=%s", _os.environ["TAU2_DATA_DIR"])

        # Load tasks and split into train/eval/test
        self.datasets = self._init_data()

        # Extract tool descriptions and domain policy from a temporary AgentGymEnv
        self.tool_descriptions, self.domain_policy = self._init_tools_and_policy()

        # Build full system prompt incorporating domain policy
        self._full_system_prompt = self._build_system_prompt()

    def __del__(self) -> None:
        """Close the tau2 env on garbage collection to prevent resource leaks."""
        if hasattr(self, "_current_tau2_env") and self._current_tau2_env is not None:
            try:
                self._current_tau2_env.close()
            except Exception:
                pass
            self._current_tau2_env = None

    def _init_data(self) -> dict[str, list[Any]]:
        """
        Load tau2 tasks and split into train/train_eval/train_all/eval/test.

        Uses tau2's `get_tasks` to load all tasks for the domain, shuffles
        with data_seed for consistent splits, then partitions by index.

        Split convention (consistent with ActPrmEnv and TextWorldEnv):
            test       — held-out tasks for final RL evaluation
            eval       — alias for test (used by RL trainers)
            train      — training tasks (for Act-PRM RL / default RL)
            train_eval — 20% of train tasks (for validation during Act-PRM/SFT)
            train_all  — all train tasks (for SFT corpus generation)

        Returns:
            Dict mapping split name to list of tau2 Task objects.
        """

        # Use registry.get_tasks_loader -- the canonical per-domain task
        # list that AgentGymEnv._get_task looks up against. tau2.run.get_tasks
        # is a parallel loader that returns persona-multiplied subsets via
        # task_set_name; for telecom the two return different ID spaces, so
        # we'd pass IDs the gym agent can't find. Aligning here keeps
        # task.id round-trippable through AgentGymEnv(task_id=task.id).
        from tau2 import registry

        all_tasks = registry.get_tasks_loader(self.domain)()

        # --- Explicit task-id selection (RL: train on logged tasks, eval on never-seen) ---
        # Overrides the shuffle-split-by-count below. task.id is round-trippable through
        # AgentGymEnv(task_id=task.id), so we just filter the canonical list by id.
        train_ids, eval_ids = self.train_task_ids, self.eval_task_ids
        if (train_ids is None or eval_ids is None) and self.task_id_map_file:
            import json as _json
            _m = _json.loads(open(self.task_id_map_file).read())
            train_ids = train_ids if train_ids is not None else _m.get("covered_tau2_ids")
            eval_ids = eval_ids if eval_ids is not None else _m.get("unseen_tau2_ids")
        if train_ids is not None and eval_ids is not None:
            by_id = {str(t.id): t for t in all_tasks}
            train_ids = [str(i) for i in train_ids]
            eval_ids = [str(i) for i in eval_ids]
            missing = [i for i in (train_ids + eval_ids) if i not in by_id]
            if missing:
                logger.warning("tau2bench %s: %d task ids not found: %s", self.domain, len(missing), missing[:10])
            all_train_tasks = [by_id[i] for i in train_ids if i in by_id]
            eval_tasks = [by_id[i] for i in eval_ids if i in by_id]
            n_te = (self.num_val_tasks if self.num_val_tasks
                    else max(1, len(all_train_tasks) // 5)) if len(all_train_tasks) > 1 else 0
            datasets = {
                "train": all_train_tasks[: len(all_train_tasks) - n_te],
                "train_eval": all_train_tasks[len(all_train_tasks) - n_te:],
                "train_all": all_train_tasks,
                "eval": eval_tasks,
                "test": eval_tasks,
            }
            for k, v in datasets.items():
                logger.info("tau2bench [%s] %s (explicit ids): %d tasks", self.domain, k, len(v))
            return datasets

        total_needed = self.num_train_tasks + self.num_test_tasks
        if self.num_val_tasks is not None:
            total_needed += self.num_val_tasks

        if len(all_tasks) < total_needed:
            logger.warning(
                "Only %d tasks available for domain '%s', but %d requested.",
                len(all_tasks),
                self.domain,
                total_needed,
            )

        # Shuffle tasks with data_seed for consistent splits across runs
        np.random.seed(self.data_seed)
        all_indices = np.arange(len(all_tasks))
        np.random.shuffle(all_indices)
        all_tasks = [all_tasks[i] for i in all_indices]

        # Split by index: test first (for stable eval), then train
        test_tasks = all_tasks[: self.num_test_tasks]
        all_train_tasks = all_tasks[self.num_test_tasks : self.num_test_tasks + self.num_train_tasks]

        # Sub-split train into train_train (80%) and train_eval (20%)
        # for Act-PRM/SFT training and validation
        if self.num_val_tasks is not None:
            n_train_eval = self.num_val_tasks
        else:
            n_train_eval = max(1, len(all_train_tasks) // 5)  # 20%
        n_train_train = len(all_train_tasks) - n_train_eval

        train_train_tasks = all_train_tasks[:n_train_train]
        train_eval_tasks = all_train_tasks[n_train_train:]

        datasets = {
            "train": train_train_tasks,
            "train_eval": train_eval_tasks,
            "train_all": all_train_tasks,
            "eval": test_tasks,
            "test": test_tasks,
        }
        for split_name, split_tasks in datasets.items():
            logger.info(
                "tau2bench [%s] %s: %d tasks",
                self.domain,
                split_name,
                len(split_tasks),
            )
        return datasets

    def _init_tools_and_policy(self) -> tuple[list[dict[str, Any]], str]:
        """
        Extract tool descriptions and domain policy from a temporary AgentGymEnv.

        Creates a throwaway environment for the first available task, resets it
        to get the info dict (which contains tools and policy), then closes it.

        Returns:
            Tuple of (flat tool descriptions list, domain policy string).
        """
        import os

        from tau2.gym.gym_agent import AgentGymEnv

        # Ensure TAU2_DATA_DIR is set
        if not os.getenv("TAU2_DATA_DIR"):
            candidate = os.path.join("data", "tau2", "_repo", "data")
            if os.path.exists(candidate):
                os.environ["TAU2_DATA_DIR"] = os.path.abspath(candidate)

        # Tools + policy are domain-level (identical across tasks), so use the
        # first task from ANY non-empty split. The *_train_all configs leave
        # test/eval empty (num_test_tasks: 0), so don't hardcode "test".
        sample_task = next(
            (ds[0] for ds in self.datasets.values() if ds is not None and len(ds) > 0),
            None,
        )
        if sample_task is None:
            raise ValueError(
                f"tau2bench [{self.domain}]: no tasks in any split to extract tools/policy from."
            )
        tmp_env = AgentGymEnv(
            domain=self.domain,
            task_id=sample_task.id,
            user_llm=self.user_llm,
            user_llm_args=self.user_llm_args,
            max_steps=self.max_steps,
        )
        _, info = tmp_env.reset()
        tau2_tools = info.get("tools", [])
        policy = info.get("policy", "")
        tool_descriptions = convert_tau2_tools(tau2_tools)
        tmp_env.close()
        return tool_descriptions, policy

    def _build_system_prompt(self) -> str:
        """
        Build the full system prompt incorporating domain policy and respond_user instruction.

        Combines the base system prompt with the airline/domain policy and
        an instruction to always use the respond_user tool for user-facing messages.

        Returns:
            Complete system prompt string.
        """
        parts = [self.system_prompt]

        if self.domain_policy:
            parts.append(f"\n\n<policy>\n{self.domain_policy}\n</policy>")

        # Instruct agent to use respond_user for all user-facing messages
        parts.append(
            "\n\nIMPORTANT: When you want to respond or send a message to the user, "
            "you MUST use the `respond_user` tool with your message as the `text` argument. "
            "Do NOT send plain text messages — always use tool calls."
        )
        return "".join(parts)

    def __len__(self) -> int:
        """Get the number of tasks for the current split."""
        return len(self.datasets[self.split])

    def reset(
        self,
        sample_id: int = 0,
        generation_id: int = 0,
        try_step: int = 0,
        batch_id: int = 0,
        **kwargs: Any,
    ) -> Tau2BenchState:
        """
        Reset environment by creating a fresh AgentGymEnv for the selected task.

        Args:
            sample_id: Index into the current split's task list (wraps around).
            generation_id: Generation index (for multi-generation rollouts).
            try_step: Retry step counter.
            batch_id: Batch index.

        Returns:
            Initial Tau2BenchState with tools, system prompt, and first user message.
        """
        # Close any previous episode's environment to prevent resource leaks
        if hasattr(self, "_current_tau2_env") and self._current_tau2_env is not None:
            try:
                self._current_tau2_env.close()
            except Exception as e:
                logger.warning(f"tau2bench: failed to close previous env: {e}")
            self._current_tau2_env = None

        sample_id_adj = sample_id % len(self.datasets[self.split])
        task = self.datasets[self.split][sample_id_adj]

        # Create fresh tau2 gym environment for this episode
        from tau2.gym.gym_agent import AgentGymEnv

        tau2_env = AgentGymEnv(
            domain=self.domain,
            task_id=task.id,
            user_llm=self.user_llm,
            user_llm_args=self.user_llm_args,
            max_steps=self.max_steps,
        )
        # Monkey-patch _get_reward to use ALL_WITH_NL_ASSERTIONS
        # (retail domain tasks require NL assertion evaluation)
        _original_get_reward = tau2_env._get_reward

        def _patched_get_reward():
            import json as _json
            import os as _os
            import time as _time

            from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation

            if tau2_env._simulation_run is None:
                return 0.0, _json.dumps({}, indent=2)

            # Retry before giving up: the judge fails stochastically (SDK turn-limit
            # errors / malformed JSON), so most failures clear on a second attempt.
            _attempts = max(1, int(getattr(type(self), "JUDGE_MAX_ATTEMPTS", 3)))
            _exc: Exception | None = None
            for _attempt in range(_attempts):
                type(self)._last_bad_judge_output = None
                try:
                    evaluation_result = evaluate_simulation(
                        simulation=tau2_env._simulation_run,
                        task=tau2_env._get_task(),
                        evaluation_type=EvaluationType.ALL_WITH_NL_ASSERTIONS,
                        solo_mode=tau2_env.solo_mode,
                        domain=tau2_env.domain,
                    )
                except Exception as _e:  # noqa: BLE001
                    _exc = _e
                    logger.warning(
                        "tau2 evaluator attempt %d/%d failed (%s: %s)",
                        _attempt + 1, _attempts, type(_e).__name__, _e,
                    )
                    continue
                if _attempt:
                    logger.info("tau2 evaluator recovered on attempt %d", _attempt + 1)
                return evaluation_result.reward, evaluation_result.model_dump_json(indent=2)

            # Every attempt failed. Do NOT fabricate reward=0.0: an unscored episode
            # is not a failed one, and under RLVR (advantage = +1 success / 0 fail, no
            # baseline) a false zero silently removes the episode's gradient while
            # still counting against the reported success rate. Mark it unscored via
            # `judge_failed` in the info payload so the caller can drop the episode.
            logger.warning(
                "tau2 evaluator failed after %d attempts (%s: %s); marking episode UNSCORED",
                _attempts, type(_exc).__name__, _exc,
            )
            try:
                _failures_log = _os.path.join("logs", "tau2_judge_failures.jsonl")
                _os.makedirs("logs", exist_ok=True)
                _bad = type(self)._last_bad_judge_output
                with open(_failures_log, "a") as _fh:
                    _fh.write(_json.dumps({
                        "ts": _time.time(),
                        "domain": tau2_env.domain,
                        "task_id": getattr(tau2_env._get_task(), "id", None),
                        "exc_type": type(_exc).__name__,
                        "exc_msg": str(_exc),
                        "attempts": _attempts,
                        # The actual judge text, so a new failure shape is diagnosable
                        # from the log alone instead of by re-deriving it.
                        "judge_output": (_bad[:4000] if isinstance(_bad, str) else None),
                        "judge_output_empty": (not _bad) if _bad is not None else None,
                    }) + "\n")
            except Exception:  # noqa: BLE001 -- diagnostic only
                pass
            return 0.0, _json.dumps(
                {"error": f"{type(_exc).__name__}: {_exc}", "judge_failed": True}, indent=2
            )

        tau2_env._get_reward = _patched_get_reward
        self._current_tau2_env = tau2_env  # track for cleanup on next reset
        obs_str, info = tau2_env.reset(seed=self.seed + sample_id)

        # Parse observation to get initial user message
        user_content = parse_observation(obs_str) if obs_str else "Hello, I need help."
        messages = [{"role": "user", "content": user_content}]

        if self.verbose and sample_id == 0 and generation_id == 0 and try_step == 0:
            logger.info(f"tau2bench reset: task_id={task.id}, obs={user_content[:100]}...")

        return Tau2BenchState(
            system_prompt=self._full_system_prompt,
            new_messages=messages,
            model_response=None,
            prior_messages=[],
            tools=self.tool_descriptions,
            # tau2-specific fields
            tau2_gym_env=tau2_env,
            task_id=task.id,
            tau2_info=info,
            # Step-wise metadata
            sample_id=sample_id,
            generation_id=generation_id,
            batch_id=batch_id,
            try_step=try_step,
            timestep=0,
            split=self.split,
            metadata={"correct": 0, "total": 1},
            first_obs_to_show=len(messages) + 1,  # keep system + initial user message visible
            # ICL / retrieval metadata
            task_prompt=user_content,  # initial user message as task prompt
            default_context=[],
            prior_context=[],
            prior_rewards=[],
            prior_returns=[],
            prior_advantages=[],
        )

    def step(self, **kwargs: Any) -> Tau2BenchStepResult:
        """Perform one step through the tau2-bench environment."""
        return self._step_impl(**kwargs)

    def _step_impl(
        self,
        parsed_actions: list[ActionFromLLM],
        model_response: Any,
        current_state: Tau2BenchState,
        current_messages: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> Tau2BenchStepResult:
        """
        Step through the environment by routing parsed actions to tau2's gym env.

        Iterates over parsed_actions (following TextWorld's per-action loop pattern):
        - For `respond_user` calls: extracts text, sends to tau2, parses user response
        - For regular tool calls: formats as JSON, sends to tau2, gets tool result
        - For plain text (fallback): sends directly to tau2

        Args:
            parsed_actions: List of ActionFromLLM objects from the LLM's response.
            model_response: Raw model response (stored in state for provenance).
            current_state: Current Tau2BenchState from the previous step.
            current_messages: Full message history (for observation hiding).

        Returns:
            Tau2BenchStepResult with new state, reward, done, and truncated flags.
        """
        done = False
        truncated = False
        reward = 0.0

        tau2_env = current_state.tau2_gym_env
        metadata = copy(current_state.metadata)
        timestep = int(current_state.timestep)
        try_step = int(current_state.try_step)
        updated_try_step = False
        tau2_info = current_state.tau2_info

        env_messages: list[dict[str, Any]] = []
        made_tool_call = False

        for action_idx, action in enumerate(parsed_actions):
            if action.type == "function_call":
                fc_name = action.name
                fc_args = action.arguments or {}

                if fc_name == "invalid_tool_call":
                    # action_utils.py synthesizes this placeholder when the
                    # model emits a malformed <tool_call>...</tool_call>.
                    # Do NOT forward it to tau2_env.step -- tau2 would
                    # record it in the trajectory and then crash at reward
                    # replay time ("Unknown tool 'invalid_tool_call' ...").
                    # Instead, surface an error message back to the model
                    # so it can self-correct, mirroring browsecomp_plus's
                    # pattern (see browsecomp_plus/env.py:505).
                    error_msg = (
                        "Invalid tool call (malformed JSON or missing fields):\n\n"
                        f"{action.text or ''}"
                    )
                    env_messages.append(
                        {
                            "role": "tool",
                            "type": "function_call_output",
                            "call_id": action.call_id,
                            "output": error_msg,
                        }
                    )
                    # Don't toggle made_tool_call -- the rollout should still
                    # advance the timestep but not bump the try_step counter.
                    continue

                # Reject hallucinated tool names BEFORE handing them to
                # tau2_env.step. tau2 would otherwise record the unknown
                # tool in the trajectory and crash at reward-replay
                # ("Unknown tool 'X' encountered during replay"). Surface
                # an "available tools" hint so the model can self-correct.
                _valid_tool_names = {
                    t.get("name") for t in (self.tool_descriptions or [])
                    if isinstance(t, dict) and t.get("name")
                }
                _valid_tool_names.add("respond_user")  # always available
                if fc_name not in _valid_tool_names:
                    _available = sorted(_valid_tool_names)
                    error_msg = (
                        f"Unknown tool '{fc_name}'. This tool is not available "
                        f"in the {self.domain} environment. Available tools: "
                        f"{', '.join(_available[:25])}"
                        + (f" (+{len(_available) - 25} more)" if len(_available) > 25 else "")
                    )
                    env_messages.append(
                        {
                            "role": "tool",
                            "type": "function_call_output",
                            "call_id": action.call_id,
                            "output": error_msg,
                        }
                    )
                    continue

                if fc_name == "respond_user":
                    # Send user-facing message through tau2's orchestrator
                    user_text = fc_args.get("text", "")
                    obs, step_reward, terminated, step_truncated, info = tau2_env.step(user_text)
                    tau2_info = info

                    # Tool result confirming message was sent
                    env_messages.append(
                        {
                            "role": "tool",
                            "type": "function_call_output",
                            "call_id": action.call_id,
                            "output": "Message sent to user.",
                        }
                    )

                    reward = float(step_reward)
                    done = bool(terminated) or bool(step_truncated)
                    if step_truncated:
                        truncated = True

                    # If not done, add the user's response as a new user message
                    if not done:
                        user_response = parse_observation(obs)
                        if user_response:
                            env_messages.append(
                                {
                                    "role": "user",
                                    "content": user_response,
                                }
                            )
                    made_tool_call = True

                else:
                    # Regular tool call: format as JSON and send to tau2
                    action_str = json.dumps(
                        {
                            "name": fc_name,
                            "arguments": fc_args,
                        }
                    )
                    try:
                        obs, step_reward, terminated, step_truncated, info = tau2_env.step(action_str)
                        tau2_info = info

                        # Parse tool result from observation
                        tool_output = parse_observation(obs) if obs else "No output."
                        env_messages.append(
                            {
                                "role": "tool",
                                "type": "function_call_output",
                                "call_id": action.call_id,
                                "output": tool_output,
                            }
                        )

                        reward = float(step_reward)
                        done = bool(terminated) or bool(step_truncated)
                        if step_truncated:
                            truncated = True
                        made_tool_call = True

                    except Exception as e:
                        error_msg = f"Tool call error for '{fc_name}': {type(e).__name__}: {e}"
                        logger.warning(error_msg)
                        env_messages.append(
                            {
                                "role": "tool",
                                "type": "function_call_output",
                                "call_id": action.call_id,
                                "output": error_msg,
                            }
                        )
                        done = True
                        truncated = True
                        if not updated_try_step:
                            try_step += 1
                            updated_try_step = True

                # Stop processing further actions if the episode ended
                if done:
                    break

            elif action.type in ["message", "reasoning"]:
                # Plain text or reasoning — only act on the last one
                if action_idx + 1 == len(parsed_actions):
                    # Fallback: send plain text to tau2 as a user-facing message
                    text = action.text or ""
                    if text.strip():
                        obs, step_reward, terminated, step_truncated, info = tau2_env.step(text)
                        tau2_info = info
                        reward = float(step_reward)
                        done = bool(terminated) or bool(step_truncated)
                        if step_truncated:
                            truncated = True

                        if not done:
                            user_response = parse_observation(obs)
                            if user_response:
                                env_messages.append(
                                    {
                                        "role": "user",
                                        "content": user_response,
                                    }
                                )
                    else:
                        # Empty text — treat as failure
                        done = True
                        truncated = True
                        env_messages.append(
                            {
                                "role": "user",
                                "content": "You must use tool calls to interact. Please try again.",
                            }
                        )
                        if not updated_try_step:
                            try_step += 1
                            updated_try_step = True
            else:
                logger.error(f"Unknown action type '{action.type}' at index {action_idx}: {action}")

        # Update timestep, check turn limit. Do NOT append a separate truncation
        # message here -- the SUCCESS/FAILURE block below provides the single
        # canonical user reply for any terminal step (truncated or otherwise).
        timestep += 1
        if timestep >= self.max_turns + 1:
            truncated = True
            done = True
            if not updated_try_step:
                try_step += 1
                updated_try_step = True

        # When the episode ends, append a single SUCCESS / FAILURE marker as the
        # final user reply, set ``metadata['correct']`` to {0, 1} on a strict
        # success threshold against tau2's evaluator reward, and binarize the
        # reward to {+1, -1} (negative_rewards=True) or {+1, 0} (False).
        #
        # Success = tau2 returned reward >= 1.0 (i.e. the ALL_WITH_NL_ASSERTIONS
        # evaluator reported all assertions passed). Anything below that --
        # partial credit, truncation, parse errors -- counts as failure.
        if done:
            is_success = reward >= 1.0
            metadata["correct"] = int(is_success)
            metadata["task_success"] = bool(is_success)
            metadata["raw_tau2_reward"] = float(reward)
            result_label = "SUCCESS" if is_success else "FAILURE"
            env_messages.append(
                {"role": "user", "content": f"# OVERALL TASK ASSESSMENT: {result_label}!"}
            )
            reward = 1.0 if is_success else (
                -1.0 if self.negative_rewards else 0.0
            )
        elif len(env_messages) == 0:
            # Episode is still active but the policy emitted no actionable
            # message -- nudge it to keep going. Not an error.
            env_messages.append(
                {
                    "role": "user",
                    "content": "No actions were parsed. Please use tool calls to continue.",
                }
            )
            logger.warning(
                "No actions parsed in tau2bench step (rollout still active, prompting policy to retry)."
            )

        # Handle observation hiding for prior messages
        current_messages = self.maybe_hide_observations(
            current_messages or [],
            first_obs_to_show=current_state.first_obs_to_show,
            last_obs_to_show=current_state.last_obs_to_show,
        )

        metadata.update(
            {
                "reward": reward,
                "done": done,
                "truncated": truncated,
                "made_tool_call": made_tool_call,
                "task_id": current_state.task_id,
            }
        )

        new_state = Tau2BenchState(
            system_prompt=current_state.system_prompt,
            new_messages=env_messages,
            model_response=model_response,
            prior_messages=current_messages,
            tools=current_state.tools,
            # tau2-specific fields
            tau2_gym_env=tau2_env,
            task_id=current_state.task_id,
            tau2_info=tau2_info,
            # Step-wise metadata
            sample_id=current_state.sample_id,
            generation_id=current_state.generation_id,
            batch_id=current_state.batch_id,
            try_step=try_step,
            timestep=timestep,
            split=self.split,
            metadata=metadata,
            first_obs_to_show=current_state.first_obs_to_show,
            last_obs_to_show=current_state.last_obs_to_show,
            # ICL / retrieval metadata (propagate from current state)
            task_prompt=current_state.task_prompt,
            default_context=current_state.default_context,
            prior_context=current_state.prior_context,
            prior_rewards=current_state.prior_rewards,
            prior_returns=current_state.prior_returns,
            prior_advantages=current_state.prior_advantages,
        )
        return Tau2BenchStepResult(
            state=new_state,
            reward=reward,
            done=done,
            truncated=truncated,
            info=new_state.metadata,
        )


class AsyncTau2BenchEnv(Tau2BenchEnv):
    """
    Asynchronous wrapper for the tau2-bench environment.

    Uses asyncio.to_thread for reset and step since tau2's AgentGymEnv
    involves I/O-bound LLM calls in its orchestrator thread.
    """

    async def reset_async(
        self,
        sample_id: int = 0,
        generation_id: int = 0,
        try_step: int = 0,
        batch_id: int = 0,
        **kwargs: Any,
    ) -> Tau2BenchState:
        """
        Asynchronous reset — delegates to sync reset via asyncio.to_thread
        since tau2's orchestrator involves LLM API calls.
        """
        return await asyncio.to_thread(
            super().reset,
            sample_id=sample_id,
            generation_id=generation_id,
            try_step=try_step,
            batch_id=batch_id,
            **kwargs,
        )

    async def step_async(self, **kwargs: Any) -> Tau2BenchStepResult:
        """
        Asynchronous step — delegates to sync step via asyncio.to_thread
        since tau2's orchestrator involves LLM API calls.
        """
        return await asyncio.to_thread(super().step, **kwargs)
