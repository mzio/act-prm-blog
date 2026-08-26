"""
``ActPrmGenerator`` — the Act-PRM EM E-step as a HuggingFace rollout harness.

For each logged action ``x_t`` in state ``s`` of an action-only trajectory:

  E-step
    1. Build a "reversal" prompt (show the action first, continue from an open
       ``<thought>``) and sample ``G`` candidate thoughts ``z``.
    2. Score each candidate by the length-normalized action likelihood
       ``p(x | s, z)`` (one teacher-forced forward pass over ``s, z, x``).
    3. Apply the length penalty and group-normalize to non-negative EM weights.
    4. Commit the best thought (it seeds the reversal prompt of later steps).

Each (step, candidate) becomes a trainable :class:`EpisodeStep` whose
``advantage`` is the EM weight and whose supervised span is the (thought+action)
tokens. The downstream ``RLTrainer`` M-step is then an importance-weighted NLL
over that span — i.e. the group-normalized REINFORCE / EM update.

This reuses the shared HF machinery (batched generation + right-padded logprob
recomputation) from :mod:`act_prm.generator.utils` and inherits usage tracking /
trajectory-group helpers from :class:`HuggingFaceGenerator`.
"""

import logging
from copy import copy
from typing import Any

import numpy as np
import torch

from act_prm.generator.utils import (
    get_action_logprobs_and_state_action_tokens,
    get_batch_model_inputs,
)
from act_prm.replay_buffer.types import EpisodeStep, Trajectory, TrajectoryGroup

from ..huggingface.base import HuggingFaceGenerator
from .prompts import (
    THOUGHT_EOS,
    build_scoring_messages,
    build_thought_prefix_messages,
    build_thought_prompt_messages,
)

logger = logging.getLogger(__name__)


def em_weights(penalized_rewards: list[float], likelihoods: list[float]) -> np.ndarray:
    """Group-normalized EM weights from possibly-negative penalized rewards:
    clamp at 0 and normalize by the sum. If the penalty pushed the whole group
    non-positive (common early in training, when likelihoods are still tiny),
    fall back to normalizing the raw likelihoods so the step still carries
    learning signal — the penalty then only affects *selection*, not weights."""
    clamped = np.maximum(np.array(penalized_rewards, dtype=np.float64), 0.0)
    total = clamped.sum()
    if total <= 0:
        lik = np.array(likelihoods, dtype=np.float64)
        return lik / lik.sum() if lik.sum() > 0 else np.full(len(lik), 1.0 / len(lik))
    return clamped / total


class ActPrmGenerator(HuggingFaceGenerator):
    """Act-PRM EM harness over logged action-only trajectories."""

    def __init__(
        self,
        *args: Any,
        max_thought_tokens: int = 96,
        length_penalty: float = 0.15,
        reward_method: str = "penalty",  # "penalty" | "lift"
        select_epsilon: float = 0.05,  # "lift" lexicographic selection tolerance
        lift_c: float = 4.0,  # "lift" length denominator constant
        thought_temperature: float = 1.0,
        use_fewshot: bool = True,
        max_steps_per_traj: int | None = None,
        # How the per-candidate rewards become EpisodeStep advantages:
        #   "em"       — clamped, group-normalized EM weights (>=0, sum 1) [EM/RL default]
        #   "best"     — 1.0 on the selected (argmax-reward) thought, else 0 [hard SFT]
        #   "top_half" — 1.0 on the better half by reward, else 0
        #   "uniform"  — 1.0 on every thought
        #   "grpo"     — mean-centered reward (r - mean), optionally /std; can be negative
        #   "action_probs" — RAW length-normalised p(x|s,z), no group normalisation, the
        #                    length penalty affects SELECTION only (Tinker's aprm_qwen3_ap)
        #   "clamped"  — max(reward, 0) unnormalised: keeps the length penalty IN the
        #                advantage, but drops the group-sum division that "em" applies
        advantage_mode: str = "em",
        grpo_normalize: bool = True,
        # Score p(x|s,z) with the frozen BASE model (LoRA disabled) instead of the
        # current policy. Better aligned when the relabelled thoughts will SFT the
        # base model, and stops the policy inflating its own reward by co-adapting.
        # The datum old_logprobs stay the policy's (for importance sampling).
        score_with_base: bool = False,
        # infer_thoughts=False -> actions-only baseline: no thought sampling, train on
        # (state -> logged action) with advantage 1.0 (the ground-truth comparison).
        infer_thoughts: bool = True,
        # Persist every generation group (thoughts, rewards, weights, selection) to JSONL.
        save_generations: bool = True,
        generations_path: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.max_thought_tokens = max_thought_tokens
        self.length_penalty = length_penalty
        self.reward_method = reward_method
        self.select_epsilon = select_epsilon
        self.lift_c = lift_c
        self.thought_temperature = thought_temperature
        self.use_fewshot = use_fewshot
        self.max_steps_per_traj = max_steps_per_traj
        self.advantage_mode = advantage_mode
        self.grpo_normalize = grpo_normalize
        self.score_with_base = score_with_base
        self.infer_thoughts = infer_thoughts
        self.save_generations = save_generations
        # Default the generations log under the run's log_path.
        _log_path = generations_path or (self.cfg.get("log_path", "./logs") if self.cfg else "./logs")
        self.generations_path = (
            generations_path if generations_path else f"{_log_path}/generations.jsonl"
        )

    # ------------------------------------------------------------------
    # tokenization helpers
    # ------------------------------------------------------------------
    def _n_tokens(self, messages: list[dict[str, str]], **template_kwargs: Any) -> int:
        """Number of tokens in a rendered chat (via the model's template)."""
        out = self.hf_tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            enable_thinking=self.enable_thinking,
            **template_kwargs,
        )
        if not isinstance(out, (list, tuple)):  # newer transformers -> BatchEncoding
            out = out["input_ids"]
            if out and isinstance(out[0], (list, tuple)):
                out = out[0]
        return len(out)

    @torch.no_grad()
    def _action_logprobs(
        self, model_inputs: Any, state_len: int, use_base: bool = False
    ) -> tuple[list[list[float]], list[list[int]]]:
        """Batched forward -> per-sequence (thought+action) logprobs + token ids.

        ``use_base=True`` disables the LoRA adapter so the frozen base model scores
        the sequence (frozen-scorer reward). Falls back to per-sample forwards on OOM.
        """
        device = self.llm.model.device
        n = model_inputs["input_ids"].shape[0]
        import contextlib

        adapter_ctx = self.llm.model.disable_adapter() if use_base else contextlib.nullcontext()
        try:
            with adapter_ctx:
                logits = self.llm.model(**model_inputs.to(device), use_cache=False).logits
                gen_logprobs, sa_tokens = get_action_logprobs_and_state_action_tokens(
                    logits=logits, state_lens=[state_len] * n, **model_inputs.to(device)
                )
            del logits
            return gen_logprobs, sa_tokens
        except torch.OutOfMemoryError:
            logger.warning(
                "OOM in %s forward at seq_len=%d; falling back to per-sample",
                "base" if use_base else "policy",
                model_inputs["input_ids"].shape[1],
            )
            torch.cuda.empty_cache()
            gen_logprobs, sa_tokens = [], []
            for s in range(n):
                single = {k: v[s : s + 1].to(device) for k, v in model_inputs.items()}
                adapter_ctx = self.llm.model.disable_adapter() if use_base else contextlib.nullcontext()
                with adapter_ctx:
                    _logits = self.llm.model(**single, use_cache=False).logits
                    _lp, _tok = get_action_logprobs_and_state_action_tokens(
                        logits=_logits, state_lens=[state_len], **single
                    )
                gen_logprobs.extend(_lp)
                sa_tokens.extend(_tok)
                del _logits
                torch.cuda.empty_cache()
            return gen_logprobs, sa_tokens

    # ------------------------------------------------------------------
    # E-step pieces
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _sample_thoughts(
        self,
        state_messages: list[dict[str, str]],
        target_action: str,
        committed: list[str],
        group_size: int,
        max_tokens: int,
        temperature: float,
    ) -> tuple[list[str], list[int]]:
        """Sample ``group_size`` candidate thoughts from the reversal prompt."""
        device = self.llm.model.device
        reversal_msgs = build_thought_prompt_messages(
            state_messages, target_action, committed, use_fewshot=self.use_fewshot
        )
        model_inputs, _ = get_batch_model_inputs(
            input_messages=[reversal_msgs],
            tools=None,
            hf_tokenizer=self.hf_tokenizer,
            padding_side="left",
            enable_thinking=self.enable_thinking,
            add_generation_prompt=False,
            continue_final_message=True,
        )
        input_len = model_inputs["input_ids"].shape[1]
        outputs = self.llm.model.generate(
            **model_inputs.to(device),
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.95,
            num_return_sequences=group_size,
            pad_token_id=self.hf_tokenizer.pad_token_id,
        )
        self._update_usage_metrics(
            input_ids_shape=model_inputs["input_ids"].shape,
            output_ids_shape=outputs.shape,
        )
        gen_texts = self.hf_tokenizer.batch_decode(
            outputs[:, input_len:], skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        thoughts, thought_lens = [], []
        for text in gen_texts:
            text = text.split(THOUGHT_EOS)[0]
            # Qwen3's template treats <think> tags specially (splits the message),
            # which breaks continue_final_message when they appear inside a thought —
            # scrub them, and never let a thought be empty.
            text = text.replace("<think>", "").replace("</think>", "").strip()
            thought = text if text else "(no thought)"
            thoughts.append(thought)
            thought_lens.append(len(self.hf_tokenizer(thought, add_special_tokens=False)["input_ids"]))
        return thoughts, thought_lens

    @torch.no_grad()
    def _score_thoughts(
        self,
        system_prompt: str,
        state_messages: list[dict[str, str]],
        thoughts: list[str],
        target_action: str,
    ) -> tuple[list[float], list[list[float]], list[list[int]], int]:
        """Teacher-forced scoring of each thought.

        Returns:
        - likelihoods: length-normalized p(x | s, z) in (0, 1], one per thought
        - gen_logprobs: policy logprobs over the (thought + action) span per thought
        - state_action_tokens: full (state + thought + action) token ids per thought
        - state_len: number of state-prompt tokens (where the thought begins)
        """
        device = self.llm.model.device

        # State-prompt boundary (with generation prompt): where the thought starts.
        scoring_state = [{"role": "system", "content": system_prompt}] + [
            m for m in state_messages if m["role"] != "system"
        ]
        state_len = self._n_tokens(scoring_state, add_generation_prompt=True)

        # Per-candidate action-token count (full render minus state+thought render).
        n_actions: list[int] = []
        for z in thoughts:
            prefix_len = self._n_tokens(
                build_thought_prefix_messages(system_prompt, state_messages, z),
                continue_final_message=True,
            )
            full_len = self._n_tokens(
                build_scoring_messages(system_prompt, state_messages, z, target_action)
            )
            n_actions.append(max(1, full_len - prefix_len))

        # Batched right-padded forward over (state, thought, action) for logprobs.
        full_messages = [
            build_scoring_messages(system_prompt, state_messages, z, target_action) for z in thoughts
        ]
        model_inputs, _ = get_batch_model_inputs(
            input_messages=full_messages,
            tools=None,
            hf_tokenizer=self.hf_tokenizer,
            padding_side="right",
            enable_thinking=self.enable_thinking,
            add_generation_prompt=False,
            continue_final_message=False,
        )
        # Policy (LoRA) forward -> old_logprobs + token ids for the M-step datum.
        gen_logprobs, state_action_tokens = self._action_logprobs(model_inputs, state_len, use_base=False)
        # Reward likelihood p(x|s,z): from the frozen base model if score_with_base,
        # else from the policy. (old_logprobs stay the policy's for importance sampling.)
        if self.score_with_base:
            reward_logprobs, _ = self._action_logprobs(model_inputs, state_len, use_base=True)
        else:
            reward_logprobs = gen_logprobs

        likelihoods: list[float] = []
        for g, n_action in enumerate(n_actions):
            action_lp = np.array(reward_logprobs[g][-n_action:], dtype=np.float64)
            likelihoods.append(float(np.exp(action_lp.mean())))  # in (0, 1]
        return likelihoods, gen_logprobs, state_action_tokens, state_len

    @torch.no_grad()
    def _score_action_only(
        self, system_prompt: str, state_messages: list[dict[str, str]], target_action: str
    ) -> tuple[list[float], list[int], int, float]:
        """Actions-only baseline: score the logged action with NO thought.

        Returns (action logprobs, (state+action) token ids, state_len, p(x|s))."""
        device = self.llm.model.device
        scoring_state = [{"role": "system", "content": system_prompt}] + [
            m for m in state_messages if m["role"] != "system"
        ]
        state_len = self._n_tokens(scoring_state, add_generation_prompt=True)
        full_msgs = scoring_state + [{"role": "assistant", "content": target_action}]
        model_inputs, _ = get_batch_model_inputs(
            input_messages=[full_msgs],
            tools=None,
            hf_tokenizer=self.hf_tokenizer,
            padding_side="right",
            enable_thinking=self.enable_thinking,
            add_generation_prompt=False,
            continue_final_message=False,
        )
        # Policy forward -> old_logprobs (datum); base forward for reward if score_with_base.
        gen_logprobs, state_action_tokens = self._action_logprobs(model_inputs, state_len, use_base=False)
        if self.score_with_base:
            reward_logprobs, _ = self._action_logprobs(model_inputs, state_len, use_base=True)
        else:
            reward_logprobs = gen_logprobs
        action_lp = np.array(reward_logprobs[0], dtype=np.float64)
        likelihood = float(np.exp(action_lp.mean())) if action_lp.size else 0.0
        return gen_logprobs[0], state_action_tokens[0], state_len, likelihood

    def _rewards_and_weights(
        self, likelihoods: list[float], thought_lens: list[int]
    ) -> tuple[list[float], np.ndarray, int]:
        """Compute per-candidate reward, advantage (per ``advantage_mode``), and
        the selected (best) index."""
        len_fracs = [min(1.0, n / self.max_thought_tokens) for n in thought_lens]
        if self.reward_method == "lift":
            # Reward = per-action-token likelihood, with a lexicographic selection
            # that prefers the shortest thought among the near-best-likelihood set.
            rewards = [lik / (n + self.lift_c) for lik, n in zip(likelihoods, thought_lens)]
            p_max = max(likelihoods)
            eligible = [
                g for g in range(len(likelihoods)) if likelihoods[g] >= p_max * (1 - self.select_epsilon)
            ]
            best = min(eligible, key=lambda g: thought_lens[g])
        else:  # "penalty" (default): p(x|s,z) minus a fraction-of-budget penalty
            rewards = [lik - self.length_penalty * lf for lik, lf in zip(likelihoods, len_fracs)]
            best = int(np.argmax(rewards))
        advantages = self._advantages(rewards, likelihoods, best)
        return rewards, advantages, best

    def _advantages(
        self, rewards: list[float], likelihoods: list[float], best: int
    ) -> np.ndarray:
        """Map per-candidate rewards to EpisodeStep advantages per ``advantage_mode``."""
        r = np.array(rewards, dtype=np.float64)
        g = len(r)
        mode = self.advantage_mode
        if mode == "em":
            return em_weights(rewards, likelihoods)
        if mode == "best":
            a = np.zeros(g, dtype=np.float64)
            a[best] = 1.0
            return a
        if mode == "uniform":
            return np.ones(g, dtype=np.float64)
        if mode == "top_half":
            k = max(1, (g + 1) // 2)  # ceil(g/2)
            top = np.argsort(r)[::-1][:k]
            a = np.zeros(g, dtype=np.float64)
            a[top] = 1.0
            return a
        if mode == "clamped":
            # Non-normalised advantage WITH the length penalty: max(lik - lp*len_frac, 0),
            # no division by the group sum. Keeps the absolute quality of the step, which
            # "em" discards -- under "em" a step whose G thoughts all score ~0.01 still gets
            # total weight 1.0 and trains as hard as a step whose best thought scores 0.99.
            # Clamped at 0 so a below-penalty thought contributes nothing rather than being
            # actively pushed down (that is what "grpo" would do).
            return np.maximum(np.array(rewards, dtype=np.float64), 0.0)
        if mode == "action_probs":
            # The Tinker reference's `reward_method: "action_probs"` -- the RAW
            # length-normalised likelihood p(x|s,z) per candidate, with NO group
            # normalisation. This is what the documented Tinker runs actually used
            # (configs/generator/aprm_qwen3_ap.yaml, referenced by act_prm_sft_rl.py and
            # act_prm_joint.py); its `em` config exists but no run command references it.
            #
            # Differs from "em" in what it preserves: "em" gives every logged action the
            # same total weight 1.0, so a step where all G thoughts are poor trains just as
            # hard as one where the best thought scores 0.99. Unnormalised keeps that
            # absolute quality, so hopeless steps contribute proportionally less.
            # Pair with --length_penalty 0 to match Tinker exactly: Tinker handles length by
            # normalising the logprob SUM by token count (already done in `likelihoods`),
            # not by our additional subtractive penalty.
            return np.array(likelihoods, dtype=np.float64)
        if mode == "grpo":
            adv = r - r.mean()
            if self.grpo_normalize:
                adv = adv / (r.std() + 1e-8)
            return adv
        raise ValueError(f"Unknown advantage_mode: {mode!r}")

    def _write_generations(self, records: list[dict[str, Any]]) -> None:
        """Append generation records (one per action-step group) to JSONL."""
        if not self.save_generations or not records:
            return
        import json
        import os

        os.makedirs(os.path.dirname(self.generations_path) or ".", exist_ok=True)
        with open(self.generations_path, "a") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")

    # ------------------------------------------------------------------
    # rollout entry point (overrides HuggingFaceGenerator.do_group_rollout)
    # ------------------------------------------------------------------
    def do_group_rollout(self, **kwargs: Any) -> dict[str, list[TrajectoryGroup]]:
        """Run the Act-PRM E-step over one logged trajectory (``sample_id``).

        Returns one :class:`TrajectoryGroup` per action-step, each holding the
        ``group_size`` candidate thoughts (as single-step trajectories) with the
        EM weights already stored as advantages.
        """
        self._reset_batch_usage_metrics()

        env = kwargs.get("env", self.env)
        cfg = kwargs.get("cfg", self.cfg)
        sample_id = kwargs["sample_id"]
        batch_id = kwargs.get("batch_id", 0)
        split = kwargs.get("split", "train")
        try_step = kwargs.get("try_step", 0)
        group_size = kwargs.get("num_return_sequences") or (
            cfg.group_size if split == "train" else cfg.eval_group_size
        )
        max_thought_tokens = kwargs.get("max_tokens") or self.max_thought_tokens
        temperature = kwargs.get("temperature") or self.thought_temperature

        was_training = self.llm.model.training
        self.llm.model.eval()
        og_padding_side = copy(self.hf_tokenizer.padding_side)

        traj = env.get_trajectory(sample_id, split)
        messages = traj["messages"]
        system_prompt = traj["system_prompt"]
        obs_max_chars = getattr(env, "obs_max_chars", 2000)
        first_obs_to_show = getattr(env, "first_obs_to_show", 1)
        last_obs_to_show = getattr(env, "last_obs_to_show", 1)
        # Thought generation sees the FULL (length-capped) observations by default;
        # the hide_observations compaction is reserved for downstream SFT/RL.
        hide_middle = getattr(env, "hide_observations", False)
        from act_prm.environments.act_prm_traces.data import compact_observations

        action_indices = [i for i, m in enumerate(messages) if m["role"] == "assistant"]
        # require_thought: train ONLY on assistant turns that carry reasoning before the
        # action. Motivation: ~50% of GPT-5-mini's logged retail actions (46% airline) are
        # a bare <tool_call> with no reasoning at all, so the expert_thoughts arm was
        # taught "usually don't think" and at rollout it reasons before 0-8% of its tool
        # calls. Filtering the TARGETS (not the messages) keeps every prior turn in the
        # context -- state is still messages[:idx] -- so trajectories stay coherent.
        #
        # Applies to TRAIN ONLY by default. Filtering eval as well would score the arm on a
        # different (and harder -- thought-bearing targets are longer) subset than every
        # other arm, making the PPL/accuracy tables silently non-comparable; that bug
        # inflated retail's mean eval target from 293.6 to 497.7 tokens and made
        # expert_thoughts_all look 24% better than baseline on finance when it was 1.4%
        # worse.
        #
        # `require_thought_eval` opts eval in as well. That is only sound when EVERY arm in
        # the comparison sets it -- then the eval subset is identical across arms (so still
        # comparable) AND on-distribution for all of them (each arm was trained on the same
        # kind of turn). This is the insurance setup: all four arms train and evaluate on
        # the 64% of turns that carry expert reasoning, which also removes the handicap that
        # sank expert_thoughts elsewhere, where a third of its targets were bare tool calls.
        _req_thought = getattr(self, "require_thought", False) or (
            cfg is not None and cfg.get("require_thought", False)
        )
        _req_thought_eval = getattr(self, "require_thought_eval", False) or (
            cfg is not None and cfg.get("require_thought_eval", False)
        )
        if _req_thought and (split == "train" or _req_thought_eval):
            from act_prm.environments.act_prm_traces.data import extract_action as _xa

            def _has_thought(i: int) -> bool:
                c = messages[i].get("content") or ""
                a = _xa(c)
                if not a or c.find(a) < 0:
                    return False  # no separable action -> not an action target
                return len(c[: c.find(a)].strip()) >= 10

            # NB: this decides targets from the CONTENT, so it only makes sense on a pool
            # that keeps the expert reasoning. An actions_only pool has it stripped, so the
            # check finds nothing and the arm would train on zero targets -- do not enable
            # require_thought for a baseline arm.
            _kept = [i for i in action_indices if _has_thought(i)]
            logger.info(
                "require_thought: %d/%d assistant turns carry reasoning (targets filtered)",
                len(_kept), len(action_indices),
            )
            action_indices = _kept
        max_steps = self.max_steps_per_traj or getattr(env, "max_steps_per_traj", None) or len(action_indices)
        action_indices = action_indices[:max_steps]

        committed: list[str] = []
        all_groups: list[TrajectoryGroup] = []
        gen_records: list[dict[str, Any]] = []

        with torch.no_grad():
            for t, idx in enumerate(action_indices):
                state = compact_observations(
                    messages[:idx],
                    obs_max_chars,
                    first_to_show=first_obs_to_show,
                    last_to_show=last_obs_to_show,
                    hide_middle=hide_middle,
                )
                x_t = messages[idx]["content"]
                scoring_state = [{"role": "system", "content": system_prompt}] + [
                    m for m in state if m["role"] != "system"
                ]

                if not self.infer_thoughts:
                    # Actions-only baseline: train on (state -> logged action), adv 1.0.
                    act_lp, sa_tokens, state_len, likelihood = self._score_action_only(
                        system_prompt, state, x_t
                    )
                    step = EpisodeStep(
                        state=scoring_state,
                        action={"role": "assistant", "content": x_t},
                        next_obs=[],
                        state_action_tokens=sa_tokens,
                        state_len=state_len,
                        old_logprobs=act_lp,
                        temperature=temperature,
                        reward=float(likelihood),
                        done=True,
                        truncated=False,
                        timestep=t,
                        try_step=try_step,
                        batch_id=batch_id,
                        sample_id=sample_id,
                        generation_id=0,
                        split=split,
                        is_train="train" in split,
                        advantage=1.0,
                        advantage_is_computed=True,
                        system_prompt=system_prompt,
                        metrics={
                            "likelihood": float(likelihood),
                            "action_tokens": float(len(act_lp)),
                            "selected": 1.0,
                        },
                    )
                    all_groups.append(
                        TrajectoryGroup(
                            trajectories=[
                                Trajectory(
                                    episode_steps=[step],
                                    try_step=try_step,
                                    discount_factor=self.discount_factor,
                                    final_reward=float(likelihood),
                                )
                            ],
                            final_rewards=[float(likelihood)],
                            discount_factor=self.discount_factor,
                        )
                    )
                    gen_records.append({
                        "batch_id": batch_id, "split": split, "sample_id": sample_id,
                        "timestep": t, "try_step": try_step, "advantage_mode": "actions_only",
                        "target_action": x_t, "thoughts": [], "likelihoods": [float(likelihood)],
                        "rewards": [float(likelihood)], "advantages": [1.0], "thought_tokens": [],
                        "best": 0,
                    })
                    continue

                # Thought inference (E-step): sample G thoughts, score, weight.
                thoughts, thought_lens = self._sample_thoughts(
                    state, x_t, committed, group_size, max_thought_tokens, temperature
                )
                likelihoods, gen_logprobs, sa_tokens, state_len = self._score_thoughts(
                    system_prompt, state, thoughts, x_t
                )
                rewards, advantages, best = self._rewards_and_weights(likelihoods, thought_lens)
                committed.append(thoughts[best])

                trajectories: list[Trajectory] = []
                for g in range(len(thoughts)):
                    step = EpisodeStep(
                        state=scoring_state,
                        action={"role": "assistant", "content": f"{thoughts[g]}\n\n{x_t}"},
                        next_obs=[],
                        state_action_tokens=sa_tokens[g],
                        state_len=state_len,
                        old_logprobs=gen_logprobs[g],
                        temperature=temperature,
                        reward=float(likelihoods[g]),
                        done=True,
                        truncated=False,
                        timestep=t,
                        try_step=try_step,
                        batch_id=batch_id,
                        sample_id=sample_id,
                        generation_id=g,
                        split=split,
                        is_train="train" in split,
                        # Advantage per advantage_mode, pre-computed so the trainer keeps it as-is
                        advantage=float(advantages[g]),
                        advantage_is_computed=True,
                        system_prompt=system_prompt,
                        metrics={
                            "thought_tokens": float(thought_lens[g]),
                            "likelihood": float(likelihoods[g]),
                            "penalized": float(rewards[g]),
                            "selected": float(g == best),
                        },
                    )
                    trajectories.append(
                        Trajectory(
                            episode_steps=[step],
                            try_step=try_step,
                            discount_factor=self.discount_factor,
                            final_reward=float(likelihoods[g]),
                        )
                    )
                all_groups.append(
                    TrajectoryGroup(
                        trajectories=trajectories,
                        final_rewards=[float(lk) for lk in likelihoods],
                        discount_factor=self.discount_factor,
                    )
                )
                gen_records.append({
                    "batch_id": batch_id, "split": split, "sample_id": sample_id,
                    "timestep": t, "try_step": try_step, "advantage_mode": self.advantage_mode,
                    "target_action": x_t, "thoughts": thoughts,
                    "likelihoods": [float(x) for x in likelihoods],
                    "rewards": [float(x) for x in rewards],
                    "advantages": [float(x) for x in advantages],
                    "thought_tokens": [int(n) for n in thought_lens], "best": int(best),
                })

                if self.verbose:
                    logger.info(
                        "sample %s step %d/%d: best p(x|s,z)=%.4f, reward=%+.4f, |z|=%d tok "
                        "(group mean |z|=%.0f)",
                        sample_id,
                        t + 1,
                        len(action_indices),
                        likelihoods[best],
                        rewards[best],
                        thought_lens[best],
                        float(np.mean(thought_lens)),
                    )

        # Advantages are pre-set; compute_advantages() returns them unchanged, then
        # we persist the groups to the replay buffer and (optionally) log generations.
        for group in all_groups:
            group.compute_advantages()
            self.replay_buffer.add_trajectory_group(group)
        self.replay_buffer.update_buffer_ds_and_df()
        self._write_generations(gen_records)

        self.hf_tokenizer.padding_side = og_padding_side
        if was_training:
            self.llm.model.train()
        return {"policy": all_groups}
