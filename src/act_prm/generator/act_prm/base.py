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
        logits = self.llm.model(**model_inputs.to(device), use_cache=False).logits
        gen_logprobs, state_action_tokens = get_action_logprobs_and_state_action_tokens(
            logits=logits,
            state_lens=[state_len] * len(thoughts),
            **model_inputs.to(device),
        )
        del logits

        likelihoods: list[float] = []
        for g, n_action in enumerate(n_actions):
            action_lp = np.array(gen_logprobs[g][-n_action:], dtype=np.float64)
            likelihoods.append(float(np.exp(action_lp.mean())))  # in (0, 1]
        return likelihoods, gen_logprobs, state_action_tokens, state_len

    def _rewards_and_weights(
        self, likelihoods: list[float], thought_lens: list[int]
    ) -> tuple[list[float], np.ndarray, int]:
        """Compute per-candidate reward, EM weights, and the selected index."""
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
        weights = em_weights(rewards, likelihoods)
        return rewards, weights, best

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
        max_steps = self.max_steps_per_traj or getattr(env, "max_steps_per_traj", None) or len(action_indices)
        action_indices = action_indices[:max_steps]

        committed: list[str] = []
        all_groups: list[TrajectoryGroup] = []

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

                thoughts, thought_lens = self._sample_thoughts(
                    state, x_t, committed, group_size, max_thought_tokens, temperature
                )
                likelihoods, gen_logprobs, sa_tokens, state_len = self._score_thoughts(
                    system_prompt, state, thoughts, x_t
                )
                rewards, weights, best = self._rewards_and_weights(likelihoods, thought_lens)
                committed.append(thoughts[best])

                scoring_state = [{"role": "system", "content": system_prompt}] + [
                    m for m in state if m["role"] != "system"
                ]
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
                        # EM weight -> advantage, pre-computed so the trainer keeps it as-is
                        advantage=float(weights[g]),
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

        # Advantages are pre-set (EM weights); compute_advantages() returns them
        # unchanged, then we persist the groups to the replay buffer.
        for group in all_groups:
            group.compute_advantages()
            self.replay_buffer.add_trajectory_group(group)
        self.replay_buffer.update_buffer_ds_and_df()

        self.hf_tokenizer.padding_side = og_padding_side
        if was_training:
            self.llm.model.train()
        return {"policy": all_groups}
