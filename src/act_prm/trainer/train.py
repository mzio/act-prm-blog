"""
Training and evaluation functions
"""

# from copy import deepcopy
import contextlib
import json
import os
from typing import Any, Callable

import numpy as np
import torch
from datasets import Dataset as HFDataset
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase

from ..environments import Environment
from ..environments.act_prm_traces.data import action_start_token
from ..generator.huggingface.base import HuggingFaceGenerator
from ..llm_handlers.huggingface import HuggingFaceLLM
from ..replay_buffer.types import Trajectory, TrajectoryGroup
from .data import DataCollatorForPolicyGradient


def run_rollouts(
    llm: HuggingFaceLLM,
    hf_tokenizer: PreTrainedTokenizerBase,
    generator_constructor: Callable[..., HuggingFaceGenerator],
    env: Environment,
    cfg: DictConfig,
    batch_id: int,
    checkpoint_name: str | None = None,
    split: str = "train",
    num_tries: int = 1,
    start_idx: int = 0,
    tasks_per_update: int | None = None,  # i.e., batch_size
    name_or_identifier: str | None = None,
    # Overrides for generation
    max_tokens: int | None = None,
    temperature: float | None = None,
    pbar_position: int = 0,
    num_return_sequences: int | None = None,
) -> tuple[dict[str, Any], dict[str, list[Trajectory]]]:
    """
    Run rollouts for a single batch, e.g., by generating rollouts and grading them

    Returns:
    - final_metrics: Metrics for the batch, keyed by "{split}/{try_idx}/{metric}"
    - new_trajectories: Trajectories for the batch, keyed by an identifier (default "policy")
    """
    num_tries = num_tries or 1  # guard: eval/train configs may leave (eval_)num_tries unset
    _has_torch_model = hasattr(llm, "model") and hasattr(getattr(llm, "model", None), "training")
    if _has_torch_model:
        was_training = llm.model.training
    else:
        was_training = False

    ctx = torch.no_grad() if _has_torch_model else contextlib.nullcontext()
    with ctx:
        if _has_torch_model:
            llm.model.eval()
        env.split = split  # Select task split

        generator = generator_constructor(
            llm=llm,
            hf_tokenizer=hf_tokenizer,
            env=env,
            cfg=cfg,
            enable_thinking=cfg.get("enable_thinking", False),
            name_or_identifier=name_or_identifier,
        )
        batch_size = tasks_per_update or len(env)  # len(env) is the number of tasks or problems
        num_return_sequences = num_return_sequences or (
            cfg.group_size if split == "train" else cfg.eval_group_size
        )
        all_eval_metrics = {}
        keys_for_correct = []
        eval_metric_keys = [
            "final_reward",
            "first_return",
            "action_prob",
            "last_state_len",
            "timesteps",
            "correct",
            "match_rate",
            "total",
        ]
        # Store new trajectories to return
        new_trajectories: dict[str, list[Trajectory]] = {}
        all_trajectory_groups: list[dict[str, list[TrajectoryGroup]]] = []

        for try_idx in range(num_tries):
            pbar_desc = f"Generating {num_return_sequences} rollouts for sample {start_idx + 1} / {start_idx + batch_size}"
            sample_pbar = tqdm(
                range(start_idx, start_idx + batch_size),
                desc=pbar_desc,
                colour="blue",
                leave=True,
                position=pbar_position,
            )
            n_success_so_far = 0
            n_total_so_far = 0
            for _, sample_id in enumerate(sample_pbar):
                group_dict = generator.do_group_rollout(
                    env=env,
                    sample_id=sample_id,
                    batch_id=batch_id,
                    split=split,
                    try_step=try_idx,
                    num_return_sequences=num_return_sequences,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    pbar_position=pbar_position + 1,
                )
                all_trajectory_groups.append(group_dict)
                # Tally successes for the live success/total pbar postfix.
                # Use the primary trajectory key ("policy" if present, else the
                # first one returned). "correct" is an int/bool on Trajectory.
                _primary_key = "policy" if "policy" in group_dict else next(iter(group_dict), None)
                if _primary_key is not None:
                    for _tg in group_dict.get(_primary_key, []) or []:
                        for _traj in getattr(_tg, "trajectories", []) or []:
                            n_total_so_far += 1
                            if int(getattr(_traj, "correct", 0) or 0) >= 1:
                                n_success_so_far += 1
                pbar_desc = f"Generating {num_return_sequences} rollouts for sample {start_idx + 1} / {start_idx + batch_size}"
                sample_pbar.set_description(pbar_desc)
                sample_pbar.set_postfix(success=f"{n_success_so_far}/{n_total_so_far}")
            # End-of-try checkpoint: push the running rollouts buffer
            # to the hub (covers every rollout collected for try_idx).
            try:
                generator._maybe_save_rollouts()
            except Exception:
                pass  # already best-effort inside _maybe_save_rollouts

        # Save metrics and samples
        trajectory_keys = all_trajectory_groups[0].keys()
        _metric_prefix = f"{checkpoint_name}_{split}" if checkpoint_name is not None else split

        for _key in trajectory_keys:
            for trajectory_groups in all_trajectory_groups:  # list of list of trajectory groups
                for traj_group in trajectory_groups[_key]:  # len(trajectory_groups) usually 1,
                    for trajectory in traj_group.trajectories:  # can be >1, e.g., if step-wise adv
                        if _key == "policy":
                            # Only store metrics for the default "policy" trajectory group
                            _try_step = trajectory.try_step
                            for metric_key in eval_metric_keys:
                                _metric_key = f"{_metric_prefix}/try_{_try_step}/{metric_key}"
                                if metric_key == "correct":
                                    keys_for_correct.append(_metric_key)
                                if _metric_key not in all_eval_metrics:
                                    all_eval_metrics[_metric_key] = []
                                val = getattr(trajectory, metric_key, 1)  # 1 for total samples
                                all_eval_metrics[_metric_key].append(val)
                            # Also log flat env metrics surfaced on the trajectory
                            # (eval env: task_completion, task_personalization, num_respond_user).
                            for _mk, _mv in getattr(trajectory, "metrics", {}).items():
                                all_eval_metrics.setdefault(f"{_metric_prefix}/try_{_try_step}/{_mk}", []).append(_mv)
                        # Add trajectory to list of new trajectories
                        if _key not in new_trajectories:
                            new_trajectories[_key] = []
                        new_trajectories[_key].append(trajectory)

    final_metrics = {}  # return these metrics for the batch
    # 1. Compute aggregate metrics
    for k, v in all_eval_metrics.items():
        if "correct" in k or "total" in k:
            final_metrics[k] = np.sum(v).item()  # convert to float for json.dumps
        else:
            final_metrics[k] = np.mean(v).item()
        final_metrics[f"{k}_std"] = np.std(v).item()
        final_metrics[f"{k}_max"] = np.max(v).item()
    # 2. Add accuracy (dummy for Act-PRM training rollouts)
    for k in keys_for_correct:
        total_v = final_metrics[k.replace("correct", "total")]
        final_metrics[k.replace("correct", "accuracy")] = final_metrics[k] / total_v

    # 3. Add generator usage metrics (token counts)
    if hasattr(generator, "get_usage_metrics"):
        usage = generator.get_usage_metrics()
        for k, v in usage.items():
            final_metrics[f"usage/{k}"] = v

    if _has_torch_model and was_training:
        llm.model.train()

    return final_metrics, new_trajectories


def run_batch_rollouts(
    llm: HuggingFaceLLM,
    hf_tokenizer: PreTrainedTokenizerBase,
    generator_constructor: Callable[..., HuggingFaceGenerator],
    env: Environment,
    cfg: DictConfig,
    batch_id: int,
    checkpoint_name: str | None = None,
    split: str = "train",
    num_tries: int = 1,
    start_idx: int = 0,
    tasks_per_update: int | None = None,  # i.e., batch_size
    name_or_identifier: str | None = None,
    # Overrides for generation
    max_tokens: int | None = None,
    temperature: float | None = None,
    pbar_position: int = 0,
    num_return_sequences: int | None = None,
) -> tuple[dict[str, Any], dict[str, list[Trajectory]]]:
    """Sample-batched variant of :func:`run_rollouts`.

    Same signature and metric aggregation, but inside the per-try loop calls
    ``generator.do_batch_group_rollout(sample_ids=[start_idx..start_idx+B], ...)``
    once with the full sample list -- one ``model.generate`` batch per try
    instead of one per (sample, try) pair.

    Cross-try retrieval semantics are preserved because the outer ``try_idx``
    loop is still sequential: try ``t``'s rollouts and replay-buffer writes
    complete before try ``t+1``'s rollouts begin.

    Returns the same shape as :func:`run_rollouts`.
    """
    num_tries = num_tries or 1  # guard: eval/train configs may leave (eval_)num_tries unset
    _has_torch_model = hasattr(llm, "model") and hasattr(getattr(llm, "model", None), "training")
    if _has_torch_model:
        was_training = llm.model.training
    else:
        was_training = False

    ctx = torch.no_grad() if _has_torch_model else contextlib.nullcontext()
    with ctx:
        if _has_torch_model:
            llm.model.eval()
        env.split = split

        generator = generator_constructor(
            llm=llm,
            hf_tokenizer=hf_tokenizer,
            env=env,
            cfg=cfg,
            enable_thinking=cfg.get("enable_thinking", False),
            name_or_identifier=name_or_identifier,
        )
        if not hasattr(generator, "do_batch_group_rollout"):
            raise TypeError(
                f"run_batch_rollouts requires generator with do_batch_group_rollout; "
                f"got {type(generator).__name__}. Use a BatchedHuggingFaceGenerator-derived "
                f"generator (e.g. --generator_config default_batched), or call run_rollouts."
            )

        batch_size = tasks_per_update or len(env)
        num_return_sequences = num_return_sequences or (
            cfg.group_size if split == "train" else cfg.eval_group_size
        )
        sample_ids = list(range(start_idx, start_idx + batch_size))

        all_eval_metrics: dict[str, list[Any]] = {}
        keys_for_correct: list[str] = []
        eval_metric_keys = [
            "final_reward",
            "first_return",
            "action_prob",
            "last_state_len",
            "timesteps",
            "correct",
            "match_rate",
            "total",
        ]
        new_trajectories: dict[str, list[Trajectory]] = {}
        all_trajectory_groups: list[dict[str, list[TrajectoryGroup]]] = []

        for try_idx in range(num_tries):
            pbar_desc = (
                f"Batched: {len(sample_ids)} samples x {num_return_sequences} gens, "
                f"try {try_idx}/{num_tries - 1}, batch {batch_id}"
            )
            try_pbar = tqdm(
                total=1,
                desc=pbar_desc,
                colour="blue",
                leave=True,
                position=pbar_position,
            )
            all_trajectory_groups.append(
                generator.do_batch_group_rollout(
                    sample_ids=sample_ids,
                    batch_id=batch_id,
                    env=env,
                    split=split,
                    try_step=try_idx,
                    num_return_sequences=num_return_sequences,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    pbar_position=pbar_position + 1,
                )
            )
            try_pbar.update(1)
            try_pbar.set_description(pbar_desc + " -- done")
            # End-of-try checkpoint: push the running rollouts buffer
            # to the hub (covers every rollout collected for try_idx).
            try:
                generator._maybe_save_rollouts()
            except Exception:
                pass  # already best-effort inside _maybe_save_rollouts

        # Same metrics aggregation as run_rollouts: each entry of
        # all_trajectory_groups is a dict[str, list[TrajectoryGroup]] -- here
        # the inner list has len == num_samples (vs len 1 for run_rollouts).
        trajectory_keys = all_trajectory_groups[0].keys()
        _metric_prefix = f"{checkpoint_name}_{split}" if checkpoint_name is not None else split

        for _key in trajectory_keys:
            for trajectory_groups in all_trajectory_groups:
                for traj_group in trajectory_groups[_key]:
                    for trajectory in traj_group.trajectories:
                        if _key == "policy":
                            _try_step = trajectory.try_step
                            for metric_key in eval_metric_keys:
                                _metric_key = f"{_metric_prefix}/try_{_try_step}/{metric_key}"
                                if metric_key == "correct":
                                    keys_for_correct.append(_metric_key)
                                if _metric_key not in all_eval_metrics:
                                    all_eval_metrics[_metric_key] = []
                                val = getattr(trajectory, metric_key, 1)
                                all_eval_metrics[_metric_key].append(val)
                            # Also log flat env metrics surfaced on the trajectory
                            # (eval env: task_completion, task_personalization, num_respond_user).
                            for _mk, _mv in getattr(trajectory, "metrics", {}).items():
                                all_eval_metrics.setdefault(f"{_metric_prefix}/try_{_try_step}/{_mk}", []).append(_mv)
                        if _key not in new_trajectories:
                            new_trajectories[_key] = []
                        new_trajectories[_key].append(trajectory)

    final_metrics: dict[str, Any] = {}
    for k, v in all_eval_metrics.items():
        if "correct" in k or "total" in k:
            final_metrics[k] = np.sum(v).item()
        else:
            final_metrics[k] = np.mean(v).item()
        final_metrics[f"{k}_std"] = np.std(v).item()
        final_metrics[f"{k}_max"] = np.max(v).item()
    for k in keys_for_correct:
        total_v = final_metrics[k.replace("correct", "total")]
        final_metrics[k.replace("correct", "accuracy")] = final_metrics[k] / total_v

    if hasattr(generator, "get_usage_metrics"):
        usage = generator.get_usage_metrics()
        for k, v in usage.items():
            final_metrics[f"usage/{k}"] = v

    if _has_torch_model and was_training:
        llm.model.train()

    return final_metrics, new_trajectories


def prepare_minibatch(
    new_trajectories: list[Trajectory],
    hf_tokenizer: PreTrainedTokenizerBase,
    batch_size: int = 2,
    batch_idx: int = 0,  # for debugging
    max_seq_len: int = 32768,
    drop_zero_advantage: bool = False,
    **dataloader_kwargs: Any,
) -> tuple[DataLoader, dict[str, Any]]:
    """
    Convert a minibatch of trajectories to a PyTorch Dataloader.

    ``drop_zero_advantage``: when True, episode steps whose advantage is 0 are
    excluded from the batch entirely (not just weighted to 0). This matters for
    modes like ``best`` / ``top_half`` where most of the group is 0 — otherwise
    those samples still run a forward/backward (wasted compute) AND inflate
    ``len(train_loader)`` → ``gradient_accumulation_steps``, diluting the update
    on the samples that actually carry signal. Kept off by default (uniform / em /
    grpo want all samples).
    """
    metrics = {}
    n_skipped = 0
    n_zero_adv = 0
    # Optional correctness dump: when STRL_VERIFY_DUMP=<path> is set, record, for the first
    # few trainable steps, the exact tokens/logprobs that become the supervised target so the
    # alignment (supervised tokens == action tokens; old_logprobs match them) can be checked
    # offline against a teacher-forced recompute. No-op unless the env var is set.
    _verify_path = os.environ.get("STRL_VERIFY_DUMP")
    _verify_records: list[dict[str, Any]] = []
    # Assemble training data
    data_dict: list[dict[str, list[float | int]]] = []
    for trajectory in new_trajectories:
        for episode_step in trajectory.episode_steps:
            if episode_step.is_train:
                # Drop zero-advantage samples entirely (best/top_half): they carry no
                # gradient but would still waste compute and dilute the update.
                if drop_zero_advantage and abs(episode_step.advantage) < 1e-12:
                    n_zero_adv += 1
                    continue
                sa_input_ids = episode_step.state_action_tokens
                # Skip stub / inference-only steps that carry no trainable tokens
                # (e.g. API-policy generators that don't compute Qwen logprobs).
                # The trainer's loss multiplies by an empty action span -> no-op,
                # but the downstream length-check assert would crash, so skip here.
                if not sa_input_ids:
                    continue
                # Skip sequences that exceed max_seq_len
                if max_seq_len is not None and len(sa_input_ids) > max_seq_len:
                    n_skipped += 1
                    continue
                act_logprobs = (
                    episode_step.old_logprobs
                )  # 1. "full" action len, first token counts (maybe not)
                # input_tokens = sa_input_ids[:-1]
                # target_tokens = sa_input_ids[1:]
                state_len = episode_step.state_len
                target_state_len = episode_step.state_len - 1
                # 2. ^So target_state_len + len(act_logprobs) == len(sa_input_ids),  (maybe not)
                #    a bit different from Tinker where they don't predict first action token.
                padded_logprobs = [0.0] * target_state_len + act_logprobs
                adv = episode_step.advantage
                padded_advantages = [0.0] * target_state_len + [adv] * len(act_logprobs)
                padded_mask = [0] * target_state_len + [1] * len(act_logprobs)
                # Add labels to double-check
                sa_labels = [-100] * state_len + sa_input_ids[state_len:]
                # sa_labels = [-100] * target_state_len + sa_input_ids[target_state_len:]

                # print(f"batch_idx: {batch_idx}")
                # print("hf_tokenizer.decode(sa_input_ids[-len(act_logprobs):])")
                # print(hf_tokenizer.decode(sa_input_ids[-len(act_logprobs):]))
                # print("-" * 100)
                # print("hf_tokenizer.decode(sa_input_ids[target_state_len:])")
                # print(hf_tokenizer.decode(sa_input_ids[target_state_len:]))
                # print("-" * 100)
                # Drop episodes with zero-length actions or sa/state length
                # mismatches. Most common cause: generator emitted an empty
                # reflection / action (0 old_logprobs) AND the chat template
                # bookend tokens differ between the state-only render
                # (continue_final_message=True, leaving the assistant turn
                # open) and the state+empty-action render
                # (continue_final_message=False, which may collapse an empty
                # assistant turn entirely). Training on a 0-length action is
                # a no-op, so dropping is the right call.
                if len(act_logprobs) == 0 or len(sa_input_ids) - 1 != target_state_len + len(act_logprobs):
                    n_skipped += 1
                    continue

                # Hacky, but we pass logprobs = 0.0 if we want to ignore them
                if sum(padded_logprobs) == 0:
                    padded_logprobs = None

                # Action-only mask (parallel to label_mask): 1 ONLY on the explicit
                # action sub-span (<tool_call>…/Final Answer:), so the SFT trainer can
                # log train/actiononly_{ppl,accuracy} — using the SAME boundary as the
                # offline eval (action_start_token). Subset of label_mask. Best-effort:
                # never break training (fall back to the whole target span).
                try:
                    _content = episode_step.action.get("content") if isinstance(episode_step.action, dict) else None
                    _a_start = action_start_token(hf_tokenizer, sa_input_ids, state_len, _content)
                    _a_off = max(target_state_len, _a_start - 1)  # first action prediction position
                    padded_action_mask = [0] * _a_off + [1] * (len(sa_input_ids) - 1 - _a_off)
                except Exception:
                    padded_action_mask = list(padded_mask)

                data_dict.append(
                    {
                        "input_ids": sa_input_ids,
                        "attention_mask": [True] * len(sa_input_ids),
                        "advantages": padded_advantages,  # Note that advantages and logprobs are already
                        "logprobs": padded_logprobs,  # shifted to account for next-token prediction
                        "label_mask": padded_mask,
                        "action_mask": padded_action_mask,  # action sub-span (train-side action-only metrics)
                        "state_len": target_state_len,
                        "action_len": len(act_logprobs),
                        "labels": sa_labels,
                        "is_icl": episode_step.is_icl,
                    }
                )
                if _verify_path is not None and len(_verify_records) < 4:
                    _verify_records.append(
                        {
                            "sample_id": episode_step.sample_id,
                            "try_step": episode_step.try_step,
                            "user_id": episode_step.user_id,
                            "state_len": state_len,               # #state tokens (unshifted)
                            "action_len": len(act_logprobs),      # #action (reflection) tokens
                            "n_sa_tokens": len(sa_input_ids),     # state + action
                            "num_label_tokens": int(sum(padded_mask)),
                            "invariant_ok": len(sa_input_ids) - 1 == target_state_len + len(act_logprobs),
                            "action_token_ids": sa_input_ids[state_len:],
                            "action_text_decoded": hf_tokenizer.decode(sa_input_ids[state_len:]),
                            "reflection_text": episode_step.prior_reflection,
                            "old_logprobs": act_logprobs,
                            "advantage": adv,
                            "input_ids": sa_input_ids,            # full seq, for offline recompute
                        }
                    )

    if n_skipped > 0:
        print(
            f"WARNING: Skipped {n_skipped}/{n_skipped + len(data_dict)} episodes exceeding max_seq_len={max_seq_len}"
        )
    metrics["n_skipped_seq_len"] = n_skipped
    metrics["n_dropped_zero_advantage"] = n_zero_adv
    if _verify_path is not None:
        with open(_verify_path, "w") as _vf:
            json.dump(
                {"n_skipped": n_skipped, "n_train_steps": len(data_dict), "records": _verify_records},
                _vf,
            )
        print(f"[STRL_VERIFY] wrote {len(_verify_records)} trainable-step records to {_verify_path}")
    dataset = HFDataset.from_list(data_dict)
    collate_fn = DataCollatorForPolicyGradient(tokenizer=hf_tokenizer, return_tensors="pt")
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, **dataloader_kwargs)
    return dataloader, metrics  # empty metrics for now


def hide_observations(
    messages: list[dict[str, str]],
    hidden_obs_content: str = "...",
    first_obs_to_show: int = 2,  # e.g., to keep prompt
    last_obs_to_show: int = 1,  # e.g., to keep last observation
) -> list[dict[str, str]]:
    """
    Maybe hide past observations from messages
    """
    user_indices = [idx for idx, message in enumerate(messages) if message["role"] in ["user", "tool"]]
    last_message_idx = user_indices[-last_obs_to_show] if last_obs_to_show > 0 else len(messages)
    return [
        {"role": message["role"], "content": hidden_obs_content}
        if (message["role"] in ["user", "tool"] and (idx >= first_obs_to_show and idx < last_message_idx))
        else message
        for idx, message in enumerate(messages)
    ]
