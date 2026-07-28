"""
Argument parsing helpers
"""

import argparse
import logging
import os

from omegaconf import OmegaConf

from .setup import get_run_name

logger = logging.getLogger(__name__)


def get_args() -> argparse.Namespace:
    """
    Load and process experiment arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_name", type=str, default="act-prm")
    # Human-readable prefix prepended to the run_name (and thus the log/checkpoint
    # leaf dir), e.g. --run_tag retail_s1_policy_heldout. Purely cosmetic/organizational:
    # it does NOT change training, but makes checkpoints self-describing and lets two
    # runs that would otherwise collide (e.g. same env_config, different split_file)
    # land in distinct dirs. The full auto-encoded run_name still follows the tag.
    parser.add_argument("--run_tag", type=str, default=None)

    # Necessary arguments + configs (to load default args from)
    parser.add_argument("--is_async", action="store_true", help="Use asynchronous environment")
    parser.add_argument(
        "--resume_run",
        action="store_true",
        default=False,
        help="Resume from checkpoint in log_path",
    )

    parser.add_argument("--env_config", type=str, help="Environment config to load default args")
    parser.add_argument("--generator_config", type=str, help="Generator config; ditto")
    parser.add_argument("--trainer_config", type=str, help="Trainer config; ditto")
    parser.add_argument("--replay_buffer_config", type=str, help="Replay buffer config; ditto")

    # Optional arguments -> overrides the defaults in trainer_config if specified
    ## Model (specified in trainer_config)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--lora_rank", type=int)

    ## Tinker training-loop overrides (consumed by main_tinker.py)
    parser.add_argument("--num_batches", type=int, help="Override trainer_cfg.num_batches")
    parser.add_argument(
        "--no_train",
        action="store_true",
        default=False,
        help="Skip training; only run rollouts/eval",
    )

    ## Show-and-Tell RL generator arguments
    parser.add_argument(
        "--use_prior_advantage", action="store_true", help="Use prior advantage", default=None
    )
    parser.add_argument("--use_user_response", action="store_true", help="Use user response", default=None)
    parser.add_argument("--use_assistant_tip", action="store_true", help="Use assistant tip", default=None)
    parser.add_argument("--use_full_outcomes", action="store_true", help="Use full outcomes", default=None)
    parser.add_argument("--use_advantage_only", action="store_true", help="Use advantage only", default=None)
    parser.add_argument("--tokenized_context", action="store_true", help="Templated context", default=None)
    parser.add_argument("--use_last_try_only", action="store_true", help="Use last try only", default=None)
    parser.add_argument(
        "--same_sample_tries", action="store_true", help="Use same sample tries only", default=None
    )
    parser.add_argument(
        "--use_same_user_ids", action="store_true", default=None,
        help="Only retrieve memory from past rollouts with the same user_id/persona",
    )
    parser.add_argument("--match_on_timestep", action="store_true", help="Match on timestep", default=None)
    parser.add_argument(
        "--initial_step_only", action="store_true", help="Use initial step only", default=None
    )
    parser.add_argument("--match_on_thoughts", action="store_true", help="Match on thoughts", default=None)
    parser.add_argument(
        "--train_first_tries", action="store_true", help="Train on first tries only", default=None
    )
    parser.add_argument(
        "--train_icl_samples", action="store_true", help="Train on ICL copy samples", default=None
    )
    parser.add_argument(
        "--train_hindsight",
        action="store_true",
        help="Train on hindsight-relabeled samples (HER)",
        default=None,
    )
    parser.add_argument("--use_thinking_tags", action="store_true", help="Use thinking tags", default=None)
    parser.add_argument("--use_bad_steps_only", action="store_true", help="Use bad steps only", default=None)

    ## Show-Don't-Tell generator arguments (StrlHuggingFaceGenerator / query_strl).
    # Defaults are None so the generator YAML value wins unless the flag is passed.
    # Retrieval-based reflection
    parser.add_argument(
        "--reflect_prompt_names",
        type=str,
        nargs="+",
        default=None,
        help="Reflection strategies to generate over a retrieved step (e.g. summary credit tips)",
    )
    parser.add_argument(
        "--reflect_prompt_types",
        type=str,
        nargs="+",
        default=None,
        help="Reflection granularity per reflect_prompt_name (e.g. trajectory step)",
    )
    parser.add_argument(
        "--retrieve_on_first_try",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Retrieve + reflect on try 0 too (e.g. with a preloaded replay buffer)",
    )
    parser.add_argument(
        "--retrieval_top_k", type=int, default=None, help="Top-k candidates for BM25/SBERT retrieval"
    )
    # Step-wise summarization / compaction
    parser.add_argument("--summary_prompt_name", type=str, default=None, help="Summary prompt template name")
    parser.add_argument(
        "--use_summary_state",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Compact the running rollout into a handoff summary before each action",
    )
    parser.add_argument(
        "--use_last_next_obs",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Summarize up to (but excluding) the last tool observation",
    )
    # LLM-as-a-judge action gating
    parser.add_argument("--judge_prompt_name", type=str, default=None, help="Self-judge prompt template name")
    parser.add_argument(
        "--sample_with_judge",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Filter sampled actions with an LLM-as-a-judge (re-sample on reject)",
    )
    parser.add_argument(
        "--max_action_tries", type=int, default=None, help="Max judge-gated action re-samples"
    )
    parser.add_argument(
        "--use_local_quality",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Judge gate on the action's local quality verdict",
    )
    parser.add_argument(
        "--use_follows_reflection",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Judge gate on whether the action follows the recalled reflection",
    )

    # Reflection prompts
    parser.add_argument("--reflection_system_prompt", type=str, help="Reflection system prompt", default=None)
    parser.add_argument(
        "--reflection_user_prompt_trajectory",
        type=str,
        help="Reflection user prompt trajectory",
        default=None,
    )
    parser.add_argument(
        "--reflection_user_prompt_step", type=str, help="Reflection user prompt step", default=None
    )
    parser.add_argument(
        "--keep_first_reflect", action="store_true", help="Keep first reflection", default=None
    )

    # PGIC reward shape + reflection-generation knobs (consumed by
    # pgic_step / pgic_traj generators). Defaults are None so the
    # corresponding YAML value wins unless the flag is passed.
    parser.add_argument(
        "--reward_form",
        type=str,
        default=None,
        choices=["linear", "logsig"],
        help="PGIC reward form: ``linear`` (R = z * Delta_c) or ``logsig`` (R = log sigma(beta * (z * Delta_c - gamma)))",
    )
    parser.add_argument(
        "--reward_beta",
        type=float,
        default=None,
        help="PGIC logsig sharpness (beta). Notebook recipe: 1-5",
    )
    parser.add_argument(
        "--reward_gamma",
        type=float,
        default=None,
        help="PGIC logsig log-margin (gamma). 0 to start; log(1.5) ~= 0.405 for sharper margin",
    )
    parser.add_argument(
        "--length_normalize_per_step",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="pgic_traj only: per-step token-mean before summing the trajectory-level Delta_c. Pass --no-length_normalize_per_step for raw token-sum.",
    )
    parser.add_argument(
        "--max_reflection_tokens",
        type=int,
        default=None,
        help="PGIC: max tokens per generated reflection",
    )
    parser.add_argument(
        "--reflection_temperature",
        type=float,
        default=None,
        help="PGIC: temperature for reflection sampling. None = reuse cfg.temperature",
    )
    parser.add_argument(
        "--match_prior_tries",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="PGIC phase-1 retrieval scope. True: BM25 only over rows with the same (split, sample_id, generation_id) -- this (sample, gen)'s prior tries. False: split-only filter (cross-sample retrieval). Defaults to YAML value when unset.",
    )

    parser.add_argument("--self_distill_only", action="store_true", help="Self-distill only")
    # Synthetic study
    parser.add_argument(
        "--ground_truth_generation", action="store_true", help="Generate ground-truth actions at each step"
    )
    parser.add_argument(
        "--ground_truth_incorrects",
        action="store_true",
        help="Add incorrect answers to ground-truth generation",
    )

    ## PyTorch training arguments -> orthogonal to Tinker arguments
    parser.add_argument("--model_config", type=str, help="Model config; ditto")
    parser.add_argument("--lora_config", type=str, help="LoRA config; ditto")
    parser.add_argument(
        "--lora_checkpoint_path",
        type=str,
        default="./checkpoints_lora",
        help="Path to save and load LoRA checkpoints",
    )
    parser.add_argument(
        "--dataloader_batch_size",
        type=int,
        help="Batch size for dataloader",
    )
    parser.add_argument(
        "--fp32_loss",
        action="store_true",
        default=None,
        help="Cast logits to float32 before computing cross-entropy loss",
    )
    parser.add_argument(
        "--no_initial_eval",
        action="store_true",
        default=None,
        help="Don't evaluate on the first step",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        help="Number of steps to accumulate gradients before updating the model. Should be <= mini_batch_size",
    )
    parser.add_argument("--num_steps", type=int, help="Number of steps to train for")
    parser.add_argument(
        "--max_input_id_len",
        type=int,
        help="Max number of tokens for training samples (if we're GPU poor)",
    )
    parser.add_argument(
        "--max_sample_ids",
        type=int,
        help="Max number of samples to load for Act-PRM SFT datasets",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Enable gradient checkpointing on the LoRA-wrapped model. Trades ~30%% extra compute for ~6-8x activation memory savings at long context. Required: also calls enable_input_require_grads() so LoRA gradients still flow through the frozen base.",
    )
    parser.add_argument(
        "--no_fa2",
        action="store_true",
        help="If True, disable Flash Attention 2 and use SDPA instead",
    )
    parser.add_argument("--max_seq_len", type=int, help="Max sequence length for training samples")
    parser.add_argument(
        "--drop_zero_advantage",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Drop zero-advantage samples from the training batch (recommended for best/top_half SFT)",
    )

    ## Environment
    parser.add_argument(
        "--actions_only",
        action="store_true",
        default=None,
        help="If True, remove thoughts / reasoning traces from observed assistant messages",
    )

    ## Act-PRM environment + generator overrides (override the corresponding YAML key when passed)
    parser.add_argument("--dataset", type=str, default=None, help="Act-PRM traces dataset (HF id, or 'synthetic')")
    parser.add_argument("--dataset_path", type=str, default=None, help="Local dir to persist/reuse processed trajectory pools (first run streams+saves; later runs load offline)")
    parser.add_argument("--num_trajectories", type=int, help="Number of logged train trajectories to load")
    parser.add_argument("--eval_trajectories", type=int, help="Number of held-out eval trajectories to load")
    parser.add_argument("--max_steps_per_traj", type=int, help="Max logged action-steps to infer thoughts for per trajectory")
    parser.add_argument("--max_traj_timestep", type=int, help="Only keep logged trajectories that finished within this many steps")
    parser.add_argument("--obs_max_chars", type=int, help="Max chars per observation before truncation")
    parser.add_argument("--first_obs_to_show", type=int, help="Act-PRM state compaction: keep the first N observations (initial user prompt)")
    parser.add_argument("--last_obs_to_show", type=int, help="Act-PRM state compaction: keep the last N observations (most recent tool/user response)")
    parser.add_argument(
        "--synthetic",
        action="store_true",
        default=None,
        help="Use the built-in synthetic action-only trajectories (offline; no HF download)",
    )
    parser.add_argument(
        "--keep_expert_thoughts",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Keep each assistant turn's ORIGINAL expert reasoning+action as the (SFT target) "
        "content instead of the action-only span (the 'expert thought-action' SFT dataset). "
        "Use a distinct --dataset_path since processed pools are cached.",
    )
    parser.add_argument("--length_penalty", type=float, default=None, help="Act-PRM thought length penalty (lambda)")
    parser.add_argument("--max_thought_tokens", type=int, help="Act-PRM max thought tokens (length-penalty budget)")
    parser.add_argument(
        "--reward_method",
        type=str,
        default=None,
        choices=["penalty", "lift"],
        help="Act-PRM reward: 'penalty' (p(x|s,z) - lambda*len_frac) or 'lift' (per-token likelihood + lexicographic length selection)",
    )
    parser.add_argument(
        "--advantage_mode",
        type=str,
        default=None,
        choices=["em", "best", "top_half", "uniform", "grpo"],
        help="How per-candidate rewards become advantages (em=EM weights, best=argmax, top_half, uniform, grpo=mean-centered)",
    )
    parser.add_argument(
        "--grpo_normalize",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="advantage_mode=grpo: divide mean-centered reward by group std",
    )
    parser.add_argument(
        "--infer_thoughts",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Infer thoughts (default). --no-infer_thoughts runs the actions-only SFT baseline",
    )
    parser.add_argument(
        "--score_with_base",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Score p(x|s,z) with the frozen base model (LoRA disabled) instead of the policy",
    )
    parser.add_argument(
        "--save_generations",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Append every generation group to <log_path>/generations.jsonl",
    )
    parser.add_argument(
        "--hide_observations",
        action="store_true",
        default=None,
        help="If True, hide observations prior to the last one, e.g., for more 'human-like' context",
    )
    parser.add_argument(
        "--loss_on_past_assistant_messages",
        action="store_true",
        default=None,
        help="If True, compute SFT loss on all assistant messages, not just the last one",
    )

    parser.add_argument("--num_train_samples", type=int, help="Number of samples to train on")
    parser.add_argument("--num_eval_samples", type=int, help="Number of samples to evaluate on")
    parser.add_argument("--num_test_samples", type=int, help="Number of samples to test on")
    parser.add_argument("--num_train_tasks", type=int, help="Number of samples to train on")
    parser.add_argument("--num_val_tasks", type=int, help="Number of samples to evaluate on")
    parser.add_argument("--num_test_tasks", type=int, help="Number of samples to test on")

    ## Evaluation / Eval Environment
    parser.add_argument(
        "--eval_env_config",
        type=str,
        help=(
            "Evaluation environment config. If unspecified, defaults to env_config "
            "(but we use different splits for evaluation, i.e., from the 'eval' split)"
        ),
    )
    parser.add_argument(
        "--base_env_config",
        type=str,
        help="For ActPrmEnvWithBaseEnv, the environment we use for taking given actions",
    )
    parser.add_argument(
        "--best_metric",
        type=str,
        help="Metric to save best checkpoints on",
    )

    ## Tinker logging + checkpointing
    parser.add_argument("--base_url", type=str, help="Tinker base URL")
    parser.add_argument(
        "--log_path",
        type=str,
        default="./logs",
        help=(
            "Parent directory for logging and saving Tinker checkpoints. Actual path is "
            "automatically determined (and created) based on our specified argparse args"
        ),
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="./checkpoints",
        help=(
            "Parent directory for saving other checkpoints and data (e.g., replay buffer samples)."
            " Similar to above, actual path is automatically determined (and created)"
        ),
    )
    parser.add_argument("--load_checkpoint_path", type=str, help="Path to load Tinker checkpoint")
    parser.add_argument(
        "--resume_from", type=str, default=None,
        help="Warm-start LoRA from a saved checkpoint at launch: a dir with adapter_model.safetensors, "
             "or a run's checkpoint dir (auto-picks step_last, else step_best).",
    )

    ## Number of tries we allow to solve each task
    parser.add_argument(
        "--num_tries",
        type=int,
        help="Number of tries to solve each task; will override if specified",
    )
    parser.add_argument("--eval_num_tries", type=int, help="num_tries during evaluation")
    ## Number of turns to override in environment
    parser.add_argument("--max_turns", type=int, help="Number of steps to solve each task")
    parser.add_argument(
        "--num_return_sequences",
        type=int,
        help="Chunk size for generation (default: group_size). Lower to reduce GPU memory.",
    )

    ## Model Generation
    parser.add_argument("--max_tokens", type=int)
    parser.add_argument("--temperature", type=float)

    ## Training Rollouts
    parser.add_argument(
        "--mean_center",
        action="store_true",
        default=None,
        help="Mean-center rewards",
    )
    parser.add_argument("--discount_factor", type=float)
    parser.add_argument(
        "--group_size",
        type=int,
        help="Number of rollouts to generate per sample; will override if specified",
    )
    parser.add_argument(
        "--eval_group_size",
        type=int,
        help="Group size for evaluation; will override if specified. Set >1 if we want error bars",
    )
    parser.add_argument(
        "--samples_per_task",
        type=int,
        help="Alias for `group_size`; will override `group_size` if specified",
    )
    parser.add_argument(
        "--eval_samples_per_task",
        type=int,
        help="Alias for `eval_group_size`; will override `eval_group_size` if specified",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Number of unique tasks or problems used for training in each training step",
    )
    parser.add_argument(
        "--tasks_per_update",
        type=int,
        help="Alias for `batch_size`; will override `batch_size` if specified",
    )

    ## Training Updates
    parser.add_argument("--advantage_threshold", type=float)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--kl_penalty_coef", type=float)
    parser.add_argument("--kl_discount_factor", type=float)
    parser.add_argument(
        "--num_substeps",
        type=int,
        help=(
            "Number of actual updates per training step. "
            "Splits total_episode_steps = batch_size * group_size * len(traejectory) into "
            "`num_substeps` mini-batches, where each mini-batch is used for one optimizer update. "
            "By default, we adjust total_episode_steps to be a multiple of `num_substeps`."
        ),
    )
    parser.add_argument(
        "--mini_batch_size",
        type=int,
        help=(
            "Alternative to --num_substeps; size of each mini-batch during updates. "
            "If specified and num_substeps is None, then we set num_substeps = "
            "total_episode_steps // mini_batch_size. If both specified, then we (super)sample the "
            "training data s.t. len(training_data) = mini_batch_size * num_substeps."
        ),
    )

    ## More Evaluation
    parser.add_argument("--eval_every", type=int, help="Iters to evaluate, 0 = disabled")
    parser.add_argument("--early_stop_patience", type=int, help="Stop if eval best_metric doesn't improve for N consecutive evals (0=off)")
    parser.add_argument("--eval_gen_every", type=int, help="Iters to evaluate generation, 0 = disabled")
    parser.add_argument(
        "--eval_rollout_every",
        type=int,
        help="Iters to evaluate rollouts, 0 = disabled",
    )
    parser.add_argument(
        "--eval_rollout_start",
        type=int,
        help="Number of batches before doing rollout evaluation",
    )
    parser.add_argument(
        "--num_eval_gen_samples",
        type=int,
        help="Number of samples to evaluate generation",
    )
    parser.add_argument(
        "--num_eval_rollout_samples",
        type=int,
        help="Number of samples to evaluate rollouts",
    )

    ## Miscellaneous
    parser.add_argument("--save_every", type=int, help="Iters to save checkpoint, 0 = disabled")
    parser.add_argument("--save_rollouts_every", type=int, help="Iters to save rollouts, 0 = disabled")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--data_seed",
        type=int,
        default=None,
        help="Seed for data splitting (train/test). Defaults to --seed if not set.",
    )
    parser.add_argument("--replicate", type=str, default="0", help="Unique identifier for run")
    parser.add_argument("--verbose", action="store_true", default=False, help="Extra details")
    parser.add_argument(
        "--verbose_retrieval", action="store_true", default=False, help="Extra details for retrieval"
    )
    parser.add_argument("--streamer", action="store_true", help="Stream generations (for PyTorch)")

    args = parser.parse_args()

    # Handle aliases
    if args.samples_per_task is not None:
        args.group_size = args.samples_per_task

    if args.eval_samples_per_task is not None:
        args.eval_group_size = args.eval_samples_per_task

    if args.tasks_per_update is not None:
        args.batch_size = args.tasks_per_update

    # For now, we always set this to number of eval tasks, i.e., len(eval_env)
    # if args.eval_tasks_per_update is not None:
    #     args.eval_batch_size = args.eval_tasks_per_update

    # Get run (i.e., experiment) name
    _ignore_args = [
        "base_url",
        "checkpoint_path",
        "log_path",
        "load_checkpoint_path",
        "lora_checkpoint_path",
        "resume_from",  # a long checkpoint path; must not go into the run name (filename too long)
        "project_name",
        "run_tag",  # prepended explicitly below; don't also auto-encode it
        "dataset_path",  # a long path; run_tag/env already identify the run (kept the name < 255)
        "verbose",
        "streamer",
    ]
    _ignore_args.extend([argn for argn in vars(args).keys() if argn.endswith("_every")])
    if args.base_env_config is not None and args.base_env_config == args.env_config:
        _ignore_args.append("base_env_config")
    args.run_name = get_run_name(args, prefix=args.project_name, ignore_args=_ignore_args)
    if args.run_tag:
        # Sanitize like get_run_name does, then prepend so the leaf dir is self-describing.
        _tag = str(args.run_tag).replace("-", "_").replace(".", "_").replace("/", "_")
        args.run_name = f"{_tag}-{args.run_name}"
    # Hard-cap the leaf dir component: a single path segment must stay < 255 bytes
    # (ext4 limit) or os.makedirs raises OSError Errno 36. Keep the readable prefix and
    # append a short hash so truncated names stay unique.
    if len(args.run_name) > 200:
        import hashlib
        _h = hashlib.md5(args.run_name.encode()).hexdigest()[:8]
        args.run_name = f"{args.run_name[:190]}-{_h}"
    logger.info("Run name: %s", args.run_name)

    # Setup log path and checkpoint / data-saving path
    # -> construct as args.log_path/args.env_config/model_name/args.run_name/
    # -> similar for checkpointing
    created_dir = False
    try:
        _model_name = OmegaConf.load(f"./configs/trainer/{args.trainer_config}.yaml")["model_name"]
    except Exception as e:
        print(f"{e.__class__.__name__}: {e}")
        try:
            _model_name = OmegaConf.load(f"./configs/model/{args.model_config}.yaml")["model_config"][
                "pretrained_model_name_or_path"
            ]
        except Exception as e:
            print(f"{e.__class__.__name__}: {e}")
            assert args.model_name, "args.model_name must be specified if not in trainer_config"
            _model_name = args.model_name
    _model_name = args.model_name or _model_name
    if _model_name is None:
        # Fallback: try model field from model config (for API models)
        try:
            _model_name = OmegaConf.load(f"./configs/model/{args.model_config}.yaml")["model_config"]["model"]
        except Exception:
            _model_name = args.model_config or "unknown_model"
    _model_name = _model_name.split("/")[-1].replace("-", "_")
    _env_config = args.env_config.replace("/", "_")

    for argname in ["log_path", "checkpoint_path", "lora_checkpoint_path"]:
        for new_dir in [_env_config, _model_name, args.run_name]:
            # setattr(args, argname, os.path.join(args.log_path, new_dir))
            try:
                setattr(args, argname, os.path.join(getattr(args, argname), new_dir))
            except Exception as e:
                _error_class = e.__class__.__name__
                print(f"{_error_class}: {e}")
                # headless-safe: fall back to a log_path-relative dir instead of pdb
                setattr(args, argname, os.path.join(args.log_path, new_dir))
            if not os.path.exists(getattr(args, argname)):
                os.makedirs(getattr(args, argname), exist_ok=True)
                created_dir = True
        if created_dir:
            logger.info("Created %s at: %s", argname, getattr(args, argname))
        else:
            logger.info("Using %s at: %s", argname, getattr(args, argname))

    # So Tinker doesn't load, delete checkpoints.jsonl at args.log_path if it exists
    if not args.resume_run and os.path.exists(os.path.join(args.log_path, "checkpoints.jsonl")):
        os.remove(os.path.join(args.log_path, "checkpoints.jsonl"))
        logger.info(
            "Deleted checkpoints.jsonl at: %s",
            os.path.join(args.log_path, "checkpoints.jsonl"),
        )

    # Setup tinker-cookbook WandB logging
    args.wandb_project = args.project_name
    args.wandb_name = args.run_name

    return args
