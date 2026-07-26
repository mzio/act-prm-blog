"""
Main script for training + evaluating Act-PRM HuggingFace (LoRA) policies in PyTorch.

Act-PRM is an offline EM over logged action-only demonstrations: the ``act_prm``
generator samples candidate thoughts ``z`` behind each logged action ``x`` in
state ``s`` (E-step), rewards them by the length-penalized action likelihood
``p(x | s, z)``, and this script hands the group-normalized weights to a
synchronous policy-gradient trainer (``act_prm.trainer.get_trainer``) for the
importance-weighted M-step over the (thought+action) tokens. LoRA is on by default
(``--lora_config``); any argparse flag whose name matches a config key overrides
that key (see ``update_configs``).

Example command (offline synthetic smoke test, Colab-friendly 0.6B model):
```bash
uv run python main_pytorch.py \
--env_config act_prm/snorkel_finance \
--model_config hf_qwen3_0_6b \
--lora_config r8_a16_linear \
--generator_config act_prm \
--trainer_config pg \
--replay_buffer_config default \
--synthetic --group_size 4 --batch_size 2 --max_steps_per_traj 2 \
--num_batches 2 --length_penalty 0.15 \
--learning_rate 4e-5 --num_substeps 1 \
--seed 42 --replicate 0 --verbose
```

Drop ``--synthetic`` (and set ``--num_trajectories``) to train on the real
Snorkel Agent Finance traces. The recommended default model is
``--model_config hf_qwen3_4b_instruct`` (Qwen3-4B-Instruct-2507, the paper's
model); ``hf_qwen3_8b`` and ``hf_qwen3_0_6b`` (tiny/CPU) are also provided.
"""

import argparse
import logging
import sys
from typing import Any

from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from rich import print as rich_print
from tinker_cookbook.utils import ml_log  # still use nice logging

from act_prm.environments import load_env
from act_prm.llm_handlers import load_llm
from act_prm.llm_handlers.huggingface import HuggingFaceLLM
from act_prm.lora import display_trainable_parameter_count, get_lora_model
from act_prm.replay_buffer import get_replay_buffer
from act_prm.trainer import get_optimizer, get_trainer
from act_prm.utils import get_args, print_config, seed_everything
from act_prm.utils.logging import AnsiColorLoggingFormatter

handler = logging.StreamHandler()
handler.setFormatter(AnsiColorLoggingFormatter())
logging.basicConfig(
    level=logging.DEBUG,
    handlers=[handler],
    force=True,
)
logger = logging.getLogger(__name__)


def update_configs(
    args: argparse.Namespace,
    *configs: DictConfig | None,
) -> tuple[DictConfig | None, ...]:
    """
    Update configs with any specified + applicable command-line arguments
    """
    # A bit heinous, but loop through all configs to update any applicable args
    for config in configs:
        if config is not None:
            for argname, argval in vars(args).items():
                if argval is not None and argname in config and argname != "model_config":
                    config[argname] = argval
    return configs


def get_param_color(param_name: str) -> str:
    if "self_attn" in param_name:
        return "yellow"
    elif "mlp" in param_name:
        return "cyan"
    return "white"


def main() -> None:
    """
    Main training function
    """
    # Initialize experiment
    args = get_args()
    seed_everything(args.seed)
    load_dotenv()  # Setup environment variables from .env file

    # Get default configs and experiment attributes
    args.generator_config = args.generator_config or "act_prm"
    args.replay_buffer_config = args.replay_buffer_config or "default"

    model_cfg = OmegaConf.load(f"./configs/model/{args.model_config}.yaml")
    lora_cfg = OmegaConf.load(f"./configs/lora/{args.lora_config}.yaml")
    env_cfg = OmegaConf.load(f"./configs/environments/{args.env_config}.yaml")
    generator_cfg = OmegaConf.load(f"./configs/generator/{args.generator_config}.yaml")
    trainer_cfg = OmegaConf.load(f"./configs/trainer/{args.trainer_config}.yaml")
    replay_buffer_cfg = OmegaConf.load(f"./configs/replay_buffer/{args.replay_buffer_config}.yaml")

    # Optional environment configs
    eval_env_cfg = env_cfg
    if args.eval_env_config is not None:
        eval_env_cfg = OmegaConf.load(f"./configs/environments/{args.eval_env_config}.yaml")
    # Update configs from args
    updated_cfgs = update_configs(
        args,
        model_cfg,
        lora_cfg,
        env_cfg,
        eval_env_cfg,
        generator_cfg,
        trainer_cfg,
        replay_buffer_cfg,
    )
    if args.verbose:
        cfg_names = ["model", "lora", "env", "eval_env", "generator", "trainer", "replay_buffer"]
        for cfg, cfg_name in zip(updated_cfgs, cfg_names):
            if cfg is not None:
                print_config(cfg, cfg_name.upper())
    # Get updated config variables
    (
        model_cfg,
        lora_cfg,
        env_cfg,
        eval_env_cfg,
        generator_cfg,
        trainer_cfg,
        replay_buffer_cfg,
    ) = updated_cfgs
    # Make env_cfg tokenizers consistent with llm.tokenizer
    env_cfg.pretrained_model_config = model_cfg.model_config
    eval_env_cfg.pretrained_model_config = model_cfg.model_config

    cfg = trainer_cfg  # Main config to reference (has all training attributes)
    cfg.run_name = args.run_name
    cfg.lora_checkpoint_path = args.lora_checkpoint_path
    # Surface a few argparse-only fields the trainer / dataset-card code reads
    for _k in ["env_config", "generator_config", "trainer_config", "seed", "replicate"]:
        if _k not in cfg and getattr(args, _k, None) is not None:
            cfg[_k] = getattr(args, _k)

    # Setup logging to WandB
    cfg_for_logger: dict[str, Any] = OmegaConf.to_container(cfg, resolve=True)
    # Make additional argparse args WandB-parseable and add to WandB
    cfg_for_logger.update({k: v for k, v in vars(args).items() if k not in cfg_for_logger and v is not None})
    for _cfg in [env_cfg, model_cfg, lora_cfg, generator_cfg, replay_buffer_cfg]:
        if _cfg is not None:
            cfg_for_logger.update(
                {k: v for k, v in _cfg.items() if k not in cfg_for_logger and v is not None}
            )
    ml_logger = ml_log.setup_logging(
        log_dir=cfg.log_path,
        wandb_project=cfg.wandb_project,
        wandb_name=cfg.wandb_name,
        config=cfg_for_logger,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("pylatexenc").setLevel(logging.WARNING)
    rich_print(
        "-> [bright_yellow]Saving LoRA checkpoints to [/bright_yellow]"
        f"[bright_blue]{cfg.lora_checkpoint_path}[/bright_blue]"
    )

    # Get LLM and attach LoRAs
    llm: HuggingFaceLLM = load_llm(**model_cfg)
    llm.model = get_lora_model(llm.model, **lora_cfg)
    # Optional warm-start: load a saved LoRA adapter at launch (crash recovery / continue training).
    # NOTE: this restores the POLICY WEIGHTS only; the step counter and replay buffer start fresh
    # (num_tries>=2 self-populates each task's buffer), so it's a warm-start, not an exact resume.
    if getattr(args, "resume_from", None):
        import os as _os
        from act_prm.lora import load_lora
        _rp = args.resume_from
        if not _os.path.isfile(_os.path.join(_rp, "adapter_model.safetensors")):
            for _sub in ("step_last", "step_best"):
                if _os.path.isfile(_os.path.join(_rp, _sub, "adapter_model.safetensors")):
                    _rp = _os.path.join(_rp, _sub)
                    break
        if not _os.path.isfile(_os.path.join(_rp, "adapter_model.safetensors")):
            raise SystemExit(f"--resume_from: no adapter_model.safetensors under {args.resume_from!r}")
        llm.model = load_lora(llm.model, _rp, is_trainable=True)
        logger.info("Warm-started LoRA weights from %s", _rp)
    if getattr(args, "gradient_checkpointing", False):
        # Recompute activations during backward instead of storing them.
        # use_reentrant=False is the modern API; the reentrant variant is
        # incompatible with some optimizers and will be removed.
        llm.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
        # CRITICAL with frozen base + LoRA: without this, the recomputed
        # forward never tracks gradients (embedding output is
        # requires_grad=False), so LoRA grads silently come out zero.
        llm.model.enable_input_require_grads()
        print("[main_pytorch] gradient_checkpointing enabled (use_reentrant=False)")
    optimizer = get_optimizer(llm.model, learning_rate=cfg.learning_rate)  # simple for now

    if args.verbose:  # Display trainable parameters
        _params_text = "Trainable parameters:\n"
        _param_names = [n for n, p in llm.model.named_parameters() if p.requires_grad]
        _params_text += "\n".join(
            f"├── [{get_param_color(n)}]{n}[/{get_param_color(n)}]" for n in _param_names
        )
        rich_print(_params_text)
    display_trainable_parameter_count(llm.model)

    # Get environment, replay buffer, and generator class
    env = load_env(**env_cfg)
    # Reuse env if eval_env not specified; we always specify the split for loading new tasks
    eval_env = load_env(**eval_env_cfg) if args.eval_env_config else env
    env.tokenizer = llm.tokenizer  # Make tokenizers consistent and available
    eval_env.tokenizer = llm.tokenizer

    # Make identifiers identifiable
    cfg.run_url = ml_logger.get_logger_url() if ml_logger is not None else None
    cfg.run_cmd = " ".join(sys.argv)
    # Add to environments
    all_envs = [env, eval_env]
    for attr in ["run_url", "run_cmd"]:
        for _idx, _env in enumerate(all_envs):
            if _env is not None:
                setattr(all_envs[_idx], attr, cfg.get(attr))

    # Set replay buffer
    replay_buffer = get_replay_buffer(hf_tokenizer=llm.tokenizer, **replay_buffer_cfg)

    # Training loop
    trainer = get_trainer(
        cfg.trainer_name,
        cfg=cfg,
        llm=llm,
        optimizer=optimizer,
        generator_cfg=generator_cfg,
        replay_buffer=replay_buffer,
        env=env,
        eval_env=eval_env,
        ml_logger=ml_logger,
        hf_tokenizer=llm.tokenizer,
        checkpoint_path=cfg.lora_checkpoint_path,
        log_path=cfg.log_path,
    )
    trainer.train()

    # Cleanup
    ml_logger.close()
    logger.info("Training completed successfully")


if __name__ == "__main__":
    main()
