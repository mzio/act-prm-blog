"""
Setup utilities
"""

import os
import random
import re
from argparse import Namespace

import numpy as np
import torch


def sanitize_for_hub(name: str, max_length: int = 96) -> str:
    """
    Sanitize a HF Hub repo id (`<org>/<repo>` or bare `<repo>`):

    - keep only `[A-Za-z0-9._-]`; collapse runs of disallowed chars (incl.
      `=` from get_run_name's argname=argval format, and `/` inside the repo
      segment) to a single `-`
    - dedupe `--` and `..`, trim leading/trailing `-` / `.`
    - cap to `max_length`, preserving the namespace (drops chars from the
      repo segment first)
    """
    def clean(s: str) -> str:
        s = re.sub(r"[^a-zA-Z0-9_\-.]+", "-", s)
        while "--" in s:
            s = s.replace("--", "-")
        while ".." in s:
            s = s.replace("..", ".")
        return s.strip("-.")

    if "/" in name:
        ns, repo = name.split("/", 1)
    else:
        ns, repo = None, name
    if ns:
        ns = clean(ns)
    repo = clean(repo)
    result = f"{ns}/{repo}" if ns else repo

    if len(result) > max_length:
        if ns is not None:
            avail = max_length - len(ns) - 1  # room for '/' + repo
            if avail < 1:
                return ns[:max_length].rstrip("-.")
            repo = repo[:avail].rstrip("-.")
            result = f"{ns}/{repo}"
        else:
            result = result[:max_length].rstrip("-.")
    return result


def seed_everything(seed: int) -> None:
    """Seed everything"""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_run_name(
    args: Namespace,
    prefix: str = "",
    ignore_args: list[str] | None = None,
) -> str:
    """Return run name"""
    run_name = prefix
    ignore_args = ignore_args or []

    for argname, argval in vars(args).items():
        if argval is None or argname in ignore_args:
            continue
        argn = "".join([c[:1] for c in argname.split("_")])
        # Remove hyphens and dots, e.g., --model_name gpt-4.1-nano-2025-04-14
        argval = str(argval).replace("-", "_").replace(".", "_").replace("/", "_")
        run_name += f"-{argn}={argval}"

    # Add checkpoint and logging path identifiers if specified
    for argname in ["load_checkpoint_path"]:  # maybe include "log_path" if specified
        if getattr(args, argname, None) is not None:
            argn = "".join([c[0] for c in argname.split("_")])
            ckpt_id = "_".join(
                [
                    "=".join(["".join([x[0] for x in c.split("_")]) for c in s.split("=")])
                    for s in args.log_path.split("/")[-1].split("-")
                ]
            )
            run_name += f"-{argn}={ckpt_id}"

    # Last cleanups (brevity, ensure no unintended directory separators)
    run_name = make_shorter(run_name.replace("/", "_"))
    return run_name


def make_shorter(run_name: str) -> str:
    """
    Apply various shortening heuristics to make run name shorter
    """
    run_name = run_name.replace("False", "0").replace("True", "1")
    run_name = run_name.replace("=textworld_", "=tw_")  # textworld
    run_name = run_name.replace("=Qwen_Qwen", "=Qwen")  # models
    run_name = run_name.replace("=Llama_Llam", "=Llama")  # models
    run_name = run_name.replace("-rebuco=default", "")  # if it's "default", don't need to specify
    run_name = run_name.replace("-sdo=0-gtg=0-gti=0", "")
    # Remove other toggle values set to 0
    # -upa=0-uur=0-uat=1-ufo=1-uao=0-tc=0-ulto=0-sst=1-mot=0-iso=0-mot=1-tft=0-tis=0-th=0
    run_name = run_name.replace("-upa=0", "").replace("-uur=0", "").replace("-uat=0", "")
    run_name = run_name.replace("-ufo=0", "").replace("-uao=0", "").replace("-tc=0", "")
    run_name = run_name.replace("-ulto=0", "").replace("-sst=0", "").replace("-mot=0", "")
    run_name = run_name.replace("-iso=0", "").replace("-mot=0", "").replace("-tft=0", "")
    run_name = run_name.replace("-tis=0", "").replace("-th=0", "").replace("-utt=0", "")
    run_name = run_name.replace("-ubso=0", "")
    run_name = run_name.replace("-mc=hf_qwen3_4b_inst_2507", "")
    return run_name
