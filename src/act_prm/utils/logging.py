"""
Basic logging helpers
"""

import logging
import time
from contextlib import contextmanager
from typing import Any

from omegaconf import DictConfig, ListConfig, OmegaConf
from rich import print as rich_print
from rich.syntax import Syntax
from rich.tree import Tree

logger = logging.getLogger(__name__)


def print_header(x, border="both") -> None:
    """
    Print with borders
    """
    match border:
        case "both":
            prefix = f"{'-' * len(x)}\n"
            suffix = f"\n{'-' * len(x)}"
        case "top":
            prefix = f"{'-' * len(x)}\n"
            suffix = ""
        case "bottom":
            prefix = ""
            suffix = f"\n{'-' * len(x)}"
        case _:
            raise ValueError(f"Invalid border: {border}")
    rich_print(f"{prefix}{x}{suffix}")


def print_config(config: DictConfig, name: str = "CONFIG", style="bright") -> None:
    """
    Prints content of DictConfig using Rich library and its tree structure.
    """
    tree = Tree(name, style=style, guide_style=style)
    fields = config.keys()
    for field in fields:
        try:
            branch = tree.add(str(field), style=style, guide_style=style)
            config_section = config.get(field)
            branch_content = str(config_section)
            if isinstance(config_section, (DictConfig, ListConfig)):
                branch_content = OmegaConf.to_yaml(config_section, resolve=True)
            branch.add(Syntax(branch_content, "yaml"))

        # except InterpolationResolutionError as e:
        except Exception as e:
            _error_text = f"({type(e).__name__}: {e})"
            print(f"-> Error resolving interpolation: {_error_text}")
            print(f"-> Field: {field}")
            print(f"-> Config section: {config_section}")

    rich_print(tree)


class AnsiColorLoggingFormatter(logging.Formatter):
    """
    ANSI color logging formatter
    """

    def format(self, record: logging.LogRecord) -> str:
        """
        Format the log record
        """
        no_style = "\033[0m"
        bold = "\033[91m"
        red = "\033[91m"
        yellow = "\033[93m"
        green = "\033[92m"
        cyan = "\033[96m"
        start_style = {
            "DEBUG": cyan,
            "INFO": green,
            "WARNING": yellow,
            "ERROR": red,
            "CRITICAL": red + bold,
        }.get(record.levelname, no_style)
        end_style = no_style
        return f"{start_style}{super().format(record)}{end_style}"


# Copied from https://github.com/thinking-machines-lab/tinker-cookbook/blob/22483a6b04400f79da13557a8229bc98b309b026/tinker_cookbook/utils/misc_utils.py#L18
@contextmanager
def timed(key: str, metrics: dict[str, Any]):
    """
    Update metrics with time taken for a given key
    """
    logger.info("Starting %s", key)
    tstart = time.time()
    yield
    logger.info("%s took %.2f seconds", key, time.time() - tstart)
    metrics[f"time/{key}"] = time.time() - tstart
