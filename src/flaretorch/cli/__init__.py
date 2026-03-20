"""
FlareTorch CLI entry points.
"""

from flaretorch.cli.train import main as train_main
from flaretorch.cli.eval import main as eval_main

__all__ = ["train_main", "eval_main"]
