"""Helpers shared by the command-line scripts."""
import argparse
import json
from pathlib import Path

import numpy as np

from .models import load_checkpoint, resolve_norm_stats


def named_path(text):
    """``name=path`` argument -> (name, Path)."""
    name, sep, path = text.partition("=")
    if not sep or not name or not path:
        raise argparse.ArgumentTypeError(f"expected NAME=PATH, got {text!r}")
    return name, Path(path)


def unique_names(pairs, parser, option):
    names = [name for name, _ in pairs]
    if len(set(names)) != len(names):
        parser.error(f"{option}: names must be unique, got {names}")
    for _, path in pairs:
        if not path.is_file():
            parser.error(f"{option}: file not found: {path}")
    return dict(pairs)


def load_model(path, variable=None, stats_path=None, device="cpu"):
    """Model, normalization statistics and checkpoint info."""
    model, info = load_checkpoint(path, variable, device=device)
    return model, resolve_norm_stats(info, stats_path), info


def to_jsonable(value):
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(path, data):
    Path(path).write_text(json.dumps(to_jsonable(data), indent=2) + "\n")
