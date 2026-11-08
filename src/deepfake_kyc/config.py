"""Load YAML experiment configs.

An experiment file can name a parent with ``base: <file>``; its values are merged
on top of the parent's. Command-line overrides use dotted keys, e.g.
``data.batch_size=32``.
"""

from __future__ import annotations

import copy
from collections.abc import Iterable
from pathlib import Path

import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def apply_overrides(cfg: dict, overrides: Iterable[str]) -> dict:
    """Apply ``key.sub=value`` strings; values are parsed as YAML (numbers, bools, null)."""
    cfg = copy.deepcopy(cfg)
    for item in overrides:
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"Override must look like key=value, got {item!r}")
        *parents, leaf = key.split(".")
        node = cfg
        for part in parents:
            node = node.setdefault(part, {})
        node[leaf] = yaml.safe_load(value)
    return cfg


def load_config(path: str | Path, overrides: Iterable[str] = ()) -> dict:
    path = Path(path)
    cfg = yaml.safe_load(path.read_text()) or {}
    if "base" in cfg:
        parent = load_config(path.parent / cfg.pop("base"))
        cfg = _deep_merge(parent, cfg)
    return apply_overrides(cfg, overrides)
