"""
utils/config.py
================
Small YAML config loader shared by train.py / eval.py.

Not called out as its own file in the original design doc, but every config
in configs/ declares `_base_: base.yaml`, so something has to resolve that
inheritance chain — that's what this module does. Kept deliberately tiny
(no external config-framework dependency) so the rest of the repo stays easy
to read.

Usage:
    from utils.config import load_config
    cfg = load_config("configs/ssl_moco.yaml")
    cfg.optim.lr            # dot-access
    cfg["optim"]["lr"]      # dict-access also works, it's a plain dict subclass
"""
from __future__ import annotations

import copy
import os
from typing import Any, Dict

import yaml


class ConfigDict(dict):
    """A dict that also allows attribute-style access (cfg.data.clip_len)."""

    def __getattr__(self, name: str) -> Any:
        try:
            value = self[name]
        except KeyError as e:
            raise AttributeError(name) from e
        return ConfigDict(value) if isinstance(value, dict) else value

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def get(self, key, default=None):
        value = super().get(key, default)
        return ConfigDict(value) if isinstance(value, dict) else value


def _deep_merge(base: Dict, override: Dict) -> Dict:
    """Recursively merge `override` into `base`, returning a new dict.
    Leaf values in `override` always win; nested dicts are merged key-by-key
    instead of being replaced wholesale (so a config only has to specify the
    handful of fields it actually changes)."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key == "_base_":
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _load_raw(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config(path: str) -> ConfigDict:
    """Load a YAML config, resolving a single-level or chained `_base_` field
    (relative to the file that references it) before applying CLI overrides."""
    raw = _load_raw(path)
    if "_base_" in raw:
        base_path = os.path.join(os.path.dirname(path), raw["_base_"])
        base_cfg = _load_raw(base_path)
        raw = _deep_merge(base_cfg, raw)
    return ConfigDict(raw)


def apply_overrides(cfg: ConfigDict, overrides: list[str]) -> ConfigDict:
    """Apply `key.subkey=value` CLI overrides (as passed via --opts) on top of
    a loaded config. Values are parsed with yaml.safe_load so ints/floats/
    bools/lists come through as the right Python type, not strings."""
    cfg = ConfigDict(copy.deepcopy(cfg))
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Bad override '{item}', expected key.subkey=value")
        key_path, value = item.split("=", 1)
        parsed_value = yaml.safe_load(value)
        node = cfg
        keys = key_path.split(".")
        for k in keys[:-1]:
            if k not in node or not isinstance(node[k], dict):
                node[k] = {}
            node = node[k]
        node[keys[-1]] = parsed_value
    return cfg
