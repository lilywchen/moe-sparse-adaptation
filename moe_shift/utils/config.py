"""YAML config loading with `_base_` inheritance + dotted-key overrides."""
import copy
from pathlib import Path

import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str) -> dict:
    """Load YAML; if it has `_base_: <relpath>`, recursively load and deep-merge
    the current file on top. `_base_` is stripped from the result."""
    path = Path(path)
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    base_ref = cfg.pop("_base_", None)
    if base_ref is not None:
        base_cfg = load_config(path.parent / base_ref)
        cfg = _deep_merge(base_cfg, cfg)
    return cfg


def apply_overrides(cfg: dict, overrides: list) -> dict:
    """Each override is 'dotted.key=value'; value parsed via yaml.safe_load."""
    for ov in overrides or []:
        if "=" not in ov:
            continue
        key, raw = ov.split("=", 1)
        val = yaml.safe_load(raw)
        d = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = val
    return cfg
