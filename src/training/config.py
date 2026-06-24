from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Mapping

import yaml


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = deepcopy(dict(base))
    for key, value in override.items():
        if key in merged and isinstance(merged[key], Mapping) and isinstance(value, Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_config(path: str | Path) -> Dict[str, Any]:
    """Load a YAML config, optionally merging a relative `base` config first."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}

    base_path = cfg.pop("base", None)
    if base_path is None:
        return cfg

    parent = load_config(config_path.parent / str(base_path))
    return _deep_merge(parent, cfg)
