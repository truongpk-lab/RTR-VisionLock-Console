from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


BACKEND_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = BACKEND_ROOT / "config" / "default.yaml"


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data


def deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged
