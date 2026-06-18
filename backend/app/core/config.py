from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


BACKEND_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = BACKEND_ROOT / "config" / "default.yaml"
LOCAL_CONFIG = BACKEND_ROOT / "config" / "local.yaml"


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    # Machine-specific overrides (model paths, GPU backbones) live in an untracked
    # config/local.yaml so the committed default stays the portable Jetson default.
    # Tests set RTR_NO_LOCAL_CONFIG (see tests/conftest.py) so the suite always runs
    # against the lightweight default, not a developer's GPU stack.
    if path == DEFAULT_CONFIG and LOCAL_CONFIG.exists() and not os.environ.get("RTR_NO_LOCAL_CONFIG"):
        with LOCAL_CONFIG.open("r", encoding="utf-8") as handle:
            local = yaml.safe_load(handle) or {}
        if isinstance(local, dict):
            data = deep_merge(data, local)
    return data


def deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged
