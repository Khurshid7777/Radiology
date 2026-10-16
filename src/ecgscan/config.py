"""Scanner configuration: YAML files under ``configs/scanner`` (TZ §0.3: no hard-coding).

The directory is ``$ECGSCAN_CONFIG_DIR`` when set, else ``configs/scanner`` at the
repository root (found by walking up from this file, as an editable install keeps
the source tree in place).
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


def config_dir() -> Path:
    env = os.environ.get("ECGSCAN_CONFIG_DIR")
    if env:
        return Path(env)
    for parent in Path(__file__).resolve().parents:
        cand = parent / "configs" / "scanner"
        if cand.is_dir():
            return cand
    raise FileNotFoundError("configs/scanner not found; set ECGSCAN_CONFIG_DIR")


def repo_root() -> Path:
    return config_dir().parent.parent


@lru_cache(maxsize=32)
def _load_cached(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_yaml(name: str) -> dict[str, Any]:
    """Load ``configs/scanner/<name>`` (a relative path, ``.yaml`` optional)."""
    p = config_dir() / name
    if p.suffix not in (".yaml", ".yml"):
        p = p.with_suffix(".yaml")
    return _load_cached(str(p))
