"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/_helpers.py

Shared test helpers for the Eco1 RT repack study unit.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("repo root not found")


def load_yaml(rel_path: str) -> dict[str, Any]:
    payload = yaml.safe_load((repo_root() / rel_path).read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload
