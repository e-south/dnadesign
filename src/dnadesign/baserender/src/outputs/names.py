"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/src/outputs/names.py

Builds deterministic filenames for BaseRender output writers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import re


def _safe_stem(raw: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", raw.strip())
    stem = stem.strip("._-")
    return stem or "record"


def _unique_stem(base: str, used: set[str]) -> str:
    if base not in used:
        used.add(base)
        return base
    i = 2
    while True:
        candidate = f"{base}_{i}"
        if candidate not in used:
            used.add(candidate)
            return candidate
        i += 1


__all__ = ["_safe_stem", "_unique_stem"]
