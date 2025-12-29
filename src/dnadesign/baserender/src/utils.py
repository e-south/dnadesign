"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/baserender/src/utils.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import re


def safe_stem(value: str, *, max_len: int = 120) -> str:
    """
    Create a filesystem-safe stem.
    - Replace path separators with '_'
    - Allow only [A-Za-z0-9._-], replace others with '_'
    - Collapse repeated '_' and trim to max_len
    """
    s = str(value or "")
    s = s.replace("/", "_").replace("\\", "_")
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    s = re.sub(r"_+", "_", s)
    s = s.strip("_ ")
    if len(s) > max_len:
        s = s[:max_len]
    if s == "":
        return "record"
    return s


def unique_stem(base: str, used: set[str]) -> str:
    """Return a unique stem by suffixing _{n} when needed."""
    if base not in used:
        used.add(base)
        return base
    i = 1
    while True:
        cand = f"{base}_{i}"
        if cand not in used:
            used.add(cand)
            return cand
        i += 1
