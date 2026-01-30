"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/src/adapters/sources/stage_a_paths.py

Stage-A filename-safe label helpers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations


def safe_label(text: str) -> str:
    return "".join(c if c.isalnum() or c in "-._" else "_" for c in str(text).strip()) or "stage_a"
