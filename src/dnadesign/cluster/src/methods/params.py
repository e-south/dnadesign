"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cluster/src/methods/params.py

Lightweight parsing helpers for clustering-method CLI parameters.

Module Author(s): Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Sequence


def parse_method_param_assignments(assignments: Sequence[str]) -> dict[str, str]:
    params: dict[str, str] = {}
    for assignment in assignments:
        text = assignment.strip()
        if not text:
            continue
        if "=" not in text:
            raise ValueError(f"Invalid --method-param '{assignment}'. Expected key=value.")
        key, value = text.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            raise ValueError(f"Invalid --method-param '{assignment}'. Parameter name cannot be empty.")
        params[key] = value
    return params


__all__ = ["parse_method_param_assignments"]
