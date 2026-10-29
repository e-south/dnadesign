"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/providers/generic.py

Generic provider formatting for webhook payloads.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any


def format_generic(payload: dict[str, Any]) -> dict[str, Any]:
    return dict(payload)
