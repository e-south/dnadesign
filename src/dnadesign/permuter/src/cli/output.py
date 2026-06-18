"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/permuter/src/cli/output.py

CLI emitters for stable machine-readable Permuter responses.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import sys
from datetime import date, datetime
from pathlib import Path


def emit_json(payload: dict[str, object]) -> None:
    sys.stdout.write(json.dumps(payload, default=_json_default, sort_keys=True) + "\n")


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except Exception:
            pass
    return str(value)
