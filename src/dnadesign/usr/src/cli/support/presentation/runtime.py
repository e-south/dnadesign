"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/src/cli/support/presentation/runtime.py

Shared CLI runtime-format helpers for the USR entrypoint facade.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import math
import sys
from collections.abc import Mapping, Sequence
from datetime import date, datetime
from decimal import Decimal

from ....contracts import SequencesError


def resolve_output_format(args, *, default: str = "auto") -> str:
    fmt = str(getattr(args, "format", default) or default).lower()
    if fmt not in {"auto", "rich", "plain", "json"}:
        raise SequencesError(f"Unsupported format '{fmt}'. Use auto|rich|plain|json.")
    if fmt == "auto":
        if is_interactive() and bool(getattr(args, "rich", True)):
            return "rich"
        return "plain"
    return fmt


def print_json(payload) -> None:
    print(json.dumps(_json_safe(payload), separators=(",", ":"), allow_nan=False))


def _json_safe(value):
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Decimal):
        return float(value) if value.is_finite() else None
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if _is_pandas_missing_scalar(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_safe(item) for item in value]
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _json_safe(item())
        except (TypeError, ValueError):
            pass
    return value


def _is_pandas_missing_scalar(value) -> bool:
    cls = type(value)
    module = str(getattr(cls, "__module__", ""))
    name = str(getattr(cls, "__name__", ""))
    return module.startswith("pandas.") and name in {"NAType", "NaTType"}


def is_interactive() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


__all__ = ["is_interactive", "print_json", "resolve_output_format"]
