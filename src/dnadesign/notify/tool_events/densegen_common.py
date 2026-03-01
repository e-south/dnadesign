"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/tool_events/densegen_common.py

Shared DenseGen tool-event parsing and value-conversion helpers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def _normalize_densegen_status(event: dict[str, Any]) -> str:
    args_raw = event.get("args")
    args = args_raw if isinstance(args_raw, dict) else {}
    return str(args.get("status") or "").strip().lower()


def _event_timestamp_seconds(event: dict[str, Any]) -> float | None:
    raw = event.get("timestamp_utc")
    if not isinstance(raw, str) or not raw.strip():
        return None
    ts = raw.strip()
    if ts.endswith("Z"):
        ts = f"{ts[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(ts)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return float(parsed.timestamp())


def _duration_hhmmss(seconds: float) -> str:
    total = max(0, int(round(float(seconds))))
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _to_int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _to_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
