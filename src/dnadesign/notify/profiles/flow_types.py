"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/profiles/flow_types.py

Shared data types and validation helpers for notify setup/profile flows.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..errors import NotifyConfigError


@dataclass(frozen=True)
class SetupEventsResolution:
    events_path: Path
    events_source: dict[str, str] | None
    policy: str | None
    tool_name: str | None
    events_require_exists: bool


def _validate_progress_step_pct(value: int | None) -> int | None:
    if value is None:
        return None
    step = int(value)
    if step < 1 or step > 100:
        raise NotifyConfigError("progress_step_pct must be an integer between 1 and 100")
    return step


def _validate_progress_min_seconds(value: float | None) -> float | None:
    if value is None:
        return None
    minimum = float(value)
    if minimum <= 0.0:
        raise NotifyConfigError("progress_min_seconds must be a positive number")
    return minimum
