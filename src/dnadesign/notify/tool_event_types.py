"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/tool_event_types.py

Core dataclasses for notify tool-event evaluation and state.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolEventDecision:
    emit: bool = True
    duration_seconds: float | None = None


@dataclass
class ToolEventState:
    per_action_state: dict[str, dict[str, Any]] = field(default_factory=dict)

    def get_bucket(self, action: str) -> dict[str, Any]:
        key = str(action or "").strip().lower()
        if not key:
            return {}
        return self.per_action_state.setdefault(key, {})
