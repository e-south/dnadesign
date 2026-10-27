"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/tool_events/densegen.py

DenseGen tool-event registration surface for notify.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .densegen_eval import _evaluate_densegen_health_event
from .densegen_messages import (
    _densegen_flush_failed_message,
    _densegen_health_message,
    _densegen_health_status_override,
)
from .types import ToolEventDecision, ToolEventState


def register_densegen_handlers(
    *,
    register_status_override: Callable[[str, Callable[[dict[str, Any]], str | None]], None],
    register_message_override: Callable[
        [
            str,
            Callable[[dict[str, Any]], str] | Callable[[dict[str, Any], str, float | None], str] | Callable[..., str],
        ],
        None,
    ],
    register_evaluator: Callable[[str, Callable[[dict[str, Any], str, ToolEventState], ToolEventDecision]], None],
) -> None:
    register_status_override("densegen_health", _densegen_health_status_override)
    register_message_override("densegen_health", _densegen_health_message)
    register_message_override("densegen_flush_failed", _densegen_flush_failed_message)
    register_evaluator("densegen_health", _evaluate_densegen_health_event)
