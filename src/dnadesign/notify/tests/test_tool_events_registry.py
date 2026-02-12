"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/tests/test_tool_events_registry.py

Tests for tool-agnostic notify tool-event registration and dispatch extension points.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from dnadesign.notify.errors import NotifyConfigError
from dnadesign.notify.tool_events import (
    ToolEventDecision,
    ToolEventState,
    evaluate_tool_event,
    register_tool_event_handlers,
    tool_event_message_override,
    tool_event_status_override,
)


def test_register_tool_event_handlers_supports_custom_action() -> None:
    action = f"custom_notify_action_{uuid4().hex}"
    event = {"action": action}
    state = ToolEventState()

    register_tool_event_handlers(
        action=action,
        status_override=lambda _event: "running",
        message_override=lambda _event, *, run_id, duration_seconds: (
            f"custom-message run={run_id} duration={duration_seconds}"
        ),
        evaluator=lambda _event, _run_id, _state: ToolEventDecision(emit=False, duration_seconds=3.0),
    )

    assert tool_event_status_override(action, event) == "running"
    assert tool_event_message_override(action, event, run_id="run-123", duration_seconds=1.0) == (
        "custom-message run=run-123 duration=1.0"
    )
    decision = evaluate_tool_event(action, event, run_id="run-123", state=state)
    assert decision.emit is False
    assert decision.duration_seconds == 3.0


def test_register_tool_event_handlers_rejects_duplicate_category_for_action() -> None:
    action = f"custom_notify_action_{uuid4().hex}"
    register_tool_event_handlers(
        action=action,
        status_override=lambda _event: "running",
    )

    with pytest.raises(NotifyConfigError, match="already registered"):
        register_tool_event_handlers(
            action=action,
            status_override=lambda _event: "success",
        )
