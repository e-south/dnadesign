"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/tests/test_tool_events_registry.py

Tests for tool-agnostic notify tool-event registration and dispatch extension points.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import ast
import inspect
from uuid import uuid4

import pytest

from dnadesign.notify.errors import NotifyConfigError
from dnadesign.notify.tool_events import (
    ToolEventDecision,
    ToolEventState,
    activate_tool_event_pack,
    evaluate_tool_event,
    register_tool_event_handlers,
    register_tool_event_pack,
    supported_tool_event_packs,
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


def test_activate_tool_event_pack_supports_custom_pack() -> None:
    action = f"custom_pack_action_{uuid4().hex}"
    pack = f"custom_pack_{uuid4().hex}"

    def _install(register):
        register(action=action, status_override=lambda _event: "running")

    register_tool_event_pack(pack=pack, installer=_install)
    activate_tool_event_pack(pack)

    assert pack in supported_tool_event_packs()
    assert tool_event_status_override(action, {"action": action}) == "running"


def test_register_tool_event_pack_rejects_duplicate_name() -> None:
    pack = f"custom_pack_{uuid4().hex}"
    register_tool_event_pack(pack=pack, installer=lambda _register: None)
    with pytest.raises(NotifyConfigError, match="already registered"):
        register_tool_event_pack(pack=pack, installer=lambda _register: None)


def test_activate_tool_event_pack_rejects_unknown_pack() -> None:
    with pytest.raises(NotifyConfigError, match="unknown tool-event pack"):
        activate_tool_event_pack(f"unknown_pack_{uuid4().hex}")


def test_activate_tool_event_pack_rejects_double_activation() -> None:
    pack = f"custom_pack_{uuid4().hex}"
    register_tool_event_pack(pack=pack, installer=lambda _register: None)
    activate_tool_event_pack(pack)
    with pytest.raises(NotifyConfigError, match="already activated"):
        activate_tool_event_pack(pack)


def test_tool_events_module_has_no_tool_specific_imports() -> None:
    import dnadesign.notify.tool_events as tool_events_module

    parsed = ast.parse(inspect.getsource(tool_events_module))
    imported_modules: set[str] = set()
    for node in ast.walk(parsed):
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)
        if isinstance(node, ast.ImportFrom):
            imported_modules.add(str(node.module or ""))
    assert "tool_events_densegen" not in imported_modules
