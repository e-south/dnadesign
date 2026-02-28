"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/runtime/watch_events.py

Event parsing, filtering, and payload preparation for notify watch runtime.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import typer

from ..delivery.payload import build_payload
from ..errors import NotifyConfigError
from ..providers import format_payload
from ..tool_events import ToolEventState
from ..tool_events import evaluate_tool_event as _evaluate_tool_event
from ..tool_events import tool_event_message_override as _tool_event_message_override
from ..tool_events import tool_event_status_override as _tool_event_status_override


def parse_event_line(*, text: str, on_invalid_event_mode: str) -> dict[str, Any] | None:
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        if on_invalid_event_mode == "skip":
            typer.echo(f"Skipping invalid event line: event line is not valid JSON: {exc}")
            return None
        raise NotifyConfigError(f"event line is not valid JSON: {exc}") from exc


def validate_event_record(
    *,
    event: dict[str, Any],
    allow_unknown_version: bool,
    on_invalid_event_mode: str,
    validate_usr_event: Callable[..., None],
) -> bool:
    try:
        validate_usr_event(event, allow_unknown_version=allow_unknown_version)
        return True
    except NotifyConfigError as exc:
        if on_invalid_event_mode == "skip":
            typer.echo(f"Skipping invalid event line: {exc}")
            return False
        raise


def resolve_event_context(
    *,
    event: dict[str, Any],
    action_filter: set[str],
    tool_filter: set[str],
    tool: str | None,
    run_id: str | None,
) -> tuple[str, dict[str, Any], str, str] | None:
    action = str(event.get("action"))
    if action_filter and action not in action_filter:
        return None

    actor_raw = event.get("actor")
    actor = actor_raw if isinstance(actor_raw, dict) else {}
    actor_tool = actor.get("tool")
    if tool_filter and actor_tool not in tool_filter:
        return None
    tool_name = tool or actor_tool
    if not tool_name:
        raise NotifyConfigError("event missing actor.tool; provide --tool to override")
    run_value = run_id or actor.get("run_id")
    if not run_value:
        raise NotifyConfigError("event missing actor.run_id; provide --run-id to override")

    return action, actor, str(tool_name), str(run_value)


def prepare_event_payload(
    *,
    action: str,
    event: dict[str, Any],
    actor: dict[str, Any],
    tool_name: str,
    run_value: str,
    provider_value: str,
    message: str | None,
    include_args_value: bool,
    include_context_value: bool,
    include_raw_event_value: bool,
    tool_event_state: ToolEventState,
    status_for_action: Callable[..., str],
    event_message: Callable[..., str],
    event_meta: Callable[..., dict[str, Any]],
) -> tuple[str, dict[str, Any], dict[str, Any]] | None:
    tool_decision = _evaluate_tool_event(action, event, run_id=run_value, state=tool_event_state)
    if not tool_decision.emit:
        return None
    status_value = _tool_event_status_override(action, event) or status_for_action(action, event=event)
    payload = build_payload(
        status=status_value,
        tool=tool_name,
        run_id=run_value,
        message=message
        or _tool_event_message_override(
            action,
            event,
            run_id=run_value,
            duration_seconds=tool_decision.duration_seconds,
        )
        or event_message(
            event,
            run_id=run_value,
            duration_seconds=tool_decision.duration_seconds,
        ),
        meta=event_meta(
            event,
            include_args=bool(include_args_value),
            include_raw_event=bool(include_raw_event_value),
            include_context=bool(include_context_value),
        ),
        timestamp=event.get("timestamp_utc"),
        host=(actor.get("host") if bool(include_context_value) else None),
        cwd=(
            ((event.get("dataset") or {}) if isinstance(event.get("dataset"), dict) else {}).get("root")
            if bool(include_context_value)
            else None
        ),
        version=event.get("version"),
    )
    formatted_payload = format_payload(provider_value, payload)
    return status_value, payload, formatted_payload
