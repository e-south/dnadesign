"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/usr_events_watch.py

USR events watch-loop execution for notify CLI.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import typer

from .cli_commands.providers import format_for_provider
from .errors import NotifyConfigError, NotifyDeliveryError
from .payload import build_payload
from .spool_ops import spool_payload as _spool_payload
from .tool_events import ToolEventState
from .tool_events import evaluate_tool_event as _evaluate_tool_event
from .tool_events import tool_event_message_override as _tool_event_message_override
from .tool_events import tool_event_status_override as _tool_event_status_override
from .watch_ops import acquire_cursor_lock as _acquire_cursor_lock
from .watch_ops import iter_file_lines as _iter_file_lines
from .watch_ops import load_cursor_offset as _load_cursor_offset
from .watch_ops import save_cursor_offset as _save_cursor_offset


def watch_usr_events_loop(
    *,
    events_path: Path,
    cursor_path: Path,
    on_truncate: str,
    follow: bool,
    wait_for_events: bool,
    idle_timeout_seconds: float | None,
    poll_interval_seconds: float,
    should_advance_cursor: bool,
    on_invalid_event_mode: str,
    allow_unknown_version: bool,
    action_filter: set[str],
    tool_filter: set[str],
    progress_step_pct: int | None,
    progress_min_seconds: float | None,
    tool: str | None,
    run_id: str | None,
    provider_value: str,
    message: str | None,
    include_args_value: bool,
    include_context_value: bool,
    include_raw_event_value: bool,
    dry_run: bool,
    stop_on_terminal_status: bool,
    webhook_url: str | None,
    resolved_tls_ca_bundle: Path | None,
    connect_timeout: float,
    read_timeout: float,
    retry_max: int,
    retry_base_seconds: float,
    fail_fast: bool,
    spool_dir_value: Path | None,
    validate_usr_event: Callable[..., None],
    status_for_action: Callable[..., str],
    event_message: Callable[..., str],
    event_meta: Callable[..., dict[str, Any]],
    post_with_backoff: Callable[..., None],
) -> None:
    tool_event_state = ToolEventState()
    densegen_bucket = tool_event_state.get_bucket("densegen_health")
    densegen_bucket["notify_config"] = {
        "progress_step_pct": progress_step_pct,
        "progress_min_seconds": progress_min_seconds,
    }

    def _save_cursor_if_enabled(next_offset: int) -> None:
        if should_advance_cursor:
            _save_cursor_offset(cursor_path, next_offset)

    with _acquire_cursor_lock(cursor_path):
        start_offset = _load_cursor_offset(cursor_path)
        failed_unsent = 0
        for next_offset, line in _iter_file_lines(
            events_path,
            start_offset=start_offset,
            on_truncate=on_truncate,
            follow=follow,
            wait_for_events=wait_for_events,
            idle_timeout_seconds=idle_timeout_seconds,
            poll_interval_seconds=poll_interval_seconds,
        ):
            text = line.strip()
            if not text:
                _save_cursor_if_enabled(next_offset)
                continue
            try:
                event = json.loads(text)
            except json.JSONDecodeError as exc:
                if on_invalid_event_mode == "skip":
                    typer.echo(f"Skipping invalid event line: event line is not valid JSON: {exc}")
                    _save_cursor_if_enabled(next_offset)
                    continue
                raise NotifyConfigError(f"event line is not valid JSON: {exc}") from exc
            try:
                validate_usr_event(event, allow_unknown_version=allow_unknown_version)
            except NotifyConfigError as exc:
                if on_invalid_event_mode == "skip":
                    typer.echo(f"Skipping invalid event line: {exc}")
                    _save_cursor_if_enabled(next_offset)
                    continue
                raise

            action = str(event.get("action"))
            if action_filter and action not in action_filter:
                _save_cursor_if_enabled(next_offset)
                continue

            actor_raw = event.get("actor")
            actor = actor_raw if isinstance(actor_raw, dict) else {}
            actor_tool = actor.get("tool")
            if tool_filter and actor_tool not in tool_filter:
                _save_cursor_if_enabled(next_offset)
                continue
            tool_name = tool or actor_tool
            if not tool_name:
                raise NotifyConfigError("event missing actor.tool; provide --tool to override")
            run_value = run_id or actor.get("run_id")
            if not run_value:
                raise NotifyConfigError("event missing actor.run_id; provide --run-id to override")

            tool_decision = _evaluate_tool_event(action, event, run_id=str(run_value), state=tool_event_state)
            if not tool_decision.emit:
                _save_cursor_if_enabled(next_offset)
                continue
            status_value = _tool_event_status_override(action, event) or status_for_action(action, event=event)

            payload = build_payload(
                status=status_value,
                tool=tool_name,
                run_id=run_value,
                message=message
                or _tool_event_message_override(
                    action,
                    event,
                    run_id=str(run_value),
                    duration_seconds=tool_decision.duration_seconds,
                )
                or event_message(
                    event,
                    run_id=str(run_value),
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
            formatted = format_for_provider(provider_value, payload)
            if dry_run:
                typer.echo(json.dumps(formatted, sort_keys=True))
                _save_cursor_if_enabled(next_offset)
                if stop_on_terminal_status and status_value in {"success", "failure"}:
                    return
                continue

            sent_or_spooled = False
            try:
                if webhook_url is None:
                    raise NotifyConfigError("webhook URL is required when not running in --dry-run mode")
                post_with_backoff(
                    webhook_url,
                    formatted,
                    tls_ca_bundle=resolved_tls_ca_bundle,
                    connect_timeout=connect_timeout,
                    read_timeout=read_timeout,
                    retry_max=retry_max,
                    retry_base_seconds=retry_base_seconds,
                )
                sent_or_spooled = True
            except NotifyDeliveryError:
                if spool_dir_value is not None:
                    _spool_payload(spool_dir_value, provider=provider_value, payload=payload)
                    sent_or_spooled = True
                elif fail_fast:
                    raise
                else:
                    failed_unsent += 1

            if sent_or_spooled:
                _save_cursor_if_enabled(next_offset)
                if stop_on_terminal_status and status_value in {"success", "failure"}:
                    return

        if failed_unsent:
            raise NotifyDeliveryError(f"{failed_unsent} event(s) failed delivery and were not spooled")
