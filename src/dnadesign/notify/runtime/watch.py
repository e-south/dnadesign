"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/runtime/watch.py

USR events watch-loop execution for notify CLI.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..errors import NotifyDeliveryError
from ..tool_events import ToolEventState
from .cursor import acquire_cursor_lock as _acquire_cursor_lock
from .cursor import iter_file_lines as _iter_file_lines
from .cursor import load_cursor_offset as _load_cursor_offset
from .cursor import save_cursor_offset as _save_cursor_offset
from .watch_delivery import emit_or_deliver_event
from .watch_events import parse_event_line, prepare_event_payload, resolve_event_context, validate_event_record


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

            event = parse_event_line(text=text, on_invalid_event_mode=on_invalid_event_mode)
            if event is None:
                _save_cursor_if_enabled(next_offset)
                continue
            is_valid = validate_event_record(
                event=event,
                allow_unknown_version=allow_unknown_version,
                on_invalid_event_mode=on_invalid_event_mode,
                validate_usr_event=validate_usr_event,
            )
            if not is_valid:
                _save_cursor_if_enabled(next_offset)
                continue

            event_context = resolve_event_context(
                event=event,
                action_filter=action_filter,
                tool_filter=tool_filter,
                tool=tool,
                run_id=run_id,
            )
            if event_context is None:
                _save_cursor_if_enabled(next_offset)
                continue
            action, actor, tool_name, run_value = event_context

            event_payload = prepare_event_payload(
                action=action,
                event=event,
                actor=actor,
                tool_name=tool_name,
                run_value=run_value,
                provider_value=provider_value,
                message=message,
                include_args_value=include_args_value,
                include_context_value=include_context_value,
                include_raw_event_value=include_raw_event_value,
                tool_event_state=tool_event_state,
                status_for_action=status_for_action,
                event_message=event_message,
                event_meta=event_meta,
            )
            if event_payload is None:
                _save_cursor_if_enabled(next_offset)
                continue
            status_value, payload, formatted_payload = event_payload

            delivery_outcome = emit_or_deliver_event(
                status_value=status_value,
                payload=payload,
                formatted_payload=formatted_payload,
                dry_run=dry_run,
                stop_on_terminal_status=stop_on_terminal_status,
                webhook_url=webhook_url,
                resolved_tls_ca_bundle=resolved_tls_ca_bundle,
                connect_timeout=connect_timeout,
                read_timeout=read_timeout,
                retry_max=retry_max,
                retry_base_seconds=retry_base_seconds,
                fail_fast=fail_fast,
                spool_dir_value=spool_dir_value,
                provider_value=provider_value,
                post_with_backoff=post_with_backoff,
            )
            failed_unsent += delivery_outcome.failed_unsent_delta
            if delivery_outcome.cursor_advanced:
                _save_cursor_if_enabled(next_offset)
            if delivery_outcome.terminal_reached:
                return

        if failed_unsent:
            raise NotifyDeliveryError(f"{failed_unsent} event(s) failed delivery and were not spooled")
