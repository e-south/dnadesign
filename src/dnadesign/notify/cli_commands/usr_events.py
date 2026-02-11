"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/cli_commands/usr_events.py

USR events watch command registration for Notify CLI.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import typer


def register_usr_events_watch_command(
    usr_events_app: typer.Typer,
    *,
    watch_handler: Callable[..., None],
) -> None:
    @usr_events_app.command("watch")
    def usr_events_watch(
        provider: str | None = typer.Option(None, help="Provider: generic|slack|discord."),
        url: str | None = typer.Option(None, help="Webhook URL."),
        url_env: str | None = typer.Option(None, help="Environment variable holding webhook URL."),
        secret_ref: str | None = typer.Option(
            None,
            "--secret-ref",
            help="Secret reference: keychain://service/account or secretservice://service/account.",
        ),
        events: Path | None = typer.Option(None, "--events", help="USR events JSONL path."),
        profile: Path | None = typer.Option(None, "--profile", help="Path to profile JSON file."),
        cursor: Path | None = typer.Option(None, "--cursor", help="Cursor file storing byte offset."),
        follow: bool = typer.Option(False, "--follow", help="Follow events file for new lines."),
        wait_for_events: bool = typer.Option(
            False,
            "--wait-for-events",
            help="When following, wait for events file creation instead of failing immediately.",
        ),
        idle_timeout: float | None = typer.Option(
            None,
            "--idle-timeout",
            help="Exit after this many seconds without new events while following.",
        ),
        poll_interval_seconds: float = typer.Option(
            0.2,
            "--poll-interval-seconds",
            help="Polling interval for follow/wait loops (seconds).",
        ),
        stop_on_terminal_status: bool = typer.Option(
            False,
            "--stop-on-terminal-status",
            help="Exit after the first event mapped to success or failure status.",
        ),
        on_truncate: str = typer.Option(
            "error",
            "--on-truncate",
            help="Behavior when cursor offset exceeds file size: error|restart.",
        ),
        only_actions: str | None = typer.Option(None, "--only-actions", help="Comma-separated action filter."),
        only_tools: str | None = typer.Option(None, "--only-tools", help="Comma-separated actor tool filter."),
        on_invalid_event: str = typer.Option(
            "error",
            "--on-invalid-event",
            help="Behavior for malformed event lines: error|skip.",
        ),
        allow_unknown_version: bool = typer.Option(
            False,
            "--allow-unknown-version",
            help="Allow unknown event_version values.",
        ),
        tool: str | None = typer.Option(None, help="Override tool name."),
        run_id: str | None = typer.Option(None, help="Override run id."),
        message: str | None = typer.Option(None, help="Override message."),
        include_args: bool | None = typer.Option(None, "--include-args/--no-include-args"),
        include_context: bool | None = typer.Option(None, "--include-context/--no-include-context"),
        include_raw_event: bool | None = typer.Option(None, "--include-raw-event/--no-include-raw-event"),
        connect_timeout: float = typer.Option(5.0, help="HTTP connect timeout seconds."),
        read_timeout: float = typer.Option(10.0, help="HTTP read timeout seconds."),
        retry_max: int = typer.Option(3, "--retry-max", help="Max retries for each delivery."),
        retry_base_seconds: float = typer.Option(0.5, "--retry-base-seconds", help="Base retry delay in seconds."),
        fail_fast: bool = typer.Option(False, "--fail-fast", help="Abort on first unsent event."),
        spool_dir: Path | None = typer.Option(None, "--spool-dir", help="Write failed payloads to spool directory."),
        dry_run: bool = typer.Option(False, help="Print formatted payloads instead of posting."),
        advance_cursor_on_dry_run: bool = typer.Option(
            True,
            "--advance-cursor-on-dry-run/--no-advance-cursor-on-dry-run",
            help="When --dry-run is enabled, persist cursor offsets (default: enabled).",
        ),
    ) -> None:
        watch_handler(
            provider=provider,
            url=url,
            url_env=url_env,
            secret_ref=secret_ref,
            events=events,
            profile=profile,
            cursor=cursor,
            follow=follow,
            wait_for_events=wait_for_events,
            idle_timeout=idle_timeout,
            poll_interval_seconds=poll_interval_seconds,
            stop_on_terminal_status=stop_on_terminal_status,
            on_truncate=on_truncate,
            only_actions=only_actions,
            only_tools=only_tools,
            on_invalid_event=on_invalid_event,
            allow_unknown_version=allow_unknown_version,
            tool=tool,
            run_id=run_id,
            message=message,
            include_args=include_args,
            include_context=include_context,
            include_raw_event=include_raw_event,
            connect_timeout=connect_timeout,
            read_timeout=read_timeout,
            retry_max=retry_max,
            retry_base_seconds=retry_base_seconds,
            fail_fast=fail_fast,
            spool_dir=spool_dir,
            dry_run=dry_run,
            advance_cursor_on_dry_run=advance_cursor_on_dry_run,
        )
