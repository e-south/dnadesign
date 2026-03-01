"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/cli/handlers/runtime/watch_cmd.py

Execution logic for notify usr-events watch runtime command.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import typer

from ....errors import NotifyError


def run_usr_events_watch_command(
    *,
    provider: str | None,
    url: str | None,
    url_env: str | None,
    secret_ref: str | None,
    tls_ca_bundle: Path | None,
    events: Path | None,
    profile: Path | None,
    config: Path | None,
    workspace: str | None,
    cursor: Path | None,
    follow: bool,
    wait_for_events: bool,
    idle_timeout: float | None,
    poll_interval_seconds: float,
    stop_on_terminal_status: bool,
    on_truncate: str,
    only_actions: str | None,
    only_tools: str | None,
    progress_step_pct: int | None,
    progress_min_seconds: float | None,
    on_invalid_event: str,
    allow_unknown_version: bool,
    tool: str | None,
    run_id: str | None,
    message: str | None,
    include_args: bool | None,
    include_context: bool | None,
    include_raw_event: bool | None,
    connect_timeout: float,
    read_timeout: float,
    retry_max: int,
    retry_base_seconds: float,
    fail_fast: bool,
    spool_dir: Path | None,
    dry_run: bool,
    advance_cursor_on_dry_run: bool,
    run_usr_events_watch_fn: Callable[..., None],
    read_profile_fn: Callable[[Path], dict[str, Any]],
    resolve_string_value_fn: Callable[..., str],
    resolve_path_value_fn: Callable[..., Path],
    resolve_optional_path_value_fn: Callable[..., Path | None],
    resolve_optional_string_value_fn: Callable[..., str | None],
    resolve_profile_events_source_fn: Callable[..., tuple[str, Path] | None],
    normalize_tool_name_fn: Callable[[str | None], str | None],
    resolve_tool_events_path_fn: Callable[..., tuple[Path, str | None]],
    resolve_tool_workspace_config_fn: Callable[..., Path],
    resolve_usr_events_path_fn: Callable[..., Path],
    resolve_profile_webhook_source_fn: Callable[[dict[str, Any]], tuple[str | None, str | None]],
    default_profile_path_for_tool_fn: Callable[[str], Path],
    resolve_cli_optional_string_fn: Callable[..., str | None],
    resolve_webhook_url_fn: Callable[..., str],
    resolve_tls_ca_bundle_fn: Callable[..., Path | None],
    validate_provider_webhook_url_fn: Callable[..., None],
    split_csv_fn: Callable[[str | None], list[str]],
    watch_usr_events_loop_fn: Callable[..., int],
    validate_usr_event_fn: Callable[..., None],
    status_for_action_fn: Callable[..., str],
    event_message_fn: Callable[..., str | None],
    event_meta_fn: Callable[..., dict[str, Any]],
    post_with_backoff_fn: Callable[..., None],
) -> None:
    try:
        run_usr_events_watch_fn(
            provider=provider,
            url=url,
            url_env=url_env,
            secret_ref=secret_ref,
            tls_ca_bundle=tls_ca_bundle,
            events=events,
            profile=profile,
            config=config,
            workspace=workspace,
            cursor=cursor,
            follow=follow,
            wait_for_events=wait_for_events,
            idle_timeout=idle_timeout,
            poll_interval_seconds=poll_interval_seconds,
            stop_on_terminal_status=stop_on_terminal_status,
            on_truncate=on_truncate,
            only_actions=only_actions,
            only_tools=only_tools,
            progress_step_pct=progress_step_pct,
            progress_min_seconds=progress_min_seconds,
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
            read_profile=read_profile_fn,
            resolve_string_value=resolve_string_value_fn,
            resolve_path_value=resolve_path_value_fn,
            resolve_optional_path_value=resolve_optional_path_value_fn,
            resolve_optional_string_value=resolve_optional_string_value_fn,
            resolve_profile_events_source=resolve_profile_events_source_fn,
            normalize_tool_name=normalize_tool_name_fn,
            resolve_tool_events_path=resolve_tool_events_path_fn,
            resolve_tool_workspace_config=resolve_tool_workspace_config_fn,
            resolve_usr_events_path=resolve_usr_events_path_fn,
            resolve_profile_webhook_source=resolve_profile_webhook_source_fn,
            default_profile_path_for_tool=default_profile_path_for_tool_fn,
            resolve_cli_optional_string=resolve_cli_optional_string_fn,
            resolve_webhook_url=resolve_webhook_url_fn,
            resolve_tls_ca_bundle=resolve_tls_ca_bundle_fn,
            validate_provider_webhook_url=validate_provider_webhook_url_fn,
            split_csv=split_csv_fn,
            watch_usr_events_loop=watch_usr_events_loop_fn,
            validate_usr_event=validate_usr_event_fn,
            status_for_action=status_for_action_fn,
            event_message=event_message_fn,
            event_meta=event_meta_fn,
            post_with_backoff=post_with_backoff_fn,
        )
    except NotifyError as exc:
        typer.echo(f"Notification failed: {exc}")
        raise typer.Exit(code=1)
