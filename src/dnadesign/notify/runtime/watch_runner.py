"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/runtime/watch_runner.py

Runtime orchestration for notify usr-events watch execution.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from .watch_runner_contract import (
    normalize_on_invalid_event_mode,
    resolve_optional_profile_bool,
    resolve_progress_min_seconds,
    resolve_progress_step_pct,
    validate_watch_request_contract,
)
from .watch_runner_resolution import (
    resolve_events_path,
    resolve_watch_mode,
    resolve_webhook_delivery,
    resolve_webhook_sources,
    validate_profile_events_source_match,
)


def run_usr_events_watch(
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
    read_profile: Callable[[Path | None], dict[str, Any]],
    resolve_string_value: Callable[..., str],
    resolve_path_value: Callable[..., Path],
    resolve_optional_path_value: Callable[..., Path | None],
    resolve_optional_string_value: Callable[..., str | None],
    resolve_profile_events_source: Callable[..., tuple[str, Path] | None],
    normalize_tool_name: Callable[[str | None], str | None],
    resolve_tool_events_path: Callable[..., tuple[Path, str | None]],
    resolve_tool_workspace_config: Callable[..., Path],
    resolve_usr_events_path: Callable[..., Path],
    resolve_profile_webhook_source: Callable[[dict[str, Any]], tuple[str | None, str | None]],
    default_profile_path_for_tool: Callable[[str | None], Path],
    resolve_cli_optional_string: Callable[..., str | None],
    resolve_webhook_url: Callable[..., str],
    resolve_tls_ca_bundle: Callable[..., Path | None],
    validate_provider_webhook_url: Callable[..., None],
    split_csv: Callable[[str | None], list[str]],
    watch_usr_events_loop: Callable[..., None],
    validate_usr_event: Callable[..., None],
    status_for_action: Callable[..., str | None],
    event_message: Callable[..., str],
    event_meta: Callable[..., dict[str, Any]],
    post_with_backoff: Callable[..., None],
) -> None:
    has_resolver_mode = validate_watch_request_contract(
        profile=profile,
        events=events,
        config=config,
        workspace=workspace,
        idle_timeout=idle_timeout,
        poll_interval_seconds=poll_interval_seconds,
    )
    mode = resolve_watch_mode(
        has_resolver_mode=has_resolver_mode,
        profile=profile,
        tool=tool,
        config=config,
        workspace=workspace,
        normalize_tool_name=normalize_tool_name,
        resolve_tool_events_path=resolve_tool_events_path,
        resolve_tool_workspace_config=resolve_tool_workspace_config,
        default_profile_path_for_tool=default_profile_path_for_tool,
    )

    profile_data = read_profile(mode.profile_path) if mode.profile_path is not None else {}
    validate_profile_events_source_match(
        mode=mode,
        profile_data=profile_data,
        resolve_profile_events_source=resolve_profile_events_source,
    )

    provider_value = resolve_string_value(field="provider", cli_value=provider, profile_data=profile_data)
    events_path = resolve_events_path(
        events=events,
        mode=mode,
        profile_data=profile_data,
        resolve_path_value=resolve_path_value,
        resolve_profile_events_source=resolve_profile_events_source,
        resolve_tool_events_path=resolve_tool_events_path,
        resolve_usr_events_path=resolve_usr_events_path,
    )
    cursor_path = resolve_optional_path_value(
        field="cursor",
        cli_value=cursor,
        profile_data=profile_data,
        profile_path=mode.profile_path,
    )
    only_actions_value = resolve_optional_string_value(
        field="only_actions",
        cli_value=only_actions,
        profile_data=profile_data,
    )
    only_tools_value = resolve_optional_string_value(
        field="only_tools",
        cli_value=only_tools,
        profile_data=profile_data,
    )
    progress_step_pct_value = resolve_progress_step_pct(
        progress_step_pct=progress_step_pct,
        profile_data=profile_data,
    )
    progress_min_seconds_value = resolve_progress_min_seconds(
        progress_min_seconds=progress_min_seconds,
        profile_data=profile_data,
    )
    spool_dir_value = resolve_optional_path_value(
        field="spool_dir",
        cli_value=spool_dir,
        profile_data=profile_data,
        profile_path=mode.profile_path,
    )
    include_args_value = resolve_optional_profile_bool(
        cli_value=include_args,
        profile_data=profile_data,
        field="include_args",
    )
    include_context_value = resolve_optional_profile_bool(
        cli_value=include_context,
        profile_data=profile_data,
        field="include_context",
    )
    include_raw_event_value = resolve_optional_profile_bool(
        cli_value=include_raw_event,
        profile_data=profile_data,
        field="include_raw_event",
    )

    url_env_value, secret_ref_value, profile_tls_ca_bundle = resolve_webhook_sources(
        profile_data=profile_data,
        url_env=url_env,
        secret_ref=secret_ref,
        tls_ca_bundle=tls_ca_bundle,
        profile_path=mode.profile_path,
        resolve_profile_webhook_source=resolve_profile_webhook_source,
        resolve_cli_optional_string=resolve_cli_optional_string,
        resolve_optional_path_value=resolve_optional_path_value,
    )
    webhook_url, resolved_tls_ca_bundle = resolve_webhook_delivery(
        dry_run=dry_run,
        url=url,
        url_env_value=url_env_value,
        secret_ref_value=secret_ref_value,
        provider_value=provider_value,
        profile_tls_ca_bundle=profile_tls_ca_bundle,
        resolve_webhook_url=resolve_webhook_url,
        validate_provider_webhook_url=validate_provider_webhook_url,
        resolve_tls_ca_bundle=resolve_tls_ca_bundle,
    )

    action_filter = set(split_csv(only_actions_value))
    tool_filter = set(split_csv(only_tools_value))
    on_invalid_event_mode = normalize_on_invalid_event_mode(on_invalid_event)

    watch_usr_events_loop(
        events_path=events_path,
        cursor_path=cursor_path,
        on_truncate=on_truncate,
        follow=follow,
        wait_for_events=wait_for_events,
        idle_timeout_seconds=idle_timeout,
        poll_interval_seconds=poll_interval_seconds,
        should_advance_cursor=(not dry_run) or bool(advance_cursor_on_dry_run),
        on_invalid_event_mode=on_invalid_event_mode,
        allow_unknown_version=allow_unknown_version,
        action_filter=action_filter,
        tool_filter=tool_filter,
        progress_step_pct=progress_step_pct_value,
        progress_min_seconds=progress_min_seconds_value,
        tool=mode.tool_value_for_events,
        run_id=run_id,
        provider_value=provider_value,
        message=message,
        include_args_value=bool(include_args_value),
        include_context_value=bool(include_context_value),
        include_raw_event_value=bool(include_raw_event_value),
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
        validate_usr_event=validate_usr_event,
        status_for_action=status_for_action,
        event_message=event_message,
        event_meta=event_meta,
        post_with_backoff=post_with_backoff,
    )
