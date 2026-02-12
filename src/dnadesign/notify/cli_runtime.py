"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/cli_runtime.py

Runtime execution helpers for notify watch and spool CLI handlers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from .errors import NotifyConfigError, NotifyDeliveryError, NotifyError


def run_usr_events_watch(
    *,
    provider: str | None,
    url: str | None,
    url_env: str | None,
    secret_ref: str | None,
    tls_ca_bundle: Path | None,
    events: Path | None,
    profile: Path | None,
    cursor: Path | None,
    follow: bool,
    wait_for_events: bool,
    idle_timeout: float | None,
    poll_interval_seconds: float,
    stop_on_terminal_status: bool,
    on_truncate: str,
    only_actions: str | None,
    only_tools: str | None,
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
    resolve_tool_events_path: Callable[..., tuple[Path, str | None]],
    resolve_usr_events_path: Callable[..., Path],
    resolve_profile_webhook_source: Callable[[dict[str, Any]], tuple[str | None, str | None]],
    resolve_cli_optional_string: Callable[..., str | None],
    resolve_webhook_url: Callable[..., str],
    resolve_tls_ca_bundle: Callable[..., Path | None],
    split_csv: Callable[[str | None], list[str]],
    watch_usr_events_loop: Callable[..., None],
    validate_usr_event: Callable[..., None],
    status_for_action: Callable[..., str | None],
    event_message: Callable[..., str],
    event_meta: Callable[..., dict[str, Any]],
    post_with_backoff: Callable[..., None],
) -> None:
    if idle_timeout is not None and float(idle_timeout) <= 0:
        raise NotifyConfigError("idle_timeout must be > 0 when provided")
    if float(poll_interval_seconds) <= 0:
        raise NotifyConfigError("poll_interval_seconds must be > 0")
    profile_path = profile.expanduser().resolve() if profile is not None else None
    profile_data = read_profile(profile_path) if profile_path is not None else {}
    provider_value = resolve_string_value(field="provider", cli_value=provider, profile_data=profile_data)
    events_path = resolve_path_value(
        field="events",
        cli_value=events,
        profile_data=profile_data,
        profile_path=profile_path,
    )
    if events is None:
        profile_events_source = resolve_profile_events_source(profile_data=profile_data, profile_path=profile_path)
        if profile_events_source is not None:
            source_tool, source_config = profile_events_source
            resolved_events_path, _default_policy = resolve_tool_events_path(
                tool=source_tool,
                config=source_config,
            )
            events_path = resolve_usr_events_path(resolved_events_path, require_exists=False)
    cursor_path = resolve_optional_path_value(
        field="cursor",
        cli_value=cursor,
        profile_data=profile_data,
        profile_path=profile_path,
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
    spool_dir_value = resolve_optional_path_value(
        field="spool_dir",
        cli_value=spool_dir,
        profile_data=profile_data,
        profile_path=profile_path,
    )
    include_args_value = include_args
    if include_args_value is None:
        include_args_profile = profile_data.get("include_args")
        include_args_value = bool(include_args_profile) if include_args_profile is not None else False
    include_context_value = include_context
    if include_context_value is None:
        include_context_profile = profile_data.get("include_context")
        include_context_value = bool(include_context_profile) if include_context_profile is not None else False
    include_raw_event_value = include_raw_event
    if include_raw_event_value is None:
        include_raw_event_profile = profile_data.get("include_raw_event")
        include_raw_event_value = bool(include_raw_event_profile) if include_raw_event_profile is not None else False
    profile_url_env, profile_secret_ref = resolve_profile_webhook_source(profile_data)
    url_env_value = resolve_cli_optional_string(field="url_env", cli_value=url_env)
    if url_env_value is None:
        url_env_value = profile_url_env
    secret_ref_value = resolve_cli_optional_string(field="secret_ref", cli_value=secret_ref)
    if secret_ref_value is None:
        secret_ref_value = profile_secret_ref
    profile_tls_ca_bundle = resolve_optional_path_value(
        field="tls_ca_bundle",
        cli_value=tls_ca_bundle,
        profile_data=profile_data,
        profile_path=profile_path,
    )

    webhook_url: str | None = None
    resolved_tls_ca_bundle: Path | None = None
    if not dry_run:
        webhook_url = resolve_webhook_url(url=url, url_env=url_env_value, secret_ref=secret_ref_value)
        resolved_tls_ca_bundle = resolve_tls_ca_bundle(
            webhook_url=webhook_url,
            tls_ca_bundle=profile_tls_ca_bundle,
        )
    action_filter = set(split_csv(only_actions_value))
    tool_filter = set(split_csv(only_tools_value))
    on_invalid_event_mode = str(on_invalid_event or "").strip().lower()
    if on_invalid_event_mode not in {"error", "skip"}:
        raise NotifyConfigError(f"unsupported on-invalid-event mode '{on_invalid_event}'")

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
        tool=tool,
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


def run_spool_drain(
    *,
    spool_dir: Path | None,
    provider: str | None,
    url: str | None,
    url_env: str | None,
    secret_ref: str | None,
    tls_ca_bundle: Path | None,
    profile: Path | None,
    connect_timeout: float,
    read_timeout: float,
    retry_max: int,
    retry_base_seconds: float,
    fail_fast: bool,
    read_profile: Callable[[Path | None], dict[str, Any]],
    resolve_cli_optional_string: Callable[..., str | None],
    resolve_path_value: Callable[..., Path],
    resolve_profile_webhook_source: Callable[[dict[str, Any]], tuple[str | None, str | None]],
    resolve_optional_path_value: Callable[..., Path | None],
    resolve_webhook_url: Callable[..., str],
    resolve_tls_ca_bundle: Callable[..., Path | None],
    format_for_provider: Callable[[str, dict[str, Any]], dict[str, Any]],
    post_with_backoff: Callable[..., None],
) -> None:
    profile_path = profile.expanduser().resolve() if profile is not None else None
    profile_data = read_profile(profile_path) if profile_path is not None else {}
    provider_override = resolve_cli_optional_string(field="provider", cli_value=provider)
    profile_provider_value = profile_data.get("provider")
    profile_provider: str | None = None
    if profile_provider_value is not None:
        if not isinstance(profile_provider_value, str) or not profile_provider_value.strip():
            raise NotifyConfigError("profile field 'provider' must be a non-empty string")
        profile_provider = profile_provider_value.strip()
    spool_dir_value = resolve_path_value(
        field="spool_dir",
        cli_value=spool_dir,
        profile_data=profile_data,
        profile_path=profile_path,
    )
    profile_url_env, profile_secret_ref = resolve_profile_webhook_source(profile_data)
    url_env_value = resolve_cli_optional_string(field="url_env", cli_value=url_env)
    if url_env_value is None:
        url_env_value = profile_url_env
    secret_ref_value = resolve_cli_optional_string(field="secret_ref", cli_value=secret_ref)
    if secret_ref_value is None:
        secret_ref_value = profile_secret_ref
    profile_tls_ca_bundle = resolve_optional_path_value(
        field="tls_ca_bundle",
        cli_value=tls_ca_bundle,
        profile_data=profile_data,
        profile_path=profile_path,
    )
    webhook_url = resolve_webhook_url(url=url, url_env=url_env_value, secret_ref=secret_ref_value)
    resolved_tls_ca_bundle = resolve_tls_ca_bundle(
        webhook_url=webhook_url,
        tls_ca_bundle=profile_tls_ca_bundle,
    )
    if not spool_dir_value.exists():
        raise NotifyConfigError(f"spool directory not found: {spool_dir_value}")
    failed = 0
    for path in sorted(spool_dir_value.glob("*.json")):
        try:
            body = json.loads(path.read_text(encoding="utf-8"))
            payload = body.get("payload")
            if not isinstance(payload, dict):
                raise NotifyConfigError(f"invalid spool payload file: {path}")
            spool_provider_value = body.get("provider")
            spool_provider: str | None = None
            if spool_provider_value is not None:
                if not isinstance(spool_provider_value, str) or not spool_provider_value.strip():
                    raise NotifyConfigError(f"invalid spool payload provider value: {path}")
                spool_provider = spool_provider_value.strip()
            provider_value = provider_override or spool_provider or profile_provider
            if not provider_value:
                raise NotifyConfigError(
                    f"spool payload missing provider: {path}. Pass --provider or include provider in spool files."
                )
            formatted = format_for_provider(provider_value, payload)
            post_with_backoff(
                webhook_url,
                formatted,
                tls_ca_bundle=resolved_tls_ca_bundle,
                connect_timeout=connect_timeout,
                read_timeout=read_timeout,
                retry_max=retry_max,
                retry_base_seconds=retry_base_seconds,
            )
            path.unlink()
        except NotifyError:
            failed += 1
            if fail_fast:
                raise
    if failed:
        raise NotifyDeliveryError(f"{failed} spool file(s) failed delivery")
