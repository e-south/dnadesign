"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/runtime/watch_runner_inputs.py

Resolution of watch-runner runtime inputs before entering the event loop.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .watch_runner_contract import (
    normalize_on_invalid_event_mode,
    resolve_optional_profile_bool,
    resolve_progress_min_seconds,
    resolve_progress_step_pct,
)
from .watch_runner_resolution import (
    WatchResolverMode,
    resolve_events_path,
    resolve_webhook_delivery,
    resolve_webhook_sources,
)


@dataclass(frozen=True)
class WatchRunnerInputs:
    provider_value: str
    events_path: Path
    cursor_path: Path | None
    action_filter: set[str]
    tool_filter: set[str]
    progress_step_pct_value: int | None
    progress_min_seconds_value: float | None
    spool_dir_value: Path | None
    include_args_value: bool
    include_context_value: bool
    include_raw_event_value: bool
    on_invalid_event_mode: str
    webhook_url: str | None
    resolved_tls_ca_bundle: Path | None


def resolve_watch_runner_inputs(
    *,
    mode: WatchResolverMode,
    profile_data: dict[str, Any],
    provider: str | None,
    url: str | None,
    url_env: str | None,
    secret_ref: str | None,
    tls_ca_bundle: Path | None,
    events: Path | None,
    cursor: Path | None,
    only_actions: str | None,
    only_tools: str | None,
    progress_step_pct: int | None,
    progress_min_seconds: float | None,
    on_invalid_event: str,
    spool_dir: Path | None,
    include_args: bool | None,
    include_context: bool | None,
    include_raw_event: bool | None,
    dry_run: bool,
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
    validate_provider_webhook_url: Callable[..., None],
    split_csv: Callable[[str | None], list[str]],
) -> WatchRunnerInputs:
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
    include_args_value = bool(
        resolve_optional_profile_bool(
            cli_value=include_args,
            profile_data=profile_data,
            field="include_args",
        )
    )
    include_context_value = bool(
        resolve_optional_profile_bool(
            cli_value=include_context,
            profile_data=profile_data,
            field="include_context",
        )
    )
    include_raw_event_value = bool(
        resolve_optional_profile_bool(
            cli_value=include_raw_event,
            profile_data=profile_data,
            field="include_raw_event",
        )
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

    return WatchRunnerInputs(
        provider_value=provider_value,
        events_path=events_path,
        cursor_path=cursor_path,
        action_filter=set(split_csv(only_actions_value)),
        tool_filter=set(split_csv(only_tools_value)),
        progress_step_pct_value=progress_step_pct_value,
        progress_min_seconds_value=progress_min_seconds_value,
        spool_dir_value=spool_dir_value,
        include_args_value=include_args_value,
        include_context_value=include_context_value,
        include_raw_event_value=include_raw_event_value,
        on_invalid_event_mode=normalize_on_invalid_event_mode(on_invalid_event),
        webhook_url=webhook_url,
        resolved_tls_ca_bundle=resolved_tls_ca_bundle,
    )
