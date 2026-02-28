"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/runtime/watch_runner_resolution.py

Profile, events-source, and webhook input resolution for notify watch runner.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from ..errors import NotifyConfigError


@dataclass(frozen=True)
class WatchResolverMode:
    profile_path: Path | None
    tool_value_for_events: str | None
    tool_name_for_config_mode: str | None
    config_path_for_config_mode: Path | None
    events_path_from_cli_tool_config: Path | None


def resolve_watch_mode(
    *,
    has_resolver_mode: bool,
    profile: Path | None,
    tool: str | None,
    config: Path | None,
    workspace: str | None,
    normalize_tool_name: Callable[[str | None], str | None],
    resolve_tool_events_path: Callable[..., tuple[Path, str | None]],
    resolve_tool_workspace_config: Callable[..., Path],
    default_profile_path_for_tool: Callable[[str | None], Path],
) -> WatchResolverMode:
    tool_value_for_events = tool
    tool_name_for_config_mode: str | None = None
    config_path_for_config_mode: Path | None = None
    events_path_from_cli_tool_config: Path | None = None
    if has_resolver_mode:
        tool_name = normalize_tool_name(tool)
        if tool_name is None:
            raise NotifyConfigError("--config/--workspace requires --tool")
        if config is not None:
            config_path = config.expanduser().resolve()
            setup_hint = f"uv run notify setup slack --tool {tool_name} --config {config_path}"
        else:
            workspace_name = str(workspace or "").strip()
            if not workspace_name:
                raise NotifyConfigError("--workspace must be a non-empty string")
            config_path = resolve_tool_workspace_config(
                tool=tool_name,
                workspace=workspace_name,
                search_start=Path.cwd(),
            )
            if not isinstance(config_path, Path):
                config_path = Path(config_path)
            config_path = config_path.expanduser().resolve()
            setup_hint = f"uv run notify setup slack --tool {tool_name} --workspace {workspace_name}"
        events_path_from_cli_tool_config, _default_policy = resolve_tool_events_path(tool=tool_name, config=config_path)
        auto_profile_path = (config_path.parent / default_profile_path_for_tool(tool_name)).resolve()
        if not auto_profile_path.exists():
            raise NotifyConfigError(
                f"profile not found for tool '{tool_name}' at {auto_profile_path}. Run `{setup_hint}` once."
            )
        profile_path = auto_profile_path
        tool_value_for_events = None
        tool_name_for_config_mode = tool_name
        config_path_for_config_mode = config_path
    else:
        profile_path = profile.expanduser().resolve() if profile is not None else None
    return WatchResolverMode(
        profile_path=profile_path,
        tool_value_for_events=tool_value_for_events,
        tool_name_for_config_mode=tool_name_for_config_mode,
        config_path_for_config_mode=config_path_for_config_mode,
        events_path_from_cli_tool_config=events_path_from_cli_tool_config,
    )


def validate_profile_events_source_match(
    *,
    mode: WatchResolverMode,
    profile_data: dict[str, Any],
    resolve_profile_events_source: Callable[..., tuple[str, Path] | None],
) -> None:
    if mode.tool_name_for_config_mode is None or mode.config_path_for_config_mode is None:
        return
    profile_events_source = resolve_profile_events_source(profile_data=profile_data, profile_path=mode.profile_path)
    if profile_events_source is None:
        return
    source_tool, source_config = profile_events_source
    if source_tool != mode.tool_name_for_config_mode or source_config != mode.config_path_for_config_mode:
        raise NotifyConfigError(
            "profile events_source does not match --tool/--config (or resolved --tool/--workspace); "
            f"expected tool={mode.tool_name_for_config_mode!r} config={mode.config_path_for_config_mode}, "
            f"found tool={source_tool!r} config={source_config}"
        )


def resolve_events_path(
    *,
    events: Path | None,
    mode: WatchResolverMode,
    profile_data: dict[str, Any],
    resolve_path_value: Callable[..., Path],
    resolve_profile_events_source: Callable[..., tuple[str, Path] | None],
    resolve_tool_events_path: Callable[..., tuple[Path, str | None]],
    resolve_usr_events_path: Callable[..., Path],
) -> Path:
    events_path = resolve_path_value(
        field="events",
        cli_value=events,
        profile_data=profile_data,
        profile_path=mode.profile_path,
    )
    if mode.events_path_from_cli_tool_config is not None:
        return resolve_usr_events_path(mode.events_path_from_cli_tool_config, require_exists=False)
    if events is None:
        profile_events_source = resolve_profile_events_source(profile_data=profile_data, profile_path=mode.profile_path)
        if profile_events_source is not None:
            source_tool, source_config = profile_events_source
            resolved_events_path, _default_policy = resolve_tool_events_path(
                tool=source_tool,
                config=source_config,
            )
            return resolve_usr_events_path(resolved_events_path, require_exists=False)
    return events_path


def resolve_webhook_sources(
    *,
    profile_data: dict[str, Any],
    url_env: str | None,
    secret_ref: str | None,
    tls_ca_bundle: Path | None,
    profile_path: Path | None,
    resolve_profile_webhook_source: Callable[[dict[str, Any]], tuple[str | None, str | None]],
    resolve_cli_optional_string: Callable[..., str | None],
    resolve_optional_path_value: Callable[..., Path | None],
) -> tuple[str | None, str | None, Path | None]:
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
    return url_env_value, secret_ref_value, profile_tls_ca_bundle


def resolve_webhook_delivery(
    *,
    dry_run: bool,
    url: str | None,
    url_env_value: str | None,
    secret_ref_value: str | None,
    provider_value: str,
    profile_tls_ca_bundle: Path | None,
    resolve_webhook_url: Callable[..., str],
    validate_provider_webhook_url: Callable[..., None],
    resolve_tls_ca_bundle: Callable[..., Path | None],
) -> tuple[str | None, Path | None]:
    if dry_run:
        return None, None
    webhook_url = resolve_webhook_url(url=url, url_env=url_env_value, secret_ref=secret_ref_value)
    validate_provider_webhook_url(provider=provider_value, webhook_url=webhook_url)
    resolved_tls_ca_bundle = resolve_tls_ca_bundle(
        webhook_url=webhook_url,
        tls_ca_bundle=profile_tls_ca_bundle,
    )
    return webhook_url, resolved_tls_ca_bundle
