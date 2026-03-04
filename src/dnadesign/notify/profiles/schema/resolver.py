"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/profiles/schema/resolver.py

Profile source-resolution helpers for webhook and events-source fields.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dnadesign._contracts.notify_webhook_profile import parse_notify_profile_webhook

from ...errors import NotifyConfigError
from ...events.source import normalize_tool_name
from .contract import PROFILE_VERSION, WEBHOOK_SOURCES


def resolve_profile_webhook_source(profile_data: dict[str, Any]) -> tuple[str | None, str | None]:
    if not profile_data:
        return None, None
    try:
        source, ref = parse_notify_profile_webhook(
            profile_data,
            required_profile_version=PROFILE_VERSION,
            allowed_sources=WEBHOOK_SOURCES,
        )
    except ValueError as exc:
        raise NotifyConfigError(str(exc)) from exc
    if source == "env":
        return ref, None
    return None, ref


def resolve_profile_events_source(
    *,
    profile_data: dict[str, Any],
    profile_path: Path | None,
) -> tuple[str, Path] | None:
    events_source = profile_data.get("events_source")
    if events_source is None:
        return None
    if not isinstance(events_source, dict):
        raise NotifyConfigError("profile field 'events_source' must be an object")
    tool_raw = events_source.get("tool")
    config_raw = events_source.get("config")
    if not isinstance(tool_raw, str) or not tool_raw.strip():
        raise NotifyConfigError("profile field 'events_source.tool' must be a non-empty string")
    if not isinstance(config_raw, str) or not config_raw.strip():
        raise NotifyConfigError("profile field 'events_source.config' must be a non-empty string path")
    tool_name = normalize_tool_name(tool_raw)
    if tool_name is None:
        raise NotifyConfigError("profile field 'events_source.tool' must be a non-empty string")
    config_path = Path(config_raw).expanduser()
    if profile_path is not None and not config_path.is_absolute():
        config_path = profile_path.parent / config_path
    return tool_name, config_path.resolve()
