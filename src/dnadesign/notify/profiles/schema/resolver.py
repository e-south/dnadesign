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

from ...errors import NotifyConfigError
from ...events.source import normalize_tool_name
from .contract import PROFILE_VERSION, WEBHOOK_SOURCES


def resolve_profile_webhook_source(profile_data: dict[str, Any]) -> tuple[str | None, str | None]:
    if not profile_data:
        return None, None
    version = profile_data.get("profile_version")
    if version != PROFILE_VERSION:
        raise NotifyConfigError(f"profile_version must be {PROFILE_VERSION}; found {version!r}")
    webhook = profile_data.get("webhook")
    if not isinstance(webhook, dict):
        raise NotifyConfigError("profile field 'webhook' must be an object")
    source = str(webhook.get("source") or "").strip().lower()
    ref = str(webhook.get("ref") or "").strip()
    if source not in WEBHOOK_SOURCES:
        allowed = ", ".join(sorted(WEBHOOK_SOURCES))
        raise NotifyConfigError(f"profile field 'webhook.source' must be one of: {allowed}")
    if not ref:
        raise NotifyConfigError("profile field 'webhook.ref' must be a non-empty string")
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
