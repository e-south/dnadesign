"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/profiles/schema/contract.py

Schema constants and field-level validators for notify profile payloads.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any

from ...errors import NotifyConfigError
from ...events.source import normalize_tool_name

PROFILE_VERSION = 2
PROFILE_REQUIRED_KEYS = {"provider", "events", "webhook"}
PROFILE_ALLOWED_KEYS = {
    "profile_version",
    "provider",
    "events",
    "events_source",
    "webhook",
    "cursor",
    "only_actions",
    "only_tools",
    "progress_step_pct",
    "progress_min_seconds",
    "spool_dir",
    "include_args",
    "include_context",
    "include_raw_event",
    "policy",
    "tls_ca_bundle",
}
WEBHOOK_SOURCES = {"env", "secret_ref"}


def validate_events_source_config(data: dict[str, Any]) -> None:
    events_source = data.get("events_source")
    if events_source is None:
        return
    if not isinstance(events_source, dict):
        raise NotifyConfigError("profile field 'events_source' must be an object")
    unknown = sorted(set(events_source.keys()) - {"tool", "config"})
    if unknown:
        raise NotifyConfigError(f"profile field 'events_source' contains unsupported keys: {', '.join(unknown)}")
    tool = events_source.get("tool")
    config = events_source.get("config")
    if not isinstance(tool, str) or not tool.strip():
        raise NotifyConfigError("profile field 'events_source.tool' must be a non-empty string")
    if not isinstance(config, str) or not config.strip():
        raise NotifyConfigError("profile field 'events_source.config' must be a non-empty string path")
    normalize_tool_name(tool)


def validate_webhook_config(data: dict[str, Any]) -> None:
    webhook = data.get("webhook")
    if not isinstance(webhook, dict):
        raise NotifyConfigError("profile field 'webhook' must be an object")
    unknown = sorted(set(webhook.keys()) - {"source", "ref"})
    if unknown:
        raise NotifyConfigError(f"profile field 'webhook' contains unsupported keys: {', '.join(unknown)}")
    source = webhook.get("source")
    ref = webhook.get("ref")
    if not isinstance(source, str) or not source.strip():
        raise NotifyConfigError("profile field 'webhook.source' must be a non-empty string")
    source_norm = source.strip().lower()
    if source_norm not in WEBHOOK_SOURCES:
        allowed = ", ".join(sorted(WEBHOOK_SOURCES))
        raise NotifyConfigError(f"profile field 'webhook.source' must be one of: {allowed}")
    if not isinstance(ref, str) or not ref.strip():
        raise NotifyConfigError("profile field 'webhook.ref' must be a non-empty string")
