"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/profiles/schema.py

Profile schema parsing, validation, and source-resolution helpers for notify.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..errors import NotifyConfigError
from ..events.source import normalize_tool_name
from .policy import normalize_policy_name, supported_workflow_policies

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


def read_profile(profile_path: Path) -> dict[str, Any]:
    if not profile_path.exists():
        raise NotifyConfigError(f"profile file not found: {profile_path}")
    try:
        data = json.loads(profile_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise NotifyConfigError(f"profile file is not valid JSON: {profile_path}") from exc
    if not isinstance(data, dict):
        raise NotifyConfigError("profile file must contain a JSON object")

    if "url" in data:
        raise NotifyConfigError("profile must not store plain webhook URLs; use webhook refs")
    if "preset" in data:
        raise NotifyConfigError("legacy profile field 'preset' is not supported; use 'policy' with explicit filters")

    version = data.get("profile_version")
    if version != PROFILE_VERSION:
        raise NotifyConfigError(f"profile_version must be {PROFILE_VERSION}; found {version!r}")
    unknown = sorted(set(data.keys()) - PROFILE_ALLOWED_KEYS)
    if unknown:
        raise NotifyConfigError(f"profile contains unsupported keys: {', '.join(unknown)}")
    for key in PROFILE_REQUIRED_KEYS:
        value = data.get(key)
        if key == "webhook":
            continue
        if not isinstance(value, str) or not value.strip():
            raise NotifyConfigError(f"profile missing required non-empty string field '{key}'")
    validate_webhook_config(data)
    validate_events_source_config(data)

    for key in ("events", "cursor", "spool_dir", "tls_ca_bundle"):
        value = data.get(key)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise NotifyConfigError(f"profile field '{key}' must be a non-empty string path when provided")

    for key in ("only_actions", "only_tools", "policy"):
        value = data.get(key)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise NotifyConfigError(f"profile field '{key}' must be a non-empty string when provided")
    policy_name = normalize_policy_name(data.get("policy"))
    if policy_name is not None and policy_name not in supported_workflow_policies():
        allowed = ", ".join(supported_workflow_policies())
        raise NotifyConfigError(f"profile field 'policy' must be one of: {allowed}")
    if policy_name is not None and policy_name != "generic":
        only_actions = data.get("only_actions")
        only_tools = data.get("only_tools")
        if not isinstance(only_actions, str) or not only_actions.strip():
            raise NotifyConfigError(
                f"profile policy '{policy_name}' requires explicit only_actions and only_tools fields in profile"
            )
        if not isinstance(only_tools, str) or not only_tools.strip():
            raise NotifyConfigError(
                f"profile policy '{policy_name}' requires explicit only_actions and only_tools fields in profile"
            )

    include_args = data.get("include_args")
    if include_args is not None and not isinstance(include_args, bool):
        raise NotifyConfigError("profile field 'include_args' must be a boolean when provided")
    include_context = data.get("include_context")
    if include_context is not None and not isinstance(include_context, bool):
        raise NotifyConfigError("profile field 'include_context' must be a boolean when provided")
    include_raw_event = data.get("include_raw_event")
    if include_raw_event is not None and not isinstance(include_raw_event, bool):
        raise NotifyConfigError("profile field 'include_raw_event' must be a boolean when provided")
    progress_step_pct = data.get("progress_step_pct")
    if progress_step_pct is not None:
        if not isinstance(progress_step_pct, int):
            raise NotifyConfigError("profile field 'progress_step_pct' must be an integer when provided")
        if progress_step_pct < 1 or progress_step_pct > 100:
            raise NotifyConfigError("profile field 'progress_step_pct' must be between 1 and 100 when provided")
    progress_min_seconds = data.get("progress_min_seconds")
    if progress_min_seconds is not None:
        if not isinstance(progress_min_seconds, (int, float)):
            raise NotifyConfigError("profile field 'progress_min_seconds' must be numeric when provided")
        if float(progress_min_seconds) <= 0.0:
            raise NotifyConfigError("profile field 'progress_min_seconds' must be > 0 when provided")
    return data


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
