"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/profiles/schema/reader.py

Profile file parsing and schema validation for notify profile payloads.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ...errors import NotifyConfigError
from ..policy import normalize_policy_name, supported_workflow_policies
from .contract import (
    PROFILE_ALLOWED_KEYS,
    PROFILE_REQUIRED_KEYS,
    PROFILE_VERSION,
    validate_events_source_config,
    validate_webhook_config,
)


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
