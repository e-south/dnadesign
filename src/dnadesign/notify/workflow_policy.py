"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/workflow_policy.py

Workflow policy defaults and profile namespacing helpers for notify.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .errors import NotifyConfigError

_DENSEGEN_PROFILE_PRESET = {
    "only_actions": "densegen_health,densegen_flush_failed,materialize",
    "only_tools": "densegen",
    "include_args": False,
    "include_context": False,
    "include_raw_event": False,
}

_WORKFLOW_POLICY_DEFAULTS: dict[str, dict[str, Any]] = {
    "densegen": dict(_DENSEGEN_PROFILE_PRESET),
    "infer_evo2": {
        "only_actions": "attach,materialize",
        "only_tools": "infer",
        "include_args": False,
        "include_context": False,
        "include_raw_event": False,
    },
    "generic": {},
}

_WORKFLOW_POLICY_ALIASES = {
    "infer-evo2": "infer_evo2",
}

DEFAULT_WEBHOOK_ENV = "NOTIFY_WEBHOOK"
DEFAULT_PROFILE_PATH = Path("outputs/notify/generic/profile.json")


def default_profile_path_for_tool(tool_name: str | None) -> Path:
    if tool_name is None:
        return DEFAULT_PROFILE_PATH
    return Path("outputs") / "notify" / tool_name / "profile.json"


def normalize_policy_name(policy: str | None) -> str | None:
    if policy is None:
        return None
    value = str(policy).strip().lower()
    if not value:
        raise NotifyConfigError("policy must be a non-empty string when provided")
    return _WORKFLOW_POLICY_ALIASES.get(value, value)


def supported_workflow_policies() -> tuple[str, ...]:
    return tuple(sorted(_WORKFLOW_POLICY_DEFAULTS))


def resolve_workflow_policy(*, policy: str | None) -> str | None:
    policy_name = normalize_policy_name(policy)
    if policy_name is None:
        return None
    if policy_name not in _WORKFLOW_POLICY_DEFAULTS:
        allowed = ", ".join(supported_workflow_policies())
        raise NotifyConfigError(f"unsupported policy '{policy}'. Supported values: {allowed}")
    return policy_name


def policy_defaults(policy: str) -> dict[str, Any]:
    policy_name = resolve_workflow_policy(policy=policy)
    if policy_name is None:
        raise NotifyConfigError("policy is required")
    return dict(_WORKFLOW_POLICY_DEFAULTS[policy_name])
