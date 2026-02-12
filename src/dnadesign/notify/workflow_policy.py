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
_WORKFLOW_POLICY_DEFAULTS: dict[str, dict[str, Any]] = {}
_WORKFLOW_POLICY_ALIASES: dict[str, str] = {}

DEFAULT_WEBHOOK_ENV = "NOTIFY_WEBHOOK"
DEFAULT_PROFILE_PATH = Path("outputs/notify/generic/profile.json")


def _normalize_name(value: str | None, *, field: str) -> str:
    if value is None:
        raise NotifyConfigError(f"{field} must be a non-empty string when provided")
    name = str(value).strip().lower()
    if not name:
        raise NotifyConfigError(f"{field} must be a non-empty string when provided")
    return name


def register_workflow_policy(
    *,
    policy: str,
    defaults: dict[str, Any] | None = None,
    aliases: tuple[str, ...] = (),
) -> None:
    policy_name = _normalize_name(policy, field="policy")
    if policy_name in _WORKFLOW_POLICY_DEFAULTS:
        raise NotifyConfigError(f"policy '{policy_name}' is already registered")

    alias_names: list[str] = []
    for alias in aliases:
        alias_name = _normalize_name(alias, field="alias")
        if alias_name == policy_name:
            raise NotifyConfigError(f"alias '{alias_name}' cannot equal policy name '{policy_name}'")
        if alias_name in _WORKFLOW_POLICY_ALIASES or alias_name in _WORKFLOW_POLICY_DEFAULTS:
            raise NotifyConfigError(f"alias '{alias_name}' is already registered")
        alias_names.append(alias_name)

    _WORKFLOW_POLICY_DEFAULTS[policy_name] = dict(defaults or {})
    for alias_name in alias_names:
        _WORKFLOW_POLICY_ALIASES[alias_name] = policy_name


def default_profile_path_for_tool(tool_name: str | None) -> Path:
    if tool_name is None:
        return DEFAULT_PROFILE_PATH
    return Path("outputs") / "notify" / tool_name / "profile.json"


def normalize_policy_name(policy: str | None) -> str | None:
    if policy is None:
        return None
    value = _normalize_name(policy, field="policy")
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


register_workflow_policy(
    policy="densegen",
    defaults=dict(_DENSEGEN_PROFILE_PRESET),
)
register_workflow_policy(
    policy="infer_evo2",
    defaults={
        "only_actions": "attach,materialize",
        "only_tools": "infer",
        "include_args": False,
        "include_context": False,
        "include_raw_event": False,
    },
    aliases=("infer-evo2",),
)
register_workflow_policy(
    policy="generic",
    defaults={},
)
