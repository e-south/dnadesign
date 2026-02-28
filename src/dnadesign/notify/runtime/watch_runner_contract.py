"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/runtime/watch_runner_contract.py

Input-contract validation and option coercion for notify watch runner.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any

from ..errors import NotifyConfigError


def validate_watch_request_contract(
    *,
    profile: object,
    events: object,
    config: object,
    workspace: object,
    idle_timeout: float | None,
    poll_interval_seconds: float,
) -> bool:
    if idle_timeout is not None and float(idle_timeout) <= 0:
        raise NotifyConfigError("idle_timeout must be > 0 when provided")
    if float(poll_interval_seconds) <= 0:
        raise NotifyConfigError("poll_interval_seconds must be > 0")

    has_resolver_mode = config is not None or workspace is not None
    if profile is None and events is None and not has_resolver_mode:
        raise NotifyConfigError("pass --profile, --events, or --tool with --config/--workspace")
    if has_resolver_mode and (profile is not None or events is not None):
        raise NotifyConfigError("--config/--workspace cannot be combined with --profile or --events")
    if config is not None and workspace is not None:
        raise NotifyConfigError("pass either --config or --workspace, not both")
    return has_resolver_mode


def resolve_progress_step_pct(
    *,
    progress_step_pct: int | None,
    profile_data: dict[str, Any],
) -> int | None:
    raw_value: object
    if progress_step_pct is not None:
        raw_value = progress_step_pct
    else:
        raw_value = profile_data.get("progress_step_pct")
    if raw_value is None:
        return None
    try:
        parsed = int(raw_value)
    except (TypeError, ValueError) as exc:
        raise NotifyConfigError("progress_step_pct must be an integer between 1 and 100") from exc
    if parsed < 1 or parsed > 100:
        raise NotifyConfigError("progress_step_pct must be an integer between 1 and 100")
    return parsed


def resolve_progress_min_seconds(
    *,
    progress_min_seconds: float | None,
    profile_data: dict[str, Any],
) -> float | None:
    raw_value: object
    if progress_min_seconds is not None:
        raw_value = progress_min_seconds
    else:
        raw_value = profile_data.get("progress_min_seconds")
    if raw_value is None:
        return None
    try:
        parsed = float(raw_value)
    except (TypeError, ValueError) as exc:
        raise NotifyConfigError("progress_min_seconds must be a positive number") from exc
    if parsed <= 0.0:
        raise NotifyConfigError("progress_min_seconds must be a positive number")
    return parsed


def resolve_optional_profile_bool(
    *,
    cli_value: bool | None,
    profile_data: dict[str, Any],
    field: str,
) -> bool:
    if cli_value is not None:
        return bool(cli_value)
    profile_value = profile_data.get(field)
    if profile_value is None:
        return False
    return bool(profile_value)


def normalize_on_invalid_event_mode(on_invalid_event: str) -> str:
    on_invalid_event_mode = str(on_invalid_event or "").strip().lower()
    if on_invalid_event_mode not in {"error", "skip"}:
        raise NotifyConfigError(f"unsupported on-invalid-event mode '{on_invalid_event}'")
    return on_invalid_event_mode
