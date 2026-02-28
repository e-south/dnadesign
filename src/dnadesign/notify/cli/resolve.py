"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/cli/resolve.py

Shared CLI resolution helpers for notify option values and USR events paths.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from ..errors import NotifyConfigError


def resolve_cli_optional_string(*, field: str, cli_value: str | None) -> str | None:
    if cli_value is None:
        return None
    value = str(cli_value).strip()
    if not value:
        raise NotifyConfigError(f"{field} must be a non-empty string when provided")
    return value


def resolve_string_value(*, field: str, cli_value: str | None, profile_data: dict[str, Any]) -> str:
    if cli_value is not None:
        value = str(cli_value).strip()
        if value:
            return value
        raise NotifyConfigError(f"{field} must be a non-empty string")
    profile_value = profile_data.get(field)
    if isinstance(profile_value, str) and profile_value.strip():
        return profile_value
    raise NotifyConfigError(f"{field} is required; pass --{field.replace('_', '-')} or provide it in --profile")


def resolve_optional_string_value(*, field: str, cli_value: str | None, profile_data: dict[str, Any]) -> str | None:
    if cli_value is not None:
        value = str(cli_value).strip()
        if not value:
            raise NotifyConfigError(f"{field} must be a non-empty string when provided")
        return value
    profile_value = profile_data.get(field)
    if profile_value is None:
        return None
    if isinstance(profile_value, str) and profile_value.strip():
        return profile_value
    raise NotifyConfigError(f"profile field '{field}' must be a non-empty string when provided")


def resolve_path_value(
    *,
    field: str,
    cli_value: Path | None,
    profile_data: dict[str, Any],
    profile_path: Path | None,
) -> Path:
    if cli_value is not None:
        return cli_value
    profile_value = profile_data.get(field)
    if isinstance(profile_value, str) and profile_value.strip():
        resolved = Path(profile_value)
        if profile_path is not None and not resolved.is_absolute():
            return profile_path.parent / resolved
        return resolved
    raise NotifyConfigError(f"{field} is required; pass --{field.replace('_', '-')} or provide it in --profile")


def resolve_optional_path_value(
    *,
    field: str,
    cli_value: Path | None,
    profile_data: dict[str, Any],
    profile_path: Path | None,
) -> Path | None:
    if cli_value is not None:
        return cli_value
    profile_value = profile_data.get(field)
    if profile_value is None:
        return None
    if isinstance(profile_value, str) and profile_value.strip():
        resolved = Path(profile_value)
        if profile_path is not None and not resolved.is_absolute():
            return profile_path.parent / resolved
        return resolved
    raise NotifyConfigError(f"profile field '{field}' must be a non-empty string path when provided")


def resolve_existing_file_path(*, field: str, path_value: Path) -> Path:
    resolved = path_value.expanduser().resolve()
    if not resolved.exists():
        raise NotifyConfigError(f"{field} file not found: {resolved}")
    if not resolved.is_file():
        raise NotifyConfigError(f"{field} path is not a file: {resolved}")
    return resolved


def events_path_hint() -> str:
    return (
        "Find the USR .events.log path with `uv run dense inspect run --usr-events-path` from the DenseGen "
        "workspace, or use `uv run dense inspect run --usr-events-path -c <config.yaml>`."
    )


def validate_usr_events_probe(
    events_path: Path,
    *,
    validate_usr_event: Callable[..., None],
) -> None:
    with events_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            raw = str(line).strip()
            if not raw:
                continue
            try:
                event = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise NotifyConfigError(
                    "events must be a USR .events.log JSONL file; first non-empty line is not JSON "
                    f"(line {line_no}) in {events_path}. {events_path_hint()}"
                ) from exc
            try:
                validate_usr_event(event, allow_unknown_version=True)
            except NotifyConfigError as exc:
                raise NotifyConfigError(
                    f"events file does not look like USR .events.log (line {line_no}) in {events_path}: "
                    f"{exc}. {events_path_hint()}"
                ) from exc
            return


def resolve_usr_events_path(
    events_path: Path,
    *,
    require_exists: bool = True,
    validate_usr_event: Callable[..., None],
) -> Path:
    resolved = events_path.expanduser().resolve()
    if resolved.suffix.lower() in {".yaml", ".yml"}:
        raise NotifyConfigError(
            f"events must point to a USR .events.log JSONL file, not a config file: {resolved}. {events_path_hint()}"
        )
    if not require_exists and not resolved.exists():
        return resolved
    if not resolved.exists():
        raise NotifyConfigError(f"events file not found: {resolved}. {events_path_hint()}")
    if not resolved.is_file():
        raise NotifyConfigError(f"events path is not a file: {resolved}")
    validate_usr_events_probe(resolved, validate_usr_event=validate_usr_event)
    return resolved
