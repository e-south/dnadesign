"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/profiles/resolve.py

Profile-local value and path resolution helpers for setup flows.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from ..errors import NotifyConfigError


def resolve_cli_optional_string(*, field: str, cli_value: str | None) -> str | None:
    if cli_value is None:
        return None
    value = str(cli_value).strip()
    if not value:
        raise NotifyConfigError(f"{field} must be a non-empty string when provided")
    return value


def resolve_existing_file_path(*, field: str, path_value: Path) -> Path:
    resolved = path_value.expanduser().resolve()
    if not resolved.exists():
        raise NotifyConfigError(f"{field} file not found: {resolved}")
    if not resolved.is_file():
        raise NotifyConfigError(f"{field} path is not a file: {resolved}")
    return resolved
