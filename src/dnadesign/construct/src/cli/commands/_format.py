"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/construct/src/cli/commands/_format.py

Shared output-format helpers for construct CLI commands.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path

import typer

from ...errors import ConstructError


def normalize_json_value(value: object) -> object:
    if is_dataclass(value):
        return normalize_json_value(asdict(value))
    if hasattr(value, "model_dump"):
        return normalize_json_value(value.model_dump())  # type: ignore[no-any-return]
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {str(key): normalize_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalize_json_value(item) for item in value]
    return value


def echo_json(payload: object) -> None:
    typer.echo(json.dumps(normalize_json_value(payload), separators=(",", ":")))


def validate_output_format(value: str, *, allowed: tuple[str, ...] = ("text", "json")) -> str:
    normalized = str(value or "").strip().lower()
    if normalized not in allowed:
        choices = ", ".join(allowed)
        raise ConstructError(f"format must be one of: {choices}.")
    return normalized
