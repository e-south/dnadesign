"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/infer/src/cli/common.py

Shared CLI safety helpers for exit-code mapping and input/config guards.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import typer
from pydantic import ValidationError as PydanticValidationError

from ..errors import (
    CapabilityError,
    ConfigError,
    IOErrorInfer,
    ModelLoadError,
    RuntimeOOMError,
    UnsafeInputError,
    ValidationError,
    WriteBackError,
)
from .console import console


def exit_for(error: Exception) -> int:
    mapping = {
        ConfigError: 2,
        PydanticValidationError: 2,
        ValidationError: 3,
        ModelLoadError: 4,
        CapabilityError: 5,
        RuntimeOOMError: 6,
        WriteBackError: 7,
        IOErrorInfer: 8,
        UnsafeInputError: 8,
    }
    for error_type, code in mapping.items():
        if isinstance(error, error_type):
            return code
    return 1


def raise_cli_error(error: Exception) -> None:
    error_label = type(error).__name__
    console.print(f"[red]{error_label}: {error}[/red]")
    raise typer.Exit(code=exit_for(error))


def discovery_config(provided: Optional[Path]) -> Path:
    if provided:
        return provided.resolve()
    cwd_cfg = Path.cwd() / "config.yaml"
    if cwd_cfg.exists():
        return cwd_cfg.resolve()
    raise ConfigError("No config found. Pass --config or place config.yaml in the current directory.")


def guard_pickle(i_know: bool) -> None:
    allow = os.environ.get("INFER_ALLOW_PICKLE", "0").lower() in {"1", "true", "yes"}
    if not i_know and not allow:
        raise UnsafeInputError(
            "Refusing to load a .pt file without explicit consent. "
            "Re-run with --i-know-this-is-pickle or set INFER_ALLOW_PICKLE=1."
        )
