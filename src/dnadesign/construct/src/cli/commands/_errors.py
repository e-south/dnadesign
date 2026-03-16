"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/construct/src/cli/commands/_errors.py

Shared CLI error shaping helpers for construct.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import NoReturn

import typer


def exit_with_error(exc: Exception, *, code: int) -> NoReturn:
    """Render a concise CLI error and exit with a deterministic status code."""
    if isinstance(exc, OSError):
        detail = exc.strerror or str(exc)
        location = getattr(exc, "filename", None)
        message = f"{detail}: {location}" if location else detail
    else:
        message = str(exc)
    typer.echo(f"Error: {message}")
    raise typer.Exit(code) from exc
