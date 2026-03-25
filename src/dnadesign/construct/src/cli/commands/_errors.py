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

from ._format import echo_json


def _augment_message(message: str) -> str:
    if message.startswith("Input dataset not initialized:") or message.startswith("Template dataset not initialized:"):
        return (
            f"{message} Seed or import the required dataset before runtime validation or run. "
            "For packaged demos, use ./runbook.sh --mode seed; otherwise use construct seed "
            "or construct seed import-manifest."
        )
    return message


def _json_error_requested(output_format: str | None) -> bool:
    return str(output_format or "").strip().lower() == "json"


def exit_with_error(exc: Exception, *, code: int, output_format: str | None = None) -> NoReturn:
    """Render a concise CLI error and exit with a deterministic status code."""
    if isinstance(exc, OSError):
        detail = exc.strerror or str(exc)
        location = getattr(exc, "filename", None)
        message = f"{detail}: {location}" if location else detail
    else:
        message = str(exc)
    message = _augment_message(message)
    if _json_error_requested(output_format):
        echo_json(
            {
                "status": "error",
                "code": code,
                "error": message,
                "error_type": exc.__class__.__name__,
            }
        )
        raise typer.Exit(code) from exc
    typer.echo(f"Error: {message}")
    raise typer.Exit(code) from exc
