"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/latentdna/src/cli/commands/validate.py

Validation CLI commands for latentdna.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import typer

from ..common import emit, fail, resolve_format

app = typer.Typer(help="Validation commands for latentdna.")


@app.command("workspace")
def workspace(
    workspace: str = typer.Option(..., "--workspace"),
    deep: bool = typer.Option(False, "--deep"),
    format_name: str = typer.Option("text", "--format"),
    json_output: bool = typer.Option(False, "--json"),
    quiet: bool = typer.Option(False, "--quiet"),
) -> None:
    try:
        from ...services.validation_service import validate_workspace

        payload = validate_workspace(workspace, deep=deep)
    except Exception as exc:
        fail(exc)
    emit(payload, format_name=resolve_format(json_output=json_output, format_name=format_name), quiet=quiet)
