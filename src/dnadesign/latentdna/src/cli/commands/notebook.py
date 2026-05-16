"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/latentdna/src/cli/commands/notebook.py

Notebook CLI commands for latentdna.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import typer

from ...contracts.errors import ContractViolationError
from ...services.notebook_service import generate_notebook, smoke_workspace_notebook
from ..common import emit, fail, resolve_format
from ..previews import preview_notebook_generate

app = typer.Typer(help="Notebook scaffold commands for latentdna.")


@app.command("generate")
def generate(
    notebook_id: str = typer.Argument(...),
    workspace: str = typer.Option(..., "--workspace"),
    force: bool = typer.Option(False, "--force"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    format_name: str = typer.Option("text", "--format"),
    json_output: bool = typer.Option(False, "--json"),
    quiet: bool = typer.Option(False, "--quiet"),
) -> None:
    try:
        payload = (
            preview_notebook_generate(workspace, notebook_id, force=force)
            if dry_run
            else generate_notebook(workspace, notebook_id, force=force).model_dump(mode="json")
        )
    except Exception as exc:
        fail(exc)
    emit(payload, format_name=resolve_format(json_output=json_output, format_name=format_name), quiet=quiet)


@app.command("smoke")
def smoke(
    workspace: str = typer.Option(..., "--workspace"),
    format_name: str = typer.Option("text", "--format"),
    json_output: bool = typer.Option(False, "--json"),
    quiet: bool = typer.Option(False, "--quiet"),
) -> None:
    try:
        payload = smoke_workspace_notebook(workspace)
    except Exception as exc:
        fail(exc)
    emit(payload, format_name=resolve_format(json_output=json_output, format_name=format_name), quiet=quiet)
    if str(payload.get("status") or "").strip() == "error":
        raise typer.Exit(code=ContractViolationError.exit_code)
