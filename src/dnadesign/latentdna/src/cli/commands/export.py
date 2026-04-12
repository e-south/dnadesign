"""
Export CLI commands for latentdna.
"""

from __future__ import annotations

import typer

from ...services.export_service import export_matrix, export_table
from ..common import emit, fail, resolve_format
from ..previews import preview_export

app = typer.Typer(help="Export commands for latentdna.")


@app.command("matrix")
def matrix(
    export_id: str = typer.Argument(...),
    workspace: str = typer.Option(..., "--workspace"),
    force: bool = typer.Option(False, "--force"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    format_name: str = typer.Option("text", "--format"),
    json_output: bool = typer.Option(False, "--json"),
    quiet: bool = typer.Option(False, "--quiet"),
) -> None:
    try:
        payload = (
            preview_export(workspace, export_id, command="export matrix", force=force)
            if dry_run
            else export_matrix(workspace, export_id, force=force).model_dump(mode="json")
        )
    except Exception as exc:
        fail(exc)
    emit(payload, format_name=resolve_format(json_output=json_output, format_name=format_name), quiet=quiet)


@app.command("table")
def table(
    export_id: str = typer.Argument(...),
    workspace: str = typer.Option(..., "--workspace"),
    force: bool = typer.Option(False, "--force"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    format_name: str = typer.Option("text", "--format"),
    json_output: bool = typer.Option(False, "--json"),
    quiet: bool = typer.Option(False, "--quiet"),
) -> None:
    try:
        payload = (
            preview_export(workspace, export_id, command="export table", force=force)
            if dry_run
            else export_table(workspace, export_id, force=force).model_dump(mode="json")
        )
    except Exception as exc:
        fail(exc)
    emit(payload, format_name=resolve_format(json_output=json_output, format_name=format_name), quiet=quiet)
