"""
Artifact inventory CLI commands for latentdna.
"""

from __future__ import annotations

import typer

from ...services.run_service import list_runs, prune_run, show_run
from ..common import emit, fail, resolve_format
from ..previews import preview_runs_prune

app = typer.Typer(help="Artifact inventory commands for latentdna.")


@app.command("list")
def list_cmd(
    workspace: str = typer.Option(..., "--workspace"),
    format_name: str = typer.Option("text", "--format"),
    json_output: bool = typer.Option(False, "--json"),
    quiet: bool = typer.Option(False, "--quiet"),
) -> None:
    try:
        payload = list_runs(workspace)
    except Exception as exc:
        fail(exc)
    emit(payload, format_name=resolve_format(json_output=json_output, format_name=format_name), quiet=quiet)


@app.command("show")
def show(
    artifact_kind: str = typer.Argument(...),
    artifact_id: str = typer.Argument(...),
    workspace: str = typer.Option(..., "--workspace"),
    format_name: str = typer.Option("text", "--format"),
    json_output: bool = typer.Option(False, "--json"),
    quiet: bool = typer.Option(False, "--quiet"),
) -> None:
    try:
        payload = show_run(workspace, artifact_kind, artifact_id)
    except Exception as exc:
        fail(exc)
    emit(payload, format_name=resolve_format(json_output=json_output, format_name=format_name), quiet=quiet)


@app.command("prune")
def prune(
    artifact_kind: str = typer.Argument(...),
    artifact_id: str = typer.Argument(...),
    workspace: str = typer.Option(..., "--workspace"),
    force: bool = typer.Option(False, "--force"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    format_name: str = typer.Option("text", "--format"),
    json_output: bool = typer.Option(False, "--json"),
    quiet: bool = typer.Option(False, "--quiet"),
) -> None:
    try:
        payload = (
            preview_runs_prune(workspace, artifact_kind, artifact_id, force=force)
            if dry_run
            else prune_run(workspace, artifact_kind, artifact_id, force=force).model_dump(mode="json")
        )
    except Exception as exc:
        fail(exc)
    emit(payload, format_name=resolve_format(json_output=json_output, format_name=format_name), quiet=quiet)
