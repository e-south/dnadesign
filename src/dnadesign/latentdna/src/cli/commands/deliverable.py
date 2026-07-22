"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/latentdna/src/cli/commands/deliverable.py

Deliverable CLI commands for latentdna.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import typer

from ...services.deliverable_service import deliverable_status, list_deliverables, run_deliverable
from ..common import emit, emit_with_progress, fail, progress_sink_for_mode, resolve_format, resolve_progress_mode
from ..previews import preview_deliverable_run

app = typer.Typer(help="Deliverable commands for latentdna.")


@app.command("list")
def list_cmd(
    workspace: str = typer.Option(..., "--workspace"),
    format_name: str = typer.Option("text", "--format"),
    json_output: bool = typer.Option(False, "--json"),
    quiet: bool = typer.Option(False, "--quiet"),
) -> None:
    try:
        payload = list_deliverables(workspace)
    except Exception as exc:
        fail(exc)
    emit(payload, format_name=resolve_format(json_output=json_output, format_name=format_name), quiet=quiet)


@app.command("status")
def status(
    deliverable_id: str = typer.Argument(...),
    workspace: str = typer.Option(..., "--workspace"),
    format_name: str = typer.Option("text", "--format"),
    json_output: bool = typer.Option(False, "--json"),
    quiet: bool = typer.Option(False, "--quiet"),
) -> None:
    try:
        payload = deliverable_status(workspace, deliverable_id).model_dump(mode="json")
    except Exception as exc:
        fail(exc)
    emit(payload, format_name=resolve_format(json_output=json_output, format_name=format_name), quiet=quiet)


@app.command("run")
def run(
    deliverable_id: str = typer.Argument(...),
    workspace: str = typer.Option(..., "--workspace"),
    force: bool = typer.Option(False, "--force"),
    allow_memory_overage: bool = typer.Option(False, "--allow-memory-overage"),
    progress: str = typer.Option("none", "--progress"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    format_name: str = typer.Option("text", "--format"),
    json_output: bool = typer.Option(False, "--json"),
    quiet: bool = typer.Option(False, "--quiet"),
) -> None:
    try:
        progress_mode = resolve_progress_mode(progress)
        payload = (
            preview_deliverable_run(workspace, deliverable_id, force=force)
            if dry_run
            else run_deliverable(
                workspace,
                deliverable_id,
                force=force,
                allow_memory_overage=allow_memory_overage,
                event_sink=progress_sink_for_mode(progress_mode),
            ).model_dump(mode="json")
        )
    except Exception as exc:
        fail(exc)
    if dry_run:
        emit(payload, format_name=resolve_format(json_output=json_output, format_name=format_name), quiet=quiet)
        return
    emit_with_progress(
        payload,
        progress_mode=progress_mode,
        format_name=resolve_format(json_output=json_output, format_name=format_name),
        quiet=quiet,
    )
