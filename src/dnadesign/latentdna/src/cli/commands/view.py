"""
View CLI commands for latentdna.
"""

from __future__ import annotations

import typer

from ...services.view_service import derive_view, materialize_view, reduce_view, view_stats
from ..common import emit, fail, resolve_format
from ..previews import preview_view_derive, preview_view_materialize, preview_view_reduce

app = typer.Typer(help="View commands for latentdna.")


@app.command("materialize")
def materialize(
    view_id: str = typer.Argument(...),
    workspace: str = typer.Option(..., "--workspace"),
    force: bool = typer.Option(False, "--force"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    format_name: str = typer.Option("text", "--format"),
    json_output: bool = typer.Option(False, "--json"),
    quiet: bool = typer.Option(False, "--quiet"),
) -> None:
    try:
        payload = (
            preview_view_materialize(workspace, view_id, force=force)
            if dry_run
            else materialize_view(workspace, view_id, force=force).model_dump(mode="json")
        )
    except Exception as exc:
        fail(exc)
    emit(payload, format_name=resolve_format(json_output=json_output, format_name=format_name), quiet=quiet)


@app.command("derive")
def derive(
    view_id: str = typer.Argument(...),
    workspace: str = typer.Option(..., "--workspace"),
    force: bool = typer.Option(False, "--force"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    format_name: str = typer.Option("text", "--format"),
    json_output: bool = typer.Option(False, "--json"),
    quiet: bool = typer.Option(False, "--quiet"),
) -> None:
    try:
        payload = (
            preview_view_derive(workspace, view_id, force=force)
            if dry_run
            else derive_view(workspace, view_id, force=force).model_dump(mode="json")
        )
    except Exception as exc:
        fail(exc)
    emit(payload, format_name=resolve_format(json_output=json_output, format_name=format_name), quiet=quiet)


@app.command("reduce")
def reduce(
    view_id: str = typer.Argument(...),
    workspace: str = typer.Option(..., "--workspace"),
    run_id: str = typer.Option(..., "--run-id"),
    dims: int = typer.Option(..., "--dims"),
    sample: str | None = typer.Option(None, "--sample"),
    alignment: str | None = typer.Option(None, "--alignment"),
    reduced_view_id: str | None = typer.Option(None, "--reduced-view-id"),
    allow_memory_overage: bool = typer.Option(False, "--allow-memory-overage"),
    force: bool = typer.Option(False, "--force"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    format_name: str = typer.Option("text", "--format"),
    json_output: bool = typer.Option(False, "--json"),
    quiet: bool = typer.Option(False, "--quiet"),
) -> None:
    try:
        payload = (
            preview_view_reduce(
                workspace,
                view_id,
                reducer_id=run_id,
                sample_id=sample,
                alignment_id=alignment,
                reduced_view_id=reduced_view_id,
                force=force,
            )
            if dry_run
            else reduce_view(
                workspace,
                view_id,
                reducer_id=run_id,
                dims=dims,
                sample_id=sample,
                alignment_id=alignment,
                reduced_view_id=reduced_view_id,
                allow_memory_overage=allow_memory_overage,
                force=force,
            ).model_dump(mode="json")
        )
    except Exception as exc:
        fail(exc)
    emit(payload, format_name=resolve_format(json_output=json_output, format_name=format_name), quiet=quiet)


@app.command("stats")
def stats(
    view_id: str = typer.Argument(...),
    workspace: str = typer.Option(..., "--workspace"),
    format_name: str = typer.Option("text", "--format"),
    json_output: bool = typer.Option(False, "--json"),
    quiet: bool = typer.Option(False, "--quiet"),
) -> None:
    try:
        payload = view_stats(workspace, view_id)
    except Exception as exc:
        fail(exc)
    emit(payload, format_name=resolve_format(json_output=json_output, format_name=format_name), quiet=quiet)
