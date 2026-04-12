"""
Snapshot CLI commands for latentdna.
"""

from __future__ import annotations

import typer

from ...services.snapshot_service import build_snapshot
from ..common import emit, fail, resolve_format
from ..previews import preview_snapshot_build

app = typer.Typer(help="Snapshot commands for latentdna.")


@app.command("build")
def build(
    snapshot_id: str = typer.Argument(...),
    source: str = typer.Option(..., "--source"),
    workspace: str = typer.Option(..., "--workspace"),
    force: bool = typer.Option(False, "--force"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    format_name: str = typer.Option("text", "--format"),
    json_output: bool = typer.Option(False, "--json"),
    quiet: bool = typer.Option(False, "--quiet"),
) -> None:
    try:
        payload = (
            preview_snapshot_build(workspace, snapshot_id, source_id=source, force=force)
            if dry_run
            else build_snapshot(workspace, snapshot_id, source_id=source, force=force).model_dump(mode="json")
        )
    except Exception as exc:
        fail(exc)
    emit(payload, format_name=resolve_format(json_output=json_output, format_name=format_name), quiet=quiet)
