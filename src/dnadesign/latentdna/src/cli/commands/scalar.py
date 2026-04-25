"""
Scalar CLI commands for latentdna.
"""

from __future__ import annotations

import typer

from ...services.scalar_service import derive_scalar
from ..common import emit, fail, resolve_format
from ..previews import preview_scalar_derive

app = typer.Typer(help="Scalar commands for latentdna.")


@app.command("derive")
def derive(
    scalar_id: str = typer.Argument(...),
    workspace: str = typer.Option(..., "--workspace"),
    force: bool = typer.Option(False, "--force"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    format_name: str = typer.Option("text", "--format"),
    json_output: bool = typer.Option(False, "--json"),
    quiet: bool = typer.Option(False, "--quiet"),
) -> None:
    try:
        payload = (
            preview_scalar_derive(workspace, scalar_id, force=force)
            if dry_run
            else derive_scalar(workspace, scalar_id, force=force).model_dump(mode="json")
        )
    except Exception as exc:
        fail(exc)
    emit(payload, format_name=resolve_format(json_output=json_output, format_name=format_name), quiet=quiet)
