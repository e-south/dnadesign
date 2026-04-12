"""
Projection CLI commands for latentdna.
"""

from __future__ import annotations

import typer

from ...services.projection_service import fit_projection
from ..common import emit, fail, resolve_format
from ..previews import preview_projection_fit

app = typer.Typer(help="Projection commands for latentdna.")


@app.command("fit")
def fit(
    view_id: str = typer.Argument(...),
    workspace: str = typer.Option(..., "--workspace"),
    sample: str = typer.Option(..., "--sample"),
    run_id: str = typer.Option(..., "--run-id"),
    metric: str | None = typer.Option(None, "--metric"),
    seed: int = typer.Option(17, "--seed"),
    force: bool = typer.Option(False, "--force"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    format_name: str = typer.Option("text", "--format"),
    json_output: bool = typer.Option(False, "--json"),
    quiet: bool = typer.Option(False, "--quiet"),
) -> None:
    try:
        payload = (
            preview_projection_fit(workspace, run_id, view_id=view_id, sample_id=sample, force=force)
            if dry_run
            else fit_projection(
                workspace,
                view_id,
                projection_id=run_id,
                sample_id=sample,
                metric=metric,
                seed=seed,
                force=force,
            ).model_dump(mode="json")
        )
    except Exception as exc:
        fail(exc)
    emit(payload, format_name=resolve_format(json_output=json_output, format_name=format_name), quiet=quiet)
