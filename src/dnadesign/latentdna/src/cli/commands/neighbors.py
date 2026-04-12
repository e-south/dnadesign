"""
Neighbor CLI commands for latentdna.
"""

from __future__ import annotations

import typer

from ...services.neighbors_service import fit_neighbors
from ..common import emit, fail, resolve_format
from ..previews import preview_neighbors_fit

app = typer.Typer(help="Neighbor commands for latentdna.")


@app.command("fit")
def fit(
    neighbor_id: str = typer.Argument(...),
    workspace: str = typer.Option(..., "--workspace"),
    view: str = typer.Option(..., "--view"),
    k: int = typer.Option(..., "--k"),
    metric: str | None = typer.Option(None, "--metric"),
    backend: str | None = typer.Option(None, "--backend"),
    sample: str | None = typer.Option(None, "--sample"),
    alignment: str | None = typer.Option(None, "--alignment"),
    seed: int | None = typer.Option(None, "--seed"),
    force: bool = typer.Option(False, "--force"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    format_name: str = typer.Option("text", "--format"),
    json_output: bool = typer.Option(False, "--json"),
    quiet: bool = typer.Option(False, "--quiet"),
) -> None:
    try:
        payload = (
            preview_neighbors_fit(
                workspace,
                neighbor_id,
                view_id=view,
                sample_id=sample,
                alignment_id=alignment,
                force=force,
            )
            if dry_run
            else fit_neighbors(
                workspace,
                neighbor_id,
                view_id=view,
                k=k,
                metric=metric,
                backend=backend,
                sample_id=sample,
                alignment_id=alignment,
                seed=seed,
                force=force,
            ).model_dump(mode="json")
        )
    except Exception as exc:
        fail(exc)
    emit(payload, format_name=resolve_format(json_output=json_output, format_name=format_name), quiet=quiet)
