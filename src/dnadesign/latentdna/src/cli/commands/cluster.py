"""
Cluster CLI commands for latentdna.
"""

from __future__ import annotations

import typer

from ...services.cluster_service import fit_cluster
from ..common import emit, fail, resolve_format
from ..previews import preview_cluster_fit

app = typer.Typer(help="Cluster commands for latentdna.")


@app.command("fit")
def fit(
    cluster_id: str = typer.Argument(...),
    workspace: str = typer.Option(..., "--workspace"),
    view: str = typer.Option(..., "--view"),
    n_clusters: int = typer.Option(..., "--n-clusters"),
    sample: str | None = typer.Option(None, "--sample"),
    alignment: str | None = typer.Option(None, "--alignment"),
    seed: int | None = typer.Option(None, "--seed"),
    max_iter: int = typer.Option(100, "--max-iter"),
    force: bool = typer.Option(False, "--force"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    format_name: str = typer.Option("text", "--format"),
    json_output: bool = typer.Option(False, "--json"),
    quiet: bool = typer.Option(False, "--quiet"),
) -> None:
    try:
        payload = (
            preview_cluster_fit(
                workspace,
                cluster_id,
                view_id=view,
                sample_id=sample,
                alignment_id=alignment,
                force=force,
            )
            if dry_run
            else fit_cluster(
                workspace,
                cluster_id,
                view_id=view,
                n_clusters=n_clusters,
                seed=seed,
                max_iter=max_iter,
                sample_id=sample,
                alignment_id=alignment,
                force=force,
            ).model_dump(mode="json")
        )
    except Exception as exc:
        fail(exc)
    emit(payload, format_name=resolve_format(json_output=json_output, format_name=format_name), quiet=quiet)
