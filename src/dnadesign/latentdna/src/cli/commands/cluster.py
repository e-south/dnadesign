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
    method: str = typer.Option("kmeans", "--method"),
    n_clusters: int | None = typer.Option(None, "--n-clusters"),
    sample: str | None = typer.Option(None, "--sample"),
    alignment: str | None = typer.Option(None, "--alignment"),
    neighbor_set: str | None = typer.Option(None, "--neighbor-set"),
    metric: str | None = typer.Option(None, "--metric"),
    k: int = typer.Option(30, "--k"),
    resolution: float = typer.Option(1.0, "--resolution"),
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
                method=method,
                force=force,
            )
            if dry_run
            else fit_cluster(
                workspace,
                cluster_id,
                view_id=view,
                method=method,
                n_clusters=n_clusters,
                seed=seed,
                max_iter=max_iter,
                sample_id=sample,
                alignment_id=alignment,
                neighbor_set_id=neighbor_set,
                metric=metric,
                k=k,
                resolution=resolution,
                force=force,
            ).model_dump(mode="json")
        )
    except Exception as exc:
        fail(exc)
    emit(payload, format_name=resolve_format(json_output=json_output, format_name=format_name), quiet=quiet)
