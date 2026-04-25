"""
Agreement CLI commands for latentdna.
"""

from __future__ import annotations

import typer

from ...services.agreement_service import compare_agreement
from ..common import emit, fail, resolve_format
from ..previews import preview_agreement_compare

app = typer.Typer(help="Agreement commands for latentdna.")


@app.command("compare")
def compare(
    agreement_id: str = typer.Argument(...),
    workspace: str = typer.Option(..., "--workspace"),
    left_neighbors: str | None = typer.Option(None, "--left-neighbors"),
    right_neighbors: str | None = typer.Option(None, "--right-neighbors"),
    left_clusters: str | None = typer.Option(None, "--left-clusters"),
    right_clusters: str | None = typer.Option(None, "--right-clusters"),
    landmark: list[str] = typer.Option([], "--landmark"),
    force: bool = typer.Option(False, "--force"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    format_name: str = typer.Option("text", "--format"),
    json_output: bool = typer.Option(False, "--json"),
    quiet: bool = typer.Option(False, "--quiet"),
) -> None:
    try:
        payload = (
            preview_agreement_compare(
                workspace,
                agreement_id,
                left_neighbors_id=left_neighbors,
                right_neighbors_id=right_neighbors,
                left_cluster_id=left_clusters,
                right_cluster_id=right_clusters,
                landmark_ids=landmark,
                force=force,
            )
            if dry_run
            else compare_agreement(
                workspace,
                agreement_id,
                left_neighbors_id=left_neighbors,
                right_neighbors_id=right_neighbors,
                left_cluster_id=left_clusters,
                right_cluster_id=right_clusters,
                landmark_ids=landmark,
                force=force,
            ).model_dump(mode="json")
        )
    except Exception as exc:
        fail(exc)
    emit(payload, format_name=resolve_format(json_output=json_output, format_name=format_name), quiet=quiet)
