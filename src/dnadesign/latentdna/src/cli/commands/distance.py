"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/latentdna/src/cli/commands/distance.py

Distance CLI commands for latentdna.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import typer

from ...services.distance_service import score_distance
from ..common import emit, fail, resolve_format
from ..previews import preview_distance_score

app = typer.Typer(help="Distance commands for latentdna.")


@app.command("score")
def score(
    distance_id: str = typer.Argument(...),
    workspace: str = typer.Option(..., "--workspace"),
    view: str = typer.Option(..., "--view"),
    landmark: list[str] = typer.Option([], "--landmark"),
    alignment: str | None = typer.Option(None, "--alignment"),
    metric: str | None = typer.Option(None, "--metric"),
    force: bool = typer.Option(False, "--force"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    format_name: str = typer.Option("text", "--format"),
    json_output: bool = typer.Option(False, "--json"),
    quiet: bool = typer.Option(False, "--quiet"),
) -> None:
    try:
        payload = (
            preview_distance_score(
                workspace,
                distance_id,
                view_id=view,
                landmark_ids=landmark,
                alignment_id=alignment,
                force=force,
            )
            if dry_run
            else score_distance(
                workspace,
                distance_id,
                view_id=view,
                landmark_ids=landmark,
                metric=metric,
                alignment_id=alignment,
                force=force,
            ).model_dump(mode="json")
        )
    except Exception as exc:
        fail(exc)
    emit(payload, format_name=resolve_format(json_output=json_output, format_name=format_name), quiet=quiet)
