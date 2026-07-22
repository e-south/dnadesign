"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/latentdna/src/cli/commands/enrich.py

Enrichment CLI commands for latentdna.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import typer

from ...services.enrichment_service import score_enrichment
from ..common import emit, fail, resolve_format
from ..previews import preview_enrich_score

app = typer.Typer(help="Enrichment commands for latentdna.")


@app.command("score")
def score(
    enrichment_id: str = typer.Argument(...),
    workspace: str = typer.Option(..., "--workspace"),
    neighbors: str = typer.Option(..., "--neighbors"),
    cohort: str = typer.Option(..., "--cohort"),
    landmark: list[str] = typer.Option([], "--landmark"),
    force: bool = typer.Option(False, "--force"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    format_name: str = typer.Option("text", "--format"),
    json_output: bool = typer.Option(False, "--json"),
    quiet: bool = typer.Option(False, "--quiet"),
) -> None:
    try:
        payload = (
            preview_enrich_score(
                workspace,
                enrichment_id,
                neighbors_id=neighbors,
                cohort_id=cohort,
                landmark_ids=landmark,
                force=force,
            )
            if dry_run
            else score_enrichment(
                workspace,
                enrichment_id,
                neighbors_id=neighbors,
                cohort_id=cohort,
                landmark_ids=landmark,
                force=force,
            ).model_dump(mode="json")
        )
    except Exception as exc:
        fail(exc)
    emit(payload, format_name=resolve_format(json_output=json_output, format_name=format_name), quiet=quiet)
