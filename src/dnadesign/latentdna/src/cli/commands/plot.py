"""
Plot CLI commands for latentdna.
"""

from __future__ import annotations

import typer

from ...services.plot_service import render_plot
from ..common import emit, fail, resolve_format
from ..previews import preview_plot_render

app = typer.Typer(help="Plot commands for latentdna.")


@app.command("render")
def render(
    plot_id: str = typer.Argument(...),
    workspace: str = typer.Option(..., "--workspace"),
    kind: str | None = typer.Option(None, "--kind"),
    projection: list[str] = typer.Option([], "--projection"),
    enrichment: str | None = typer.Option(None, "--enrichment"),
    distance: str | None = typer.Option(None, "--distance"),
    scalar: str | None = typer.Option(None, "--scalar"),
    agreement: str | None = typer.Option(None, "--agreement"),
    value_column: str | None = typer.Option(None, "--value-column"),
    x_column: str | None = typer.Option(None, "--x-column"),
    y_column: str | None = typer.Option(None, "--y-column"),
    color_column: str | None = typer.Option(None, "--color-column"),
    force: bool = typer.Option(False, "--force"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    format_name: str = typer.Option("text", "--format"),
    json_output: bool = typer.Option(False, "--json"),
    quiet: bool = typer.Option(False, "--quiet"),
) -> None:
    try:
        payload = (
            preview_plot_render(
                workspace,
                plot_id,
                kind=kind,
                projection_ids=projection,
                enrichment_id=enrichment,
                distance_id=distance,
                scalar_id=scalar,
                agreement_id=agreement,
                value_column=value_column,
                x_column=x_column,
                y_column=y_column,
                color_column=color_column,
                force=force,
            )
            if dry_run
            else render_plot(
                workspace,
                plot_id,
                kind=kind,
                projection_ids=projection,
                enrichment_id=enrichment,
                distance_id=distance,
                scalar_id=scalar,
                agreement_id=agreement,
                value_column=value_column,
                x_column=x_column,
                y_column=y_column,
                color_column=color_column,
                force=force,
            ).model_dump(mode="json")
        )
    except Exception as exc:
        fail(exc)
    emit(payload, format_name=resolve_format(json_output=json_output, format_name=format_name), quiet=quiet)
