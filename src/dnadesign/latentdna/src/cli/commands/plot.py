"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/latentdna/src/cli/commands/plot.py

Plot CLI commands for latentdna.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
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
    panel_title: list[str] = typer.Option([], "--panel-title"),
    enrichment: str | None = typer.Option(None, "--enrichment"),
    distance: str | None = typer.Option(None, "--distance"),
    scalar: str | None = typer.Option(None, "--scalar"),
    scalar_panel: list[str] = typer.Option([], "--scalar-panel"),
    agreement: str | None = typer.Option(None, "--agreement"),
    agreement_panel: list[str] = typer.Option([], "--agreement-panel"),
    reducer: str | None = typer.Option(None, "--reducer"),
    left_cluster: str | None = typer.Option(None, "--left-cluster"),
    right_cluster: str | None = typer.Option(None, "--right-cluster"),
    value_column: str | None = typer.Option(None, "--value-column"),
    x_column: str | None = typer.Option(None, "--x-column"),
    y_column: str | None = typer.Option(None, "--y-column"),
    color_column: str | None = typer.Option(None, "--color-column"),
    shape_column: str | None = typer.Option(None, "--shape-column"),
    render_mode: str | None = typer.Option(None, "--render-mode"),
    label_column: str | None = typer.Option(None, "--label-column"),
    label_value: list[str] = typer.Option([], "--label-value"),
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
                panel_titles=panel_title,
                enrichment_id=enrichment,
                distance_id=distance,
                scalar_id=scalar,
                scalar_ids=scalar_panel,
                agreement_id=agreement,
                agreement_ids=agreement_panel,
                reducer_id=reducer,
                left_cluster_id=left_cluster,
                right_cluster_id=right_cluster,
                value_column=value_column,
                x_column=x_column,
                y_column=y_column,
                color_column=color_column,
                shape_column=shape_column,
                render_mode=render_mode,
                label_column=label_column,
                label_values=label_value,
                force=force,
            )
            if dry_run
            else render_plot(
                workspace,
                plot_id,
                kind=kind,
                projection_ids=projection,
                panel_titles=panel_title,
                enrichment_id=enrichment,
                distance_id=distance,
                scalar_id=scalar,
                scalar_ids=scalar_panel,
                agreement_id=agreement,
                agreement_ids=agreement_panel,
                reducer_id=reducer,
                left_cluster_id=left_cluster,
                right_cluster_id=right_cluster,
                value_column=value_column,
                x_column=x_column,
                y_column=y_column,
                color_column=color_column,
                shape_column=shape_column,
                render_mode=render_mode,
                label_column=label_column,
                label_values=label_value,
                force=force,
            ).model_dump(mode="json")
        )
    except Exception as exc:
        fail(exc)
    emit(payload, format_name=resolve_format(json_output=json_output, format_name=format_name), quiet=quiet)
