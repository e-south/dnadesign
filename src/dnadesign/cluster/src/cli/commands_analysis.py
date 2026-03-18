"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cluster/src/cli/commands_analysis.py

Analysis-related cluster CLI command registration.

Module Author(s): Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Optional

import typer
from rich.console import Console

from .resolution import resolve_workspace_context, resolve_workspace_value, runs_root_or_exit


def register_analyze_command(app: typer.Typer, *, console: Console) -> None:
    @app.command(
        "analyze",
        help="Run composition/diversity/differential analyses on an existing cluster__<NAME>.",
    )
    def cmd_analyze(
        ctx: typer.Context,
        workspace: Optional[str] = typer.Option(None, help="Workspace directory or packaged workspace id."),
        results_root: Optional[str] = typer.Option(
            None,
            help="Standalone artifact root. Required unless --workspace is set.",
        ),
        dataset: Optional[str] = typer.Option(None),
        file: Optional[str] = typer.Option(None),
        usr_root: Optional[str] = typer.Option(None),
        cluster_col: Optional[str] = typer.Option(None, help="e.g., cluster__perm_v1"),
        group_by: str = typer.Option("source"),
        preset: Optional[str] = typer.Option(None, help="Preset name (kind: 'analysis') to pre-fill parameters"),
        out_dir: Optional[str] = typer.Option(
            None,
            help="If omitted, cluster records analysis under <results-root>/<fit-alias>/analysis/<run-slug>/.",
        ),
        composition: bool = typer.Option(False),
        diversity: bool = typer.Option(False),
        difffeat: bool = typer.Option(False),
        plots: bool = typer.Option(False),
        numeric: Optional[str] = typer.Option(
            None,
            help="Comma-separated numeric columns to summarize/plot per cluster (e.g., infer__...__ll_mean,opal__...__latest_pred_scalar,obj__logic_fidelity)",  # noqa: E501
        ),
        numeric_plots: bool = typer.Option(True, help="Whether to render plots for --numeric"),
        font_scale: Optional[float] = typer.Option(
            None,
            help="Font scale for analysis plots (overrides workspace config or preset).",
        ),
        opal_campaign: Optional[str] = typer.Option(None, help="Optional: OPAL campaign dir or name to join metrics"),
        opal_as_of_round: Optional[int] = typer.Option(None, help="Optional: round filter for OPAL join"),
        opal_fields: Optional[str] = typer.Option(None, help="If set, join these OPAL fields before analysis"),
    ) -> None:
        from ..execution import run_analyze

        workspace_ctx = resolve_workspace_context(workspace, expected_section="analyze")
        wp = workspace_ctx.params
        wp_plot = workspace_ctx.plot
        dataset = resolve_workspace_value(ctx, option_name="dataset", cli_value=dataset, config_params=wp)
        file = resolve_workspace_value(ctx, option_name="file", cli_value=file, config_params=wp)
        usr_root = resolve_workspace_value(ctx, option_name="usr_root", cli_value=usr_root, config_params=wp)
        cluster_col = resolve_workspace_value(
            ctx,
            option_name="cluster_col",
            cli_value=cluster_col,
            config_params=wp,
        )
        if not cluster_col:
            raise typer.BadParameter(
                "Missing --cluster-col.\nProvide --cluster-col cluster__<NAME> or set it in the workspace config."
            )
        group_by = resolve_workspace_value(ctx, option_name="group_by", cli_value=group_by, config_params=wp)
        preset = resolve_workspace_value(ctx, option_name="preset", cli_value=preset, config_params=wp)
        out_dir = resolve_workspace_value(ctx, option_name="out_dir", cli_value=out_dir, config_params=wp)
        composition = bool(
            resolve_workspace_value(
                ctx,
                option_name="composition",
                cli_value=composition,
                config_params=wp,
            )
        )
        diversity = bool(
            resolve_workspace_value(
                ctx,
                option_name="diversity",
                cli_value=diversity,
                config_params=wp,
            )
        )
        difffeat = bool(
            resolve_workspace_value(
                ctx,
                option_name="difffeat",
                cli_value=difffeat,
                config_params=wp,
            )
        )
        plots = bool(resolve_workspace_value(ctx, option_name="plots", cli_value=plots, config_params=wp))
        numeric = resolve_workspace_value(
            ctx,
            option_name="numeric",
            cli_value=numeric,
            config_params=wp,
            config_value=",".join(wp["numeric"]) if isinstance(wp.get("numeric"), (list, tuple)) else wp.get("numeric"),
        )
        numeric_plots = bool(
            resolve_workspace_value(
                ctx,
                option_name="numeric_plots",
                cli_value=numeric_plots,
                config_params=wp,
            )
        )
        font_scale = (
            float(wp.get("font_scale", font_scale))
            if font_scale is not None
            else (float(wp_plot.get("font_scale")) if wp_plot.get("font_scale") is not None else None)
        )
        opal_campaign = resolve_workspace_value(
            ctx,
            option_name="opal_campaign",
            cli_value=opal_campaign,
            config_params=wp,
        )
        opal_as_of_round = resolve_workspace_value(
            ctx,
            option_name="opal_as_of_round",
            cli_value=opal_as_of_round,
            config_params=wp,
        )
        opal_fields = resolve_workspace_value(
            ctx,
            option_name="opal_fields",
            cli_value=opal_fields,
            config_params=wp,
            config_value=",".join(wp["opal_fields"])
            if isinstance(wp.get("opal_fields"), (list, tuple))
            else wp.get("opal_fields"),
        )
        results_store_root = runs_root_or_exit(
            console=console,
            workspace_root=workspace_ctx.results_root,
            results_root=results_root,
        )
        run_analyze(
            dataset=dataset,
            file=file,
            usr_root=usr_root,
            cluster_col=cluster_col,
            group_by=group_by,
            preset=preset,
            out_dir=out_dir,
            composition=composition,
            diversity=diversity,
            difffeat=difffeat,
            plots=plots,
            numeric=numeric,
            numeric_plots=numeric_plots,
            font_scale=font_scale,
            opal_campaign=opal_campaign,
            opal_as_of_round=opal_as_of_round,
            opal_fields=opal_fields,
            root=results_store_root,
            workspace_id=workspace_ctx.workspace_id,
            workspace_plot=wp_plot,
            console=console,
        )


__all__ = ["register_analyze_command"]
