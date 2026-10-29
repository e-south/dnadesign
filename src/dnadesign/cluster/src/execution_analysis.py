"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cluster/src/execution_analysis.py

Analysis execution runtime for cluster.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from rich.console import Console

from .execution_analysis_support import (
    join_required_opal_fields,
    prepare_analysis_plan,
    run_grouped_analyses,
    run_numeric_analysis,
)
from .execution_support import CommandExecution, _log, _rule, append_command_record_or_warn, context_and_df
from .runs.recorder import CommandRecord, record_analysis_run


def run_analyze(
    *,
    dataset: str | None,
    file: str | None,
    usr_root: str | None,
    cluster_col: str,
    group_by: str,
    preset: str | None,
    out_dir: str | None,
    composition: bool,
    diversity: bool,
    difffeat: bool,
    plots: bool,
    numeric: str | None,
    numeric_plots: bool,
    font_scale: float | None,
    opal_campaign: str | None,
    opal_as_of_round: int | None,
    opal_fields: str | None,
    root: Path,
    workspace_id: str | None = None,
    workspace_plot: dict[str, Any] | None = None,
    console: Console | None = None,
) -> CommandExecution:
    ictx, df = context_and_df(dataset, file, usr_root)
    _rule(console, "[bold]cluster analyze[/]")

    plan = prepare_analysis_plan(
        ictx=ictx,
        df=df,
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
        root=root,
        workspace_plot=workspace_plot,
    )
    request = plan.request
    fit_alias = request.fit_alias

    df = join_required_opal_fields(df, request=request, console=console)
    run_numeric_analysis(df, request=request, out_root=plan.out_root, console=console)
    run_grouped_analyses(df, request=request, out_root=plan.out_root, console=console)

    _log(console, "print", f"[green]Analyses complete[/green]. Outputs at {plan.out_root}")
    analysis_run = replace(
        request.to_run(alias=plan.alias, slug=plan.slug, created_utc=plan.created_utc),
        out_dir=plan.out_root,
    )
    record_analysis_run(root=root, out_dir=plan.out_root, run=analysis_run)
    if fit_alias is not None:
        append_command_record_or_warn(
            root / fit_alias,
            CommandRecord(
                command="analyze",
                subject=fit_alias,
                workspace=workspace_id,
                preset=preset or None,
                resolved={**request.command_payload(), "out_dir": str(plan.out_root)},
            ),
            console=console,
        )
    return CommandExecution(
        command="analyze",
        subject=fit_alias or cluster_col,
        artifact_path=plan.out_root,
        run_record_subject=fit_alias or cluster_col,
    )


__all__ = ["run_analyze"]
