"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cluster/src/execution_analysis_support.py

Shared request planning and dispatch helpers for cluster analysis execution.

Module Author(s): Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import typer
from rich.console import Console

from .analysis.contracts import AnalysisRequest
from .execution_support import _log, resolve_scoped_out_dir
from .opal.join import join_fields as opal_join_fields
from .opal.join import resolve_campaign_dir as resolve_opal_campaign_dir
from .presets.runtime import apply_preset
from .runs.contracts import utc_now_iso
from .runs.store import analysis_run_dir
from .runtime_contracts import InputSource
from .util.slug import artifact_slug, slugify


@dataclass(frozen=True, slots=True)
class AnalysisExecutionPlan:
    request: AnalysisRequest
    out_root: Path
    alias: str
    slug: str
    created_utc: str


def prepare_analysis_plan(
    *,
    ictx: dict[str, Any],
    df: pd.DataFrame,
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
    workspace_plot: dict[str, Any] | None,
) -> AnalysisExecutionPlan:
    workspace_plot = workspace_plot or {}
    preset_params = apply_preset("analysis", preset)
    if preset_params:
        group_by_from_preset = preset_params.get("group_by")
        if group_by_from_preset:
            group_bys = [group_by_from_preset] if isinstance(group_by_from_preset, str) else list(group_by_from_preset)
        else:
            group_bys = [group_by] if group_by else ["source"]
        composition = composition or bool(preset_params.get("composition", False))
        diversity = diversity or bool(preset_params.get("diversity", False))
        difffeat = difffeat or bool(preset_params.get("difffeat", False))
        plots = plots or bool(preset_params.get("plots", False))
        if not numeric and preset_params.get("numeric"):
            numeric = (
                ",".join(preset_params["numeric"])
                if isinstance(preset_params["numeric"], (list, tuple))
                else str(preset_params["numeric"])
            )
        if font_scale is None:
            raw_font_scale = preset_params.get("font_scale")
            font_scale = float(raw_font_scale) if raw_font_scale is not None else None
        if font_scale is None and workspace_plot.get("font_scale") is not None:
            font_scale = float(workspace_plot["font_scale"])
        if font_scale is None:
            font_scale = 1.2
        numeric_missing_policy = str(preset_params.get("missing_policy", "error"))
        opal_campaign = opal_campaign or preset_params.get("opal_campaign")
        opal_as_of_round = opal_as_of_round or preset_params.get("opal_as_of_round")
        if not opal_fields and preset_params.get("opal_fields"):
            opal_fields = (
                ",".join(preset_params["opal_fields"])
                if isinstance(preset_params["opal_fields"], (list, tuple))
                else str(preset_params["opal_fields"])
            )
    else:
        group_bys = group_by
        numeric_missing_policy = "error"
        if font_scale is None:
            font_scale = float(workspace_plot.get("font_scale", 1.2))

    try:
        request = AnalysisRequest.from_runtime(
            source=InputSource.from_context(ictx),
            df_columns=list(df.columns),
            cluster_col=cluster_col,
            group_by=group_bys,
            out_dir=out_dir,
            results_root=root,
            composition=composition,
            diversity=diversity,
            difffeat=difffeat,
            plots=plots,
            numeric=numeric,
            numeric_missing_policy=numeric_missing_policy,
            numeric_plots=numeric_plots,
            font_scale=float(font_scale),
            opal_campaign=opal_campaign,
            opal_as_of_round=opal_as_of_round,
            opal_fields=opal_fields,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    alias = request.fit_alias or slugify(request.cluster_col)
    created_utc = utc_now_iso()
    if out_dir is None:
        slug = artifact_slug(
            alias,
            created_utc=created_utc,
            fingerprint=json.dumps(request.command_payload(), sort_keys=True),
        )
        out_root = analysis_run_dir(root, alias, slug)
    else:
        slug = slugify(Path(request.out_dir).name)
        out_root = resolve_scoped_out_dir(requested=str(request.out_dir), root=root)
    out_root.mkdir(parents=True, exist_ok=True)
    return AnalysisExecutionPlan(
        request=request,
        out_root=out_root,
        alias=alias,
        slug=slug,
        created_utc=created_utc,
    )


def join_required_opal_fields(
    df: pd.DataFrame,
    *,
    request: AnalysisRequest,
    console: Console | None,
) -> pd.DataFrame:
    required_fields = set(request.required_opal_fields)
    if not required_fields:
        return df
    if not request.opal_campaign:
        raise typer.BadParameter(
            "Analysis requires OPAL metrics "
            f"({', '.join(sorted(required_fields))}) but --opal-campaign is not set. "
            "Pass --opal-campaign <name|path> and optionally --opal-as-of-round <n>."
        )
    try:
        campaign_dir = resolve_opal_campaign_dir(request.opal_campaign)
    except FileNotFoundError as exc:
        raise typer.BadParameter(str(exc)) from exc
    joined = opal_join_fields(
        df,
        campaign_dir=campaign_dir,
        run_selector="latest",
        fields=sorted(required_fields),
        as_of_round=request.opal_as_of_round,
        log_fn=lambda message: _log(console, "log", message),
    )
    for column in required_fields:
        if column not in joined.columns:
            raise typer.BadParameter(f"Joined OPAL field '{column}' missing after join.")
        miss = float(joined[column].isna().mean())
        if miss > 0.0:
            _log(console, "print", f"[yellow]Warning[/yellow]: joined '{column}' has {miss:.1%} missing values.")
    _log(console, "log", "Joined OPAL fields: " + ", ".join(sorted(required_fields)))
    return joined


def run_numeric_analysis(
    df: pd.DataFrame,
    *,
    request: AnalysisRequest,
    out_root: Path,
    console: Console | None,
) -> None:
    if not request.numeric_cols:
        return
    from .analysis.numeric_per_cluster import summarize_numeric_by_cluster

    summarize_numeric_by_cluster(
        df,
        cluster_col=request.cluster_col,
        numeric_cols=list(request.numeric_cols),
        out_dir=out_root,
        plots=request.numeric_plots,
        font_scale=request.font_scale,
        missing_policy=request.numeric_missing_policy,
        log_fn=lambda message: _log(console, "print", f"[yellow]Note[/yellow]: {message}"),
    )
    _log(console, "log", "Numeric summaries/plots written.")


def run_grouped_analyses(
    df: pd.DataFrame,
    *,
    request: AnalysisRequest,
    out_root: Path,
    console: Console | None,
) -> None:
    for group_by in request.group_by:
        if request.composition:
            from .analysis.composition import composition as composition_fn

            composition_fn(
                df, cluster_col=request.cluster_col, group_by=group_by, out_dir=out_root, plots=request.plots
            )
        if request.diversity:
            from .analysis.diversity import diversity as diversity_fn

            diversity_fn(df, cluster_col=request.cluster_col, group_by=group_by, out_dir=out_root, plots=request.plots)
        if request.difffeat:
            from .analysis.differential import differential as differential_fn

            differential_fn(df, cluster_col=request.cluster_col, group_by=group_by, out_dir=out_root)
        _log(console, "log", f"Completed group_by='{group_by}'.")


__all__ = [
    "AnalysisExecutionPlan",
    "join_required_opal_fields",
    "prepare_analysis_plan",
    "run_grouped_analyses",
    "run_numeric_analysis",
]
