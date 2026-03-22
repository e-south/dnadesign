"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cluster/src/cli/umap_resolution.py

UMAP CLI resolution helpers for cluster.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import typer
from rich.console import Console

from .resolution import (
    assert_no_method_overlap_with_preset,
    resolve_workspace_context,
    resolve_workspace_value,
    runs_root_or_exit,
)


@dataclass(frozen=True, slots=True)
class ResolvedUmapCommand:
    run_kwargs: dict[str, Any]


def resolve_umap_command(
    *,
    ctx: typer.Context,
    console: Console,
    workspace: str | None,
    results_root: str | None,
    dataset: str | None,
    file: str | None,
    usr_root: str | None,
    name: str | None,
    key_col: str,
    x_col: str | None,
    x_cols: str | None,
    neighbors: int | None,
    min_dist: float | None,
    metric: str | None,
    random_state: int | None,
    preset: str | None,
    color_by: list[str],
    highlight: str | None,
    highlight_topn: int | None,
    highlight_topn_col: str | None,
    highlight_topn_asc: bool,
    highlight_hue_col: str | None,
    alpha: float | None,
    size: float | None,
    dims: str | None,
    font_scale: float | None,
    plots: bool,
    opal_campaign: str | None,
    opal_run: str | None,
    opal_as_of_round: int | None,
    opal_fields: str | None,
    derive_ratio: list[str],
    attach_coords: bool,
    write: bool,
    allow_overwrite: bool,
    inplace: bool,
    out: str | None,
) -> ResolvedUmapCommand:
    workspace_ctx = resolve_workspace_context(workspace, expected_section="umap")
    wp = workspace_ctx.params
    wp_plot = workspace_ctx.plot

    dataset = resolve_workspace_value(ctx, option_name="dataset", cli_value=dataset, config_params=wp)
    file = resolve_workspace_value(ctx, option_name="file", cli_value=file, config_params=wp)
    usr_root = resolve_workspace_value(ctx, option_name="usr_root", cli_value=usr_root, config_params=wp)
    name = resolve_workspace_value(ctx, option_name="name", cli_value=name, config_params=wp)
    if not name:
        raise typer.BadParameter("UMAP requires a fit alias. Provide --name or set it in the workspace config.")
    key_col = resolve_workspace_value(ctx, option_name="key_col", cli_value=key_col, config_params=wp)
    x_col = resolve_workspace_value(ctx, option_name="x_col", cli_value=x_col, config_params=wp)
    if x_col:
        x_col = str(x_col).strip()
    x_cols = resolve_workspace_value(ctx, option_name="x_cols", cli_value=x_cols, config_params=wp)
    neighbors = resolve_workspace_value(ctx, option_name="neighbors", cli_value=neighbors, config_params=wp)
    min_dist = resolve_workspace_value(ctx, option_name="min_dist", cli_value=min_dist, config_params=wp)
    metric = resolve_workspace_value(ctx, option_name="metric", cli_value=metric, config_params=wp)
    random_state = resolve_workspace_value(ctx, option_name="random_state", cli_value=random_state, config_params=wp)
    preset = resolve_workspace_value(ctx, option_name="preset", cli_value=preset, config_params=wp)
    assert_no_method_overlap_with_preset("umap", wp, preset)
    if color_by == ["cluster"] and isinstance(wp.get("color_by"), (list, tuple)):
        color_by = list(wp["color_by"])
    highlight = resolve_workspace_value(ctx, option_name="highlight", cli_value=highlight, config_params=wp)
    highlight_hue_col = resolve_workspace_value(
        ctx,
        option_name="highlight_hue_col",
        cli_value=highlight_hue_col,
        config_params=wp,
    )
    highlight_topn = resolve_workspace_value(
        ctx,
        option_name="highlight_topn",
        cli_value=highlight_topn,
        config_params=wp,
    )
    highlight_topn_col = resolve_workspace_value(
        ctx,
        option_name="highlight_topn_col",
        cli_value=highlight_topn_col,
        config_params=wp,
    )
    highlight_topn_asc = bool(
        resolve_workspace_value(
            ctx,
            option_name="highlight_topn_asc",
            cli_value=highlight_topn_asc,
            config_params=wp,
        )
    )
    alpha = resolve_workspace_value(ctx, option_name="alpha", cli_value=alpha, config_params=wp)
    size = resolve_workspace_value(ctx, option_name="size", cli_value=size, config_params=wp)
    dims = resolve_workspace_value(ctx, option_name="dims", cli_value=dims, config_params=wp)
    font_scale = resolve_workspace_value(
        ctx,
        option_name="font_scale",
        cli_value=font_scale,
        config_params=wp_plot,
        config_value=wp.get("font_scale", wp_plot.get("font_scale")),
    )
    plots = bool(
        resolve_workspace_value(
            ctx,
            option_name="plots",
            cli_value=plots,
            config_params=wp_plot,
            config_key="enabled",
        )
    )
    opal_campaign = resolve_workspace_value(ctx, option_name="opal_campaign", cli_value=opal_campaign, config_params=wp)
    opal_run = resolve_workspace_value(ctx, option_name="opal_run", cli_value=opal_run, config_params=wp)
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
    attach_coords = bool(
        resolve_workspace_value(
            ctx,
            option_name="attach_coords",
            cli_value=attach_coords,
            config_params=wp,
        )
    )
    write = bool(resolve_workspace_value(ctx, option_name="write", cli_value=write, config_params=wp))
    allow_overwrite = bool(
        resolve_workspace_value(
            ctx,
            option_name="yes",
            cli_value=allow_overwrite,
            config_params=wp,
            config_key="allow_overwrite",
        )
    )
    inplace = bool(resolve_workspace_value(ctx, option_name="inplace", cli_value=inplace, config_params=wp))
    out = resolve_workspace_value(ctx, option_name="out", cli_value=out, config_params=wp)
    root = runs_root_or_exit(
        console=console,
        workspace_root=workspace_ctx.results_root,
        results_root=results_root,
    )
    if not derive_ratio and wp.get("derive_ratio"):
        derive_ratio = (
            list(wp["derive_ratio"]) if isinstance(wp["derive_ratio"], (list, tuple)) else [str(wp["derive_ratio"])]
        )

    return ResolvedUmapCommand(
        run_kwargs={
            "dataset": dataset,
            "file": file,
            "usr_root": usr_root,
            "name": name,
            "key_col": key_col,
            "x_col": x_col,
            "x_cols": x_cols,
            "neighbors": neighbors,
            "min_dist": min_dist,
            "metric": metric,
            "random_state": random_state,
            "preset": preset,
            "color_by": list(color_by),
            "highlight": highlight,
            "highlight_topn": highlight_topn,
            "highlight_topn_col": highlight_topn_col,
            "highlight_topn_asc": highlight_topn_asc,
            "highlight_hue_col": highlight_hue_col,
            "alpha": alpha,
            "size": size,
            "dims": dims,
            "font_scale": font_scale,
            "render_plots": plots,
            "opal_campaign": opal_campaign,
            "opal_run": opal_run,
            "opal_as_of_round": opal_as_of_round,
            "opal_fields": opal_fields,
            "derive_ratio": list(derive_ratio),
            "attach_coords": attach_coords,
            "write": write,
            "allow_overwrite": allow_overwrite,
            "inplace": inplace,
            "out": out,
            "root": root,
            "workspace_id": workspace_ctx.workspace_id,
            "workspace_params": wp,
            "workspace_plot": wp_plot,
            "console": console,
        }
    )


__all__ = ["ResolvedUmapCommand", "resolve_umap_command"]
