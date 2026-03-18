"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cluster/src/umap/requests.py

UMAP request normalization helpers.

Module Author(s): Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import typer
from rich.console import Console

from ..execution_support import _log, load_highlight_ids_from_file, resolve_color_by
from ..presets.runtime import apply_plot_preset, apply_preset
from .contracts import DEFAULT_PLOT_CONFIG, ResolvedUmapRequest


def _deep_merge(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    out = dict(left)
    for key, value in (right or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _resolve_highlight_payload(
    *,
    df: pd.DataFrame,
    key_col: str,
    highlight: str | None,
    highlight_topn: int | None,
    highlight_topn_col: str | None,
    highlight_topn_asc: bool,
    highlight_hue_col: str | None,
    console: Console | None,
) -> dict[str, Any] | None:
    if highlight and highlight_topn:
        raise typer.BadParameter("Use either --highlight (file) OR --highlight-topn, not both.")
    if highlight:
        return load_highlight_ids_from_file(
            highlight,
            df,
            key_col,
            warn_fn=lambda message: _log(console, "print", f"[yellow]Warning:[/yellow] {message}"),
            groupby_col=highlight_hue_col,
        )
    if highlight_topn is None:
        return None
    if not highlight_topn_col:
        raise typer.BadParameter("--highlight-topn requires --highlight-topn-col.")
    if highlight_hue_col:
        _log(
            console,
            "print",
            "[yellow]Note[/yellow]: --highlight-hue-col is ignored when using --highlight-topn.",
        )
    if highlight_topn <= 0:
        raise typer.BadParameter("--highlight-topn must be a positive integer.")
    if highlight_topn_col not in df.columns:
        raise typer.BadParameter(f"--highlight-topn-col '{highlight_topn_col}' not found in the table.")
    try:
        numeric_series = pd.to_numeric(df[highlight_topn_col], errors="raise")
    except Exception as exc:
        raise typer.BadParameter(f"--highlight-topn-col '{highlight_topn_col}' is not numeric: {exc}") from exc
    values = numeric_series.to_numpy(dtype="float64", copy=False)
    nonfinite = ~np.isfinite(values)
    if nonfinite.any():
        _log(
            console,
            "print",
            f"[yellow]Note[/yellow]: excluding {int(nonfinite.sum())} non-finite row(s) "
            f"from Top-N selection of '{highlight_topn_col}'.",
        )
        numeric_series = numeric_series[~nonfinite]
    ordered = numeric_series.sort_values(ascending=bool(highlight_topn_asc))
    take = int(min(len(ordered), int(highlight_topn)))
    chosen_idx = ordered.iloc[:take].index
    ids = (
        pd.Index(chosen_idx).astype(str).tolist()
        if df.index.name == key_col
        else df.loc[chosen_idx, key_col].astype(str).tolist()
    )
    return {"ids": ids}


def _resolve_dims(dims_value: str | list[Any] | tuple[Any, ...]) -> tuple[int, int]:
    if isinstance(dims_value, str):
        width, height = [int(x) for x in dims_value.split(",")]
        return width, height
    return int(dims_value[0]), int(dims_value[1])


def _resolve_legend(legend_cfg: dict[str, Any]) -> dict[str, Any]:
    legend: dict[str, Any] = {}
    if "ncol" in legend_cfg:
        legend["ncol"] = int(legend_cfg["ncol"])
    if "bbox" in legend_cfg:
        bbox = legend_cfg["bbox"]
        legend["bbox"] = tuple(bbox[:2]) if isinstance(bbox, (list, tuple)) else (1.05, 1.0)
    else:
        legend["bbox"] = (1.05, 1.0)
    if "max_items" in legend_cfg:
        legend["max_items"] = int(legend_cfg["max_items"])
    if "frameon" in legend_cfg:
        legend["frameon"] = bool(legend_cfg["frameon"])
    return legend


def _resolve_highlight_style(preset_plot: dict[str, Any], workspace_plot: dict[str, Any]) -> dict[str, Any]:
    highlight_style = _deep_merge(DEFAULT_PLOT_CONFIG.get("highlight", {}), preset_plot.get("highlight", {}))
    highlight_style = _deep_merge(highlight_style, workspace_plot.get("highlight", {}))
    if "palette" in (preset_plot.get("highlight", {}) or {}):
        highlight_style["palette"] = preset_plot["highlight"]["palette"]
    if "highlight" in workspace_plot and "palette" in (workspace_plot["highlight"] or {}):
        highlight_style["palette"] = workspace_plot["highlight"]["palette"]
    return highlight_style


def resolve_umap_request(
    *,
    df: pd.DataFrame,
    key_col: str,
    preset: str | None,
    neighbors: int | None,
    min_dist: float | None,
    metric: str | None,
    random_state: int | None,
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
    render_plots: bool | None,
    workspace_params: dict[str, Any],
    workspace_plot: dict[str, Any],
    console: Console | None,
) -> ResolvedUmapRequest:
    preset_umap = apply_preset("umap", preset)
    resolved_neighbors = neighbors if neighbors is not None else int(preset_umap.get("neighbors", 15))
    resolved_min_dist = min_dist if min_dist is not None else float(preset_umap.get("min_dist", 0.10))
    resolved_metric = metric if metric is not None else str(preset_umap.get("metric", "euclidean"))
    resolved_random_state = random_state if random_state is not None else int(preset_umap.get("random_state", 42))

    preset_plot = apply_plot_preset(preset)
    resolved_color_by = resolve_color_by(color_by, workspace_params, workspace_plot, preset_plot)
    wants_highlight = bool(highlight or highlight_topn)
    if wants_highlight and "highlight" not in resolved_color_by:
        resolved_color_by = [*resolved_color_by, "highlight"]
    if "highlight" in resolved_color_by and not wants_highlight:
        raise typer.BadParameter(
            "hue 'highlight' was requested (via preset/workspace), "
            "but neither --highlight nor --highlight-topn was provided."
        )

    merged_plot = {**DEFAULT_PLOT_CONFIG, **preset_plot, **workspace_plot}
    if alpha is not None:
        merged_plot["alpha"] = float(alpha)
    if size is not None:
        merged_plot["size"] = float(size)
    if dims is not None:
        merged_plot["dims"] = dims
    if font_scale is not None:
        merged_plot["font_scale"] = float(font_scale)
    if render_plots is not None:
        merged_plot["enabled"] = bool(render_plots)

    highlight_payload = _resolve_highlight_payload(
        df=df,
        key_col=key_col,
        highlight=highlight,
        highlight_topn=highlight_topn,
        highlight_topn_col=highlight_topn_col,
        highlight_topn_asc=highlight_topn_asc,
        highlight_hue_col=highlight_hue_col,
        console=console,
    )
    return ResolvedUmapRequest(
        neighbors=resolved_neighbors,
        min_dist=resolved_min_dist,
        metric=resolved_metric,
        random_state=resolved_random_state,
        render_plots=bool(merged_plot.get("enabled", True)),
        color_by=tuple(resolved_color_by),
        highlight_payload=highlight_payload,
        alpha=float(merged_plot["alpha"]),
        size=float(merged_plot["size"]),
        dims=_resolve_dims(merged_plot.get("dims", [12, 12])),
        font_scale=float(merged_plot.get("font_scale", 1.2)),
        legend=_resolve_legend(dict(merged_plot.get("legend", {}))),
        highlight_style=_resolve_highlight_style(preset_plot, workspace_plot),
    )
