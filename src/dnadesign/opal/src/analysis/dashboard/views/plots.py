"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/analysis/dashboard/views/plots.py

Prepares dashboard plot data for UMAP and cluster visualizations. Provides
pure-data view objects for chart builders.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import polars as pl

from ..hues import HueOption
from ..util import safe_is_numeric


@dataclass(frozen=True)
class ColorSpec:
    field: str
    title: str
    kind: str
    domain: list[str] | None = None
    range: list[str] | None = None
    scheme: str | None = None


@dataclass(frozen=True)
class UmapExplorerView:
    df_plot: pl.DataFrame
    valid: bool
    note: str | None
    x_col: str
    y_col: str
    color_spec: ColorSpec | None
    color_tooltip: str | None


@dataclass(frozen=True)
class UmapChartView:
    df_plot: pl.DataFrame
    x_col: str
    y_col: str
    color_col: str | None
    color_title: str | None
    color_kind: str | None
    tooltip_cols: list[str]


@dataclass(frozen=True)
class ClusterView:
    df_points: pl.DataFrame
    metric_type: str
    hue_label: str
    color_spec: ColorSpec | None
    color_tooltip: str | None
    label_col: str | None
    top_k_mode: bool
    yes_label: str | None
    no_label: str | None
    sort_field: str
    okabe_ito: list[str]


@dataclass(frozen=True)
class UmapOverlayView:
    df_cluster: pl.DataFrame | None
    df_score: pl.DataFrame | None


def _dedupe(cols: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for col in cols:
        if col in seen:
            continue
        seen.add(col)
        out.append(col)
    return out


def add_stable_jitter(
    df: pl.DataFrame,
    *,
    id_col: str,
    jitter_col: str = "__jitter",
    scale: float = 0.6,
) -> pl.DataFrame:
    if id_col not in df.columns:
        return df
    jitter_expr = ((pl.col(id_col).cast(pl.Utf8).hash(seed=0) % 1000).cast(pl.Float64) / 1000.0 - 0.5) * float(scale)
    return df.with_columns(jitter_expr.alias(jitter_col))


def prepare_umap_explorer_view(
    *,
    df: pl.DataFrame,
    x_col: str | None,
    y_col: str | None,
    hue: HueOption | None,
) -> UmapExplorerView:
    _x_col = (x_col or "").strip()
    _y_col = (y_col or "").strip()
    x_name = _x_col or "umap_x"
    y_name = _y_col or "umap_y"
    valid = True
    note: str | None = None

    if "id" not in df.columns:
        note = "UMAP missing: required column `id` is absent."
        valid = False
    elif not _x_col or not _y_col:
        note = (
            "UMAP missing: provide x/y columns. "
            "To attach coords: `uv run cluster umap --dataset <dataset> "
            "--name ldn_v1 --attach-coords --write --allow-overwrite`"
        )
        valid = False
    elif _x_col not in df.columns or _y_col not in df.columns:
        note = (
            "UMAP missing: x/y columns must exist. "
            "To attach coords: `uv run cluster umap --dataset <dataset> "
            "--name ldn_v1 --attach-coords --write --allow-overwrite`"
        )
        valid = False
    elif not (safe_is_numeric(df.schema[_x_col]) and safe_is_numeric(df.schema[_y_col])):
        note = (
            "UMAP missing: x/y columns must be numeric. "
            "To attach coords: `uv run cluster umap --dataset <dataset> "
            "--name ldn_v1 --attach-coords --write --allow-overwrite`"
        )
        valid = False

    if not valid:
        df_chart = pl.DataFrame(
            schema={
                "__row_id": pl.Int64,
                "id": pl.Utf8,
                x_name: pl.Float64,
                y_name: pl.Float64,
            }
        )
        return UmapExplorerView(
            df_plot=df_chart,
            valid=False,
            note=note,
            x_col=x_name,
            y_col=y_name,
            color_spec=None,
            color_tooltip=None,
        )

    note = f"Plotting full dataset: `{df.height}` points."
    okabe_ito = [
        "#E69F00",
        "#56B4E9",
        "#009E73",
        "#F0E442",
        "#0072B2",
        "#D55E00",
        "#CC79A7",
        "#000000",
    ]
    fallback_scheme = "tableau20"

    color_spec: ColorSpec | None = None
    color_tooltip = None
    plot_cols = _dedupe([col for col in ["__row_id", "id", _x_col, _y_col] if col in df.columns])
    if hue is not None and hue.key in df.columns and hue.key not in plot_cols:
        plot_cols.append(hue.key)
    df_chart = df.select(plot_cols)

    if hue is not None and hue.key in df_chart.columns:
        color_field = hue.key
        color_title = hue.label
        color_tooltip = hue.key
        non_null_count = df_chart.select(pl.col(hue.key).count()).item() if df_chart.height else 0
        if non_null_count == 0:
            note = (
                f"Plotting full dataset: `{df.height}` points. "
                f"Color `{hue.key}` has no non-null values; rendering without color."
            )
            color_spec = None
            color_tooltip = None
        elif hue.kind == "categorical" and hue.category_labels:
            label_col = f"{hue.key}__label"
            yes_label, no_label = hue.category_labels
            df_chart = df_chart.with_columns(
                pl.when(pl.col(hue.key)).then(pl.lit(yes_label)).otherwise(pl.lit(no_label)).alias(label_col)
            )
            color_tooltip = label_col
            if "top_k" in hue.key:
                color_spec = ColorSpec(
                    field=label_col,
                    title=hue.label,
                    kind="categorical",
                    domain=[yes_label, no_label],
                    range=[okabe_ito[5], "#B0B0B0"],
                )
            else:
                color_spec = ColorSpec(
                    field=label_col,
                    title=hue.label,
                    kind="categorical",
                    domain=[yes_label, no_label],
                    range=[okabe_ito[2], "#B0B0B0"],
                )
        elif hue.kind == "numeric":
            color_spec = ColorSpec(
                field=color_field,
                title=color_title,
                kind="numeric",
            )
        else:
            n_unique = df_chart.select(pl.col(hue.key).n_unique()).item() if df_chart.height else 0
            if n_unique <= len(okabe_ito):
                color_spec = ColorSpec(
                    field=color_field,
                    title=color_title,
                    kind="categorical",
                    range=okabe_ito,
                )
            else:
                color_spec = ColorSpec(
                    field=color_field,
                    title=color_title,
                    kind="categorical",
                    scheme=fallback_scheme,
                )

    return UmapExplorerView(
        df_plot=df_chart,
        valid=True,
        note=note,
        x_col=_x_col,
        y_col=_y_col,
        color_spec=color_spec,
        color_tooltip=color_tooltip,
    )


def prepare_umap_chart_view(
    *,
    df: pl.DataFrame,
    x_col: str,
    y_col: str,
    color_col: str | None,
    color_title: str | None,
    tooltip_cols: Iterable[str] | None = None,
) -> UmapChartView | None:
    if df.is_empty() or x_col not in df.columns or y_col not in df.columns:
        return None
    base_cols = _dedupe([x_col, y_col, *(tooltip_cols or [])])
    df_plot = df.select([c for c in base_cols if c in df.columns]).filter(
        pl.col(x_col).is_not_null() & pl.col(y_col).is_not_null()
    )
    if df_plot.is_empty():
        return None
    color_kind = None
    if color_col and color_col in df_plot.columns:
        color_kind = "numeric" if safe_is_numeric(df_plot.schema.get(color_col, pl.Null)) else "categorical"
    return UmapChartView(
        df_plot=df_plot,
        x_col=x_col,
        y_col=y_col,
        color_col=color_col if color_col in df_plot.columns else None,
        color_title=color_title,
        color_kind=color_kind,
        tooltip_cols=[c for c in tooltip_cols or [] if c in df_plot.columns],
    )


def prepare_cluster_view(
    *,
    df: pl.DataFrame,
    cluster_col: str,
    metric_col: str,
    hue: HueOption | None,
    id_col: str,
) -> ClusterView | None:
    if df.is_empty() or cluster_col not in df.columns or metric_col not in df.columns:
        return None
    hue_key = hue.key if hue is not None else ""
    cols = _dedupe([cluster_col, metric_col, id_col, hue_key])
    df_points = df.select([c for c in cols if c in df.columns]).filter(pl.col(metric_col).is_not_null())
    if df_points.is_empty():
        return None

    df_points = df_points.with_columns(
        pl.col(cluster_col).cast(pl.Int64, strict=False).alias(f"{cluster_col}__ord"),
    ).with_columns(
        pl.when(pl.col(f"{cluster_col}__ord").is_null())
        .then(pl.lit(1_000_000_000))
        .otherwise(pl.col(f"{cluster_col}__ord"))
        .alias(f"{cluster_col}__ord")
    )
    df_points = add_stable_jitter(df_points, id_col=id_col)
    sort_field = f"{cluster_col}__ord"

    okabe_ito = [
        "#E69F00",
        "#56B4E9",
        "#009E73",
        "#F0E442",
        "#0072B2",
        "#D55E00",
        "#CC79A7",
        "#000000",
    ]

    hue_label = hue.label if hue is not None else "(none)"
    color_spec: ColorSpec | None = None
    color_tooltip = None
    label_col = None
    top_k_mode = False
    yes_label = None
    no_label = None

    if hue is not None and hue.key in df_points.columns:
        if hue.kind == "categorical" and hue.category_labels:
            label_col = f"{hue.key}__label"
            yes_label, no_label = hue.category_labels
            top_k_mode = "top_k" in hue.key
            df_points = df_points.with_columns(
                pl.when(pl.col(hue.key)).then(pl.lit(yes_label)).otherwise(pl.lit(no_label)).alias(label_col)
            )
            color_tooltip = label_col
            color_spec = ColorSpec(
                field=label_col,
                title=hue_label,
                kind="categorical",
                domain=[yes_label, no_label],
                range=["#D62728", "#B0B0B0"] if top_k_mode else [okabe_ito[2], "#B0B0B0"],
            )
        elif hue.kind == "numeric":
            color_tooltip = hue.key
            color_spec = ColorSpec(
                field=hue.key,
                title=hue_label,
                kind="numeric",
            )
        else:
            color_tooltip = hue.key
            color_spec = ColorSpec(
                field=hue.key,
                title=hue_label,
                kind="categorical",
                range=okabe_ito,
            )

    metric_type = "Q" if safe_is_numeric(df_points.schema.get(metric_col, pl.Null)) else "N"

    return ClusterView(
        df_points=df_points,
        metric_type=metric_type,
        hue_label=hue_label,
        color_spec=color_spec,
        color_tooltip=color_tooltip,
        label_col=label_col,
        top_k_mode=top_k_mode,
        yes_label=yes_label,
        no_label=no_label,
        sort_field=sort_field,
        okabe_ito=okabe_ito,
    )


def prepare_umap_overlay_view(
    *,
    df: pl.DataFrame,
    id_col: str,
    umap_x_col: str,
    umap_y_col: str,
    cluster_col: str,
) -> UmapOverlayView:
    if df.is_empty() or umap_x_col not in df.columns or umap_y_col not in df.columns:
        return UmapOverlayView(df_cluster=None, df_score=None)

    base_cols = [umap_x_col, umap_y_col]
    if id_col in df.columns:
        base_cols.append(id_col)

    df_cluster = None
    if cluster_col in df.columns:
        df_cluster = df.select(_dedupe(base_cols + [cluster_col])).filter(pl.col(cluster_col).is_not_null())
        if df_cluster.is_empty():
            df_cluster = None

    score_col = "opal__view__score" if "opal__view__score" in df.columns else None
    df_score = None
    if score_col is not None:
        df_score = df.select(_dedupe(base_cols + [score_col])).filter(pl.col(score_col).is_not_null())
        if df_score.is_empty():
            df_score = None

    return UmapOverlayView(df_cluster=df_cluster, df_score=df_score)
