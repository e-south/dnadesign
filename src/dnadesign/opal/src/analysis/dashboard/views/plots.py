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
    if df.is_empty():
        raise ValueError("UMAP missing: dataset is empty.")
    if "id" not in df.columns:
        raise ValueError("UMAP missing: required column `id` is absent.")
    if not _x_col or not _y_col:
        raise ValueError(
            "UMAP missing: provide x/y columns. "
            "To attach coords: `uv run cluster umap --dataset <dataset> "
            "--name ldn_v1 --attach-coords --write --allow-overwrite`"
        )
    if _x_col not in df.columns or _y_col not in df.columns:
        raise ValueError(
            "UMAP missing: x/y columns must exist. "
            "To attach coords: `uv run cluster umap --dataset <dataset> "
            "--name ldn_v1 --attach-coords --write --allow-overwrite`"
        )
    if not (safe_is_numeric(df.schema[_x_col]) and safe_is_numeric(df.schema[_y_col])):
        raise ValueError(
            "UMAP missing: x/y columns must be numeric. "
            "To attach coords: `uv run cluster umap --dataset <dataset> "
            "--name ldn_v1 --attach-coords --write --allow-overwrite`"
        )

    if (
        not df.select(pl.col(_x_col).is_not_null().all()).item()
        or not df.select(pl.col(_y_col).is_not_null().all()).item()
    ):
        raise ValueError("UMAP missing: x/y columns must be non-null for all rows.")
    if not df.select(pl.col(_x_col).is_finite().all()).item() or not df.select(pl.col(_y_col).is_finite().all()).item():
        raise ValueError("UMAP missing: x/y columns must be finite for all rows.")

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
    if hue is not None:
        if hue.key not in df.columns:
            raise ValueError(f"UMAP missing: hue column `{hue.key}` not found.")
        if hue.key not in plot_cols:
            plot_cols.append(hue.key)
    df_chart = df.select(plot_cols)

    if hue is not None:
        color_field = hue.key
        color_title = hue.label
        color_tooltip = hue.key
        non_null_count = df_chart.select(pl.col(hue.key).is_not_null().sum()).item()
        if non_null_count == 0:
            raise ValueError(f"UMAP missing: color `{hue.key}` has no non-null values.")
        if safe_is_numeric(df_chart.schema.get(hue.key, pl.Null)):
            if not df_chart.select(pl.col(hue.key).is_finite().all()).item():
                raise ValueError(f"UMAP missing: color `{hue.key}` must be finite for all rows.")
        elif not df_chart.select(pl.col(hue.key).is_not_null().all()).item():
            raise ValueError(f"UMAP missing: color `{hue.key}` must be non-null for all rows.")

        if hue.kind == "categorical" and hue.category_labels:
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
) -> UmapChartView:
    if df.is_empty():
        raise ValueError("UMAP chart missing: dataset is empty.")
    if x_col not in df.columns or y_col not in df.columns:
        raise ValueError("UMAP chart missing: required x/y columns are absent.")
    if not (safe_is_numeric(df.schema.get(x_col, pl.Null)) and safe_is_numeric(df.schema.get(y_col, pl.Null))):
        raise ValueError("UMAP chart missing: x/y columns must be numeric.")
    if (
        not df.select(pl.col(x_col).is_not_null().all()).item()
        or not df.select(pl.col(y_col).is_not_null().all()).item()
    ):
        raise ValueError("UMAP chart missing: x/y columns must be non-null for all rows.")
    if not df.select(pl.col(x_col).is_finite().all()).item() or not df.select(pl.col(y_col).is_finite().all()).item():
        raise ValueError("UMAP chart missing: x/y columns must be finite for all rows.")
    if color_col is not None and color_col not in df.columns:
        raise ValueError(f"UMAP chart missing: color column `{color_col}` is absent.")
    base_cols = _dedupe([x_col, y_col, *(tooltip_cols or [])])
    df_plot = df.select([c for c in base_cols if c in df.columns])
    if df_plot.is_empty():
        raise ValueError("UMAP chart missing: no rows available after selection.")
    if color_col is not None and color_col in df_plot.columns:
        if safe_is_numeric(df_plot.schema.get(color_col, pl.Null)):
            if not df_plot.select(pl.col(color_col).is_finite().all()).item():
                raise ValueError(f"UMAP chart missing: color `{color_col}` must be finite for all rows.")
        elif not df_plot.select(pl.col(color_col).is_not_null().all()).item():
            raise ValueError(f"UMAP chart missing: color `{color_col}` must be non-null for all rows.")
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
) -> ClusterView:
    if df.is_empty():
        raise ValueError("Cluster plot missing: dataset is empty.")
    if id_col not in df.columns:
        raise ValueError(f"Cluster plot missing: required column `{id_col}` is absent.")
    if cluster_col not in df.columns:
        raise ValueError(f"Cluster plot missing: required column `{cluster_col}` is absent.")
    if metric_col not in df.columns:
        raise ValueError(f"Cluster plot missing: required column `{metric_col}` is absent.")
    if not df.select(pl.col(id_col).is_not_null().all()).item():
        raise ValueError(f"Cluster plot missing: `{id_col}` must be non-null for all rows.")
    if not df.select(pl.col(cluster_col).is_not_null().all()).item():
        raise ValueError(f"Cluster plot missing: `{cluster_col}` must be non-null for all rows.")
    if not df.select(pl.col(metric_col).is_not_null().all()).item():
        raise ValueError(f"Cluster plot missing: `{metric_col}` must be non-null for all rows.")
    hue_key = hue.key if hue is not None else ""
    cols = _dedupe([cluster_col, metric_col, id_col, hue_key])
    df_points = df.select([c for c in cols if c in df.columns])
    if df_points.is_empty():
        raise ValueError("Cluster plot missing: no rows available after selection.")

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

    if hue is not None:
        if hue.key not in df_points.columns:
            raise ValueError(f"Cluster plot missing: hue column `{hue.key}` is absent.")
        if not df_points.select(pl.col(hue.key).is_not_null().all()).item():
            raise ValueError(f"Cluster plot missing: hue column `{hue.key}` has null values.")
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
    if df.is_empty():
        raise ValueError("UMAP overlay missing: dataset is empty.")
    if id_col not in df.columns:
        raise ValueError(f"UMAP overlay missing: required column `{id_col}` is absent.")
    if umap_x_col not in df.columns or umap_y_col not in df.columns:
        raise ValueError("UMAP overlay missing: required UMAP x/y columns are absent.")
    if not (
        safe_is_numeric(df.schema.get(umap_x_col, pl.Null)) and safe_is_numeric(df.schema.get(umap_y_col, pl.Null))
    ):
        raise ValueError("UMAP overlay missing: UMAP x/y columns must be numeric.")
    if (
        not df.select(pl.col(umap_x_col).is_not_null().all()).item()
        or not df.select(pl.col(umap_y_col).is_not_null().all()).item()
    ):
        raise ValueError("UMAP overlay missing: UMAP x/y columns must be non-null for all rows.")
    if (
        not df.select(pl.col(umap_x_col).is_finite().all()).item()
        or not df.select(pl.col(umap_y_col).is_finite().all()).item()
    ):
        raise ValueError("UMAP overlay missing: UMAP x/y columns must be finite for all rows.")

    base_cols = [umap_x_col, umap_y_col]
    base_cols.append(id_col)

    if cluster_col not in df.columns:
        raise ValueError(f"UMAP overlay missing: cluster column `{cluster_col}` is absent.")
    df_cluster = df.select(_dedupe(base_cols + [cluster_col]))
    if not df_cluster.select(pl.col(cluster_col).is_not_null().all()).item():
        raise ValueError("UMAP overlay missing: cluster column has null values.")

    score_col = "opal__view__score"
    if score_col not in df.columns:
        raise ValueError("UMAP overlay missing: `opal__view__score` column is absent.")
    df_score = df.select(_dedupe(base_cols + [score_col]))
    if not df_score.select(pl.col(score_col).is_not_null().all()).item():
        raise ValueError("UMAP overlay missing: score column has null values.")

    return UmapOverlayView(df_cluster=df_cluster, df_score=df_score)
