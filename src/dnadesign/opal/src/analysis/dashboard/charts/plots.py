"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/analysis/dashboard/charts/plots.py

Altair chart builders for dashboard UMAP and cluster plots. Consumes view data
from analysis.dashboard.views.plots.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import altair as alt
import polars as pl
from altair.utils.schemapi import UndefinedType

from ..hues import HueOption
from ..util import safe_is_numeric
from ..views.plots import (
    ColorSpec,
    prepare_cluster_view,
    prepare_umap_chart_view,
    prepare_umap_explorer_view,
    prepare_umap_overlay_view,
)


def _chart_title(text: str, subtitle: str | None = None) -> alt.TitleParams:
    if subtitle:
        return alt.TitleParams(text=text, subtitle=subtitle)
    return alt.TitleParams(text=text)


def with_title(chart: alt.Chart, title: str, subtitle: str | None = None) -> alt.Chart:
    return chart.properties(title=_chart_title(title, subtitle))


def _color_encoding(spec: ColorSpec | None) -> alt.Color | UndefinedType:
    if spec is None:
        return alt.Undefined
    if spec.kind == "numeric":
        return alt.Color(
            f"{spec.field}:Q",
            title=spec.title,
            legend=alt.Legend(title=spec.title, format=".2f", tickCount=5),
        )
    if spec.kind == "categorical":
        scale = None
        if spec.domain and spec.range:
            scale = alt.Scale(domain=spec.domain, range=spec.range)
        elif spec.scheme:
            scale = alt.Scale(scheme=spec.scheme)
        elif spec.range:
            scale = alt.Scale(range=spec.range)
        return alt.Color(
            f"{spec.field}:N",
            title=spec.title,
            scale=scale,
            legend=alt.Legend(title=spec.title),
        )
    return alt.Undefined


def _tooltip_fields(df: pl.DataFrame, cols: Iterable[str]) -> list[alt.Tooltip]:
    seen: set[str] = set()
    tooltips: list[alt.Tooltip] = []
    for col in cols:
        if not col or col in seen or col not in df.columns:
            continue
        seen.add(col)
        kind = "Q" if safe_is_numeric(df.schema.get(col, pl.Null)) else "N"
        tooltips.append(alt.Tooltip(f"{col}:{kind}", title=col))
    return tooltips


@dataclass(frozen=True)
class UmapExplorerResult:
    chart: alt.Chart
    df_plot: pl.DataFrame
    valid: bool
    note: str | None


def build_umap_explorer_chart(
    *,
    df: pl.DataFrame,
    x_col: str | None,
    y_col: str | None,
    hue: HueOption | None,
    point_size: float,
    opacity: float,
    plot_size: int,
    dataset_name: str | None,
) -> UmapExplorerResult:
    view = prepare_umap_explorer_view(df=df, x_col=x_col, y_col=y_col, hue=hue)
    if not view.valid:
        chart = (
            alt.Chart(view.df_plot)
            .mark_circle(stroke=None, strokeWidth=0)
            .encode(
                x=alt.X(view.x_col),
                y=alt.Y(view.y_col),
                tooltip=_tooltip_fields(view.df_plot, ["id", "__row_id", view.x_col, view.y_col]),
            )
            .properties(width=plot_size, height=plot_size)
        )
        chart = with_title(
            chart,
            "UMAP explorer (Evo2 embedding)",
            f"{dataset_name or 'dataset'} · color=none",
        )
        return UmapExplorerResult(chart=chart, df_plot=df, valid=False, note=view.note)

    color_encoding = _color_encoding(view.color_spec)
    brush = alt.selection_interval(name="umap_brush", encodings=["x", "y"])
    tooltip_candidates = ["id", "__row_id", view.x_col, view.y_col]
    if view.color_tooltip:
        tooltip_candidates.append(view.color_tooltip)
    tooltip_fields = _tooltip_fields(view.df_plot, tooltip_candidates)

    chart = (
        alt.Chart(view.df_plot)
        .mark_circle(size=point_size, opacity=opacity, stroke=None, strokeWidth=0)
        .encode(
            x=alt.X(view.x_col, title=view.x_col),
            y=alt.Y(view.y_col, title=view.y_col),
            color=color_encoding,
            tooltip=tooltip_fields,
        )
        .add_params(brush)
        .properties(width=plot_size, height=plot_size)
    )
    if "opal__view__observed" in df.columns and view.x_col in df.columns and view.y_col in df.columns:
        df_obs = df.select([view.x_col, view.y_col, "opal__view__observed"]).filter(pl.col("opal__view__observed"))
        if df_obs.height:
            obs_layer = (
                alt.Chart(df_obs)
                .mark_circle(
                    size=point_size * 1.6,
                    stroke="#000000",
                    strokeWidth=1.3,
                    fillOpacity=0.0,
                    opacity=1.0,
                )
                .encode(
                    x=alt.X(view.x_col, title=view.x_col),
                    y=alt.Y(view.y_col, title=view.y_col),
                    tooltip=tooltip_fields,
                )
            )
            chart = chart + obs_layer
    if "opal__view__top_k" in view.df_plot.columns:
        df_top = view.df_plot.filter(pl.col("opal__view__top_k"))
        if df_top.height:
            top_layer = (
                alt.Chart(df_top)
                .mark_circle(
                    size=point_size * 1.8,
                    stroke="#000000",
                    strokeWidth=1.5,
                    fillOpacity=0.0,
                    opacity=1.0,
                )
                .encode(
                    x=alt.X(view.x_col, title=view.x_col),
                    y=alt.Y(view.y_col, title=view.y_col),
                    tooltip=tooltip_fields,
                )
            )
            chart = chart + top_layer

    color_context = (view.color_spec.title if view.color_spec else None) or "none"
    chart = with_title(
        chart,
        "UMAP explorer (Evo2 embedding)",
        f"{dataset_name or 'dataset'} · color={color_context}",
    )
    return UmapExplorerResult(chart=chart, df_plot=df, valid=True, note=view.note)


def build_umap_chart(
    *,
    df: pl.DataFrame,
    x_col: str,
    y_col: str,
    color_col: str | None,
    color_title: str | None,
    title: str,
    subtitle: str,
    tooltip_cols: Iterable[str] | None = None,
    size: int = 40,
    opacity: float = 0.7,
    plot_size: int = 420,
) -> alt.Chart | None:
    view = prepare_umap_chart_view(
        df=df,
        x_col=x_col,
        y_col=y_col,
        color_col=color_col,
        color_title=color_title,
        tooltip_cols=tooltip_cols,
    )
    if view is None:
        return None

    tooltip_fields = _tooltip_fields(view.df_plot, view.tooltip_cols)
    enc = {
        "x": alt.X(view.x_col, title="UMAP X"),
        "y": alt.Y(view.y_col, title="UMAP Y"),
        "tooltip": tooltip_fields,
    }
    if view.color_col:
        kind = "Q" if view.color_kind == "numeric" else "N"
        enc["color"] = alt.Color(
            f"{view.color_col}:{kind}",
            title=view.color_title or view.color_col,
            legend=alt.Legend(title=view.color_title or view.color_col, format=".2f", tickCount=5),
        )
    else:
        enc["color"] = alt.value("#4C78A8")

    chart = alt.Chart(view.df_plot).mark_circle(opacity=opacity, stroke=None, strokeWidth=0, size=size).encode(**enc)
    return with_title(chart, title, subtitle).properties(width=plot_size, height=plot_size)


def build_cluster_chart(
    *,
    df: pl.DataFrame,
    cluster_col: str,
    metric_col: str,
    metric_label: str,
    hue: HueOption | None,
    dataset_name: str | None,
    id_col: str,
    title: str,
    plot_height: int = 240,
) -> alt.Chart | None:
    view = prepare_cluster_view(
        df=df,
        cluster_col=cluster_col,
        metric_col=metric_col,
        hue=hue,
        id_col=id_col,
    )
    if view is None:
        return None

    sort_field = alt.SortField(field=view.sort_field, order="ascending")
    tooltip_cols = [
        c for c in [cluster_col, metric_col, id_col, view.color_tooltip] if c and c in view.df_points.columns
    ]
    tooltip_fields = _tooltip_fields(view.df_points, tooltip_cols)

    color_encoding = _color_encoding(view.color_spec)

    if view.top_k_mode and view.label_col and view.yes_label and view.no_label:
        df_not = view.df_points.filter(pl.col(view.label_col) == view.no_label)
        df_top = view.df_points.filter(pl.col(view.label_col) == view.yes_label)
        base = (
            alt.Chart(df_not)
            .mark_circle(size=18, opacity=0.45)
            .encode(
                x=alt.X(
                    f"{cluster_col}:N",
                    sort=sort_field,
                    title="Leiden cluster",
                    axis=alt.Axis(labelAngle=90, labelFontSize=8),
                ),
                xOffset="__jitter:Q",
                y=alt.Y(f"{metric_col}:{view.metric_type}", title=metric_label),
                color=color_encoding,
                tooltip=tooltip_fields,
            )
        ) + (
            alt.Chart(df_top)
            .mark_circle(size=36, opacity=0.85)
            .encode(
                x=alt.X(
                    f"{cluster_col}:N",
                    sort=sort_field,
                    title="Leiden cluster",
                    axis=alt.Axis(labelAngle=90, labelFontSize=8),
                ),
                xOffset="__jitter:Q",
                y=alt.Y(f"{metric_col}:{view.metric_type}", title=metric_label),
                color=color_encoding,
                tooltip=tooltip_fields,
            )
        )
    else:
        base = (
            alt.Chart(view.df_points)
            .mark_circle(size=22, opacity=0.6)
            .encode(
                x=alt.X(
                    f"{cluster_col}:N",
                    sort=sort_field,
                    title="Leiden cluster",
                    axis=alt.Axis(labelAngle=90, labelFontSize=8),
                ),
                xOffset="__jitter:Q",
                y=alt.Y(f"{metric_col}:{view.metric_type}", title=metric_label),
                color=color_encoding if color_encoding is not alt.Undefined else alt.value(view.okabe_ito[4]),
                tooltip=tooltip_fields,
            )
        )

    subtitle = f"{dataset_name or 'dataset'} · y={metric_label} · hue={view.hue_label} · n={view.df_points.height}"
    return (
        with_title(base, title, subtitle).properties(width="container", height=plot_height).configure_view(stroke=None)
    )


def build_umap_overlay_charts(
    *,
    df: pl.DataFrame,
    dataset_name: str | None,
    plot_size: int = 420,
    id_col: str = "id",
    umap_x_col: str = "cluster__ldn_v1__umap_x",
    umap_y_col: str = "cluster__ldn_v1__umap_y",
    cluster_col: str = "cluster__ldn_v1",
) -> tuple[alt.Chart | None, alt.Chart | None]:
    view = prepare_umap_overlay_view(
        df=df,
        id_col=id_col,
        umap_x_col=umap_x_col,
        umap_y_col=umap_y_col,
        cluster_col=cluster_col,
    )

    cluster_chart = None
    if view.df_cluster is not None:
        chart = build_umap_chart(
            df=view.df_cluster,
            x_col=umap_x_col,
            y_col=umap_y_col,
            color_col=cluster_col,
            color_title="Leiden cluster",
            title="UMAP colored by Leiden cluster",
            subtitle=f"{dataset_name or 'dataset'} · n={view.df_cluster.height}",
            tooltip_cols=[c for c in [id_col, cluster_col] if c in view.df_cluster.columns],
            plot_size=plot_size,
        )
        if chart is not None:
            cluster_chart = chart

    score_col = "opal__view__score" if "opal__view__score" in df.columns else None
    score_chart = None
    if view.df_score is not None and score_col is not None:
        chart = build_umap_chart(
            df=view.df_score,
            x_col=umap_x_col,
            y_col=umap_y_col,
            color_col=score_col,
            color_title="Score",
            title="UMAP colored by score",
            subtitle=f"{dataset_name or 'dataset'} · n={view.df_score.height}",
            tooltip_cols=[c for c in [id_col, score_col] if c in view.df_score.columns],
            plot_size=plot_size,
        )
        if chart is not None:
            score_chart = chart

    return cluster_chart, score_chart
