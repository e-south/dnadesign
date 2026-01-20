"""Shared chart builders for marimo dashboards."""

from __future__ import annotations

from typing import Iterable

import altair as alt
import polars as pl

from .util import safe_is_numeric


def _chart_title(text: str, subtitle: str | None = None) -> alt.TitleParams:
    if subtitle:
        return alt.TitleParams(text=text, subtitle=subtitle)
    return alt.TitleParams(text=text)


def with_title(chart: alt.Chart, title: str, subtitle: str | None = None) -> alt.Chart:
    return chart.properties(title=_chart_title(title, subtitle))


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


def _dedupe(cols: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for col in cols:
        if col in seen:
            continue
        seen.add(col)
        out.append(col)
    return out


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
    if df.is_empty() or x_col not in df.columns or y_col not in df.columns:
        return None
    base_cols = _dedupe([x_col, y_col, *(tooltip_cols or [])])
    df_plot = df.select([c for c in base_cols if c in df.columns]).filter(
        pl.col(x_col).is_not_null() & pl.col(y_col).is_not_null()
    )
    if df_plot.is_empty():
        return None

    enc = {
        "x": alt.X(x_col, title="UMAP X"),
        "y": alt.Y(y_col, title="UMAP Y"),
        "tooltip": [c for c in tooltip_cols or [] if c in df_plot.columns],
    }
    if color_col and color_col in df_plot.columns:
        enc["color"] = alt.Color(
            f"{color_col}:Q" if df_plot.schema.get(color_col, pl.Null).is_numeric() else f"{color_col}:N",
            title=color_title or color_col,
            legend=alt.Legend(title=color_title or color_col, format=".2f", tickCount=5),
        )
    else:
        enc["color"] = alt.value("#4C78A8")

    chart = alt.Chart(df_plot).mark_circle(opacity=opacity, stroke=None, strokeWidth=0, size=size).encode(**enc)
    return with_title(chart, title, subtitle).properties(width=plot_size, height=plot_size)


def build_cluster_chart(
    *,
    df: pl.DataFrame,
    cluster_col: str,
    metric_col: str,
    metric_label: str,
    hue_value: str | None,
    hue_label_display: str | None,
    rf_prefix: str,
    score_source_label: str,
    dataset_name: str | None,
    id_col: str,
    title: str,
    plot_height: int = 240,
) -> alt.Chart | None:
    if df.is_empty() or cluster_col not in df.columns or metric_col not in df.columns:
        return None
    cols = _dedupe([cluster_col, metric_col, id_col, hue_value or ""])
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
    sort_field = alt.SortField(field=f"{cluster_col}__ord", order="ascending")

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

    label_map = {
        "opal__score__top_k": ("Top-K", "Not Top-K", f"{score_source_label} Top-K"),
        "opal__transient__top_k": ("Top-K", "Not Top-K", f"{rf_prefix} Top-K"),
        "opal__transient__observed_event": ("Observed", "Not observed", "Observed events (ingest_y)"),
        "opal__transient__sfxi_scored_label": ("SFXI label", "Not label", "SFXI scored label"),
    }
    hue_label = hue_label_display if hue_label_display else "(none)"
    color_encoding = alt.Undefined
    color_tooltip = None
    label_col = None
    top_k_mode = False

    if hue_value == "Leiden cluster":
        hue_label = "Leiden cluster"
        color_tooltip = cluster_col
        color_encoding = alt.Color(
            f"{cluster_col}:N",
            title="Leiden cluster",
            scale=alt.Scale(range=okabe_ito),
            legend=alt.Legend(title="Leiden cluster"),
        )
    elif hue_value in label_map and hue_value in df_points.columns:
        label_col = f"{hue_value}__label"
        yes_label, no_label, hue_label = label_map[hue_value]
        top_k_mode = hue_value in {"opal__transient__top_k", "opal__score__top_k"}
        df_points = df_points.with_columns(
            pl.when(pl.col(hue_value)).then(pl.lit(yes_label)).otherwise(pl.lit(no_label)).alias(label_col)
        )
        color_tooltip = label_col
        color_scale = (
            alt.Scale(domain=[yes_label, no_label], range=["#D62728", "#B0B0B0"])
            if top_k_mode
            else alt.Scale(domain=[yes_label, no_label], range=[okabe_ito[2], "#B0B0B0"])
        )
        color_encoding = alt.Color(
            f"{label_col}:N",
            title=hue_label,
            scale=color_scale,
            legend=alt.Legend(title=hue_label),
        )
    elif hue_value and hue_value in df_points.columns:
        hue_dtype = df_points.schema.get(hue_value)
        if hue_dtype is not None and safe_is_numeric(hue_dtype):
            hue_label = hue_label_display or hue_value
            color_tooltip = hue_value
            color_encoding = alt.Color(
                f"{hue_value}:Q",
                title=hue_label,
                legend=alt.Legend(title=hue_label, format=".2f", tickCount=5),
            )

    tooltip_cols = [c for c in [cluster_col, metric_col, id_col, color_tooltip] if c and c in df_points.columns]

    metric_type = "Q" if safe_is_numeric(df_points.schema.get(metric_col, pl.Null)) else "N"

    if top_k_mode and label_col:
        yes_label, no_label, _ = label_map[hue_value]
        df_not = df_points.filter(pl.col(label_col) == no_label)
        df_top = df_points.filter(pl.col(label_col) == yes_label)
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
                y=alt.Y(f"{metric_col}:{metric_type}", title=metric_label),
                color=color_encoding,
                tooltip=tooltip_cols,
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
                y=alt.Y(f"{metric_col}:{metric_type}", title=metric_label),
                color=color_encoding,
                tooltip=tooltip_cols,
            )
        )
    else:
        base = (
            alt.Chart(df_points)
            .mark_circle(size=22, opacity=0.6)
            .encode(
                x=alt.X(
                    f"{cluster_col}:N",
                    sort=sort_field,
                    title="Leiden cluster",
                    axis=alt.Axis(labelAngle=90, labelFontSize=8),
                ),
                xOffset="__jitter:Q",
                y=alt.Y(f"{metric_col}:{metric_type}", title=metric_label),
                color=color_encoding if color_encoding is not alt.Undefined else alt.value(okabe_ito[4]),
                tooltip=tooltip_cols,
            )
        )

    subtitle = f"{dataset_name or 'dataset'} · y={metric_label} · hue={hue_label} · n={df_points.height}"
    return (
        with_title(base, title, subtitle).properties(width="container", height=plot_height).configure_view(stroke=None)
    )


def build_umap_overlay_charts(
    *,
    df: pl.DataFrame,
    dataset_name: str | None,
    campaign_slug: str | None,
    use_artifact: bool,
    plot_size: int = 420,
    id_col: str = "id",
    umap_x_col: str = "cluster__ldn_v1__umap_x",
    umap_y_col: str = "cluster__ldn_v1__umap_y",
    cluster_col: str = "cluster__ldn_v1",
) -> tuple[alt.Chart | None, alt.Chart | None]:
    if df.is_empty() or umap_x_col not in df.columns or umap_y_col not in df.columns:
        return None, None

    base_cols = [umap_x_col, umap_y_col]
    if id_col in df.columns:
        base_cols.append(id_col)

    cluster_chart = None
    if cluster_col in df.columns:
        df_cluster = df.select(_dedupe(base_cols + [cluster_col])).filter(pl.col(cluster_col).is_not_null())
        if not df_cluster.is_empty():
            chart = build_umap_chart(
                df=df_cluster,
                x_col=umap_x_col,
                y_col=umap_y_col,
                color_col=cluster_col,
                color_title="Leiden cluster",
                title="UMAP colored by Leiden cluster",
                subtitle=f"{dataset_name or 'dataset'} · n={df_cluster.height}",
                tooltip_cols=[c for c in [id_col, cluster_col] if c in df_cluster.columns],
                plot_size=plot_size,
            )
            if chart is not None:
                cluster_chart = chart

    score_col = None
    score_title = None
    score_chart_title = None
    if "opal__score__scalar" in df.columns:
        score_col = "opal__score__scalar"
        score_title = "Selected score source (scalar)"
        score_chart_title = "UMAP colored by selected score source"
    elif campaign_slug:
        latest_col = f"opal__{campaign_slug}__latest_pred_scalar"
        if latest_col in df.columns:
            score_col = latest_col
            score_title = "OPAL latest predicted scalar"
            score_chart_title = "UMAP colored by OPAL latest scalar"
    if score_col is None and "opal__transient__score" in df.columns:
        score_col = "opal__transient__score"
        score_prefix = "OPAL artifact" if use_artifact else "Transient"
        score_title = f"{score_prefix} score (SFXI)"
        score_chart_title = f"UMAP colored by {score_prefix} score (SFXI)"

    score_chart = None
    if score_col is not None:
        df_score = df.select(_dedupe(base_cols + [score_col])).filter(pl.col(score_col).is_not_null())
        if not df_score.is_empty():
            chart = build_umap_chart(
                df=df_score,
                x_col=umap_x_col,
                y_col=umap_y_col,
                color_col=score_col,
                color_title=score_title,
                title=score_chart_title or "UMAP colored by score",
                subtitle=f"{dataset_name or 'dataset'} · n={df_score.height}",
                tooltip_cols=[c for c in [id_col, score_col] if c in df_score.columns],
                plot_size=plot_size,
            )
            if chart is not None:
                score_chart = chart

    return cluster_chart, score_chart
