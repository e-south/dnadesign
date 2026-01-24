# ABOUTME: Builds overlay scoring charts for the promoter dashboard.
# ABOUTME: Generates transient score histograms and feature importance views.
"""Overlay scoring helpers for the promoter dashboard."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import altair as alt
import numpy as np
import polars as pl

from .diagnostics import Diagnostics
from .selection import compute_selection_overlay
from .util import dedupe_columns
from .views.sfxi import SFXIParams, compute_sfxi_metrics


@dataclass(frozen=True)
class TransientOverlayResult:
    df_overlay: pl.DataFrame
    df_pred_scored: pl.DataFrame
    diagnostics: Diagnostics
    feature_chart: alt.Chart | None
    hist_chart: alt.Chart | None
    hist_note: str | None


def _chart_title(text: str, subtitle: str | None = None) -> alt.TitleParams:
    if subtitle:
        return alt.TitleParams(text=text, subtitle=subtitle)
    return alt.TitleParams(text=text)


def _with_title(chart: alt.Chart, title: str, subtitle: str | None = None) -> alt.Chart:
    return chart.properties(title=_chart_title(title, subtitle))


def build_feature_importance_chart(
    *,
    feature_importances: np.ndarray | None,
    dataset_name: str | None,
    selected_round: int | None,
    n_labels: int,
    x_dim: int | None,
    model_params: dict[str, Any] | None,
) -> alt.Chart | None:
    if feature_importances is None or not np.size(feature_importances):
        return None
    feature_count = int(np.size(feature_importances))
    df_importance = pl.DataFrame(
        {
            "feature_idx": list(range(feature_count)),
            "importance": np.asarray(feature_importances, dtype=float),
        }
    )
    if not df_importance.height:
        return None

    n_estimators = (model_params or {}).get("n_estimators", "default")
    max_depth = (model_params or {}).get("max_depth", "default")
    df_sorted = df_importance.sort("feature_idx")
    max_ticks = 40
    stride = max(1, math.ceil(feature_count / max_ticks))
    axis_values = list(range(0, feature_count, stride))
    if axis_values and axis_values[-1] != feature_count - 1:
        axis_values.append(feature_count - 1)

    round_label = str(selected_round) if selected_round is not None else "unknown"
    subtitle = (
        f"{dataset_name or 'dataset'} · round={round_label} · "
        f"n_labels={n_labels} · x_dim={x_dim or 'unknown'} · n_features={feature_count} · "
        f"n_estimators={n_estimators} · max_depth={max_depth}"
    )
    chart = (
        alt.Chart(df_sorted)
        .mark_bar()
        .encode(
            x=alt.X(
                "feature_idx:O",
                sort=alt.SortField(field="feature_idx", order="ascending"),
                axis=alt.Axis(
                    values=axis_values,
                    title="Feature index",
                    labelFontSize=8,
                    labelAngle=0,
                ),
            ),
            y=alt.Y("importance:Q", title="Importance"),
            tooltip=["feature_idx", "importance"],
        )
    )
    return (
        _with_title(chart, "Random Forest feature importance", subtitle)
        .properties(width="container", height=240, autosize={"type": "fit", "contains": "padding"})
        .configure_view(stroke=None)
    )


def build_score_histogram(
    *,
    df_pred_scored: pl.DataFrame,
    score_col: str,
    df_sfxi: pl.DataFrame,
    df_train: pl.DataFrame,
    dataset_name: str | None,
    context_label: str | None = None,
    title: str = "Predicted scalar score distribution",
) -> tuple[alt.Chart | None, str | None]:
    if df_pred_scored.is_empty() or score_col not in df_pred_scored.columns:
        label = f"{context_label} " if context_label else ""
        return None, f"No {label}predictions available."

    hist_source = df_pred_scored
    if "__row_id" in hist_source.columns and "__row_id" in df_train.columns:
        hist_source = hist_source.join(df_train.select("__row_id"), on="__row_id", how="anti")
    elif "id" in hist_source.columns and "id" in df_train.columns:
        hist_source = hist_source.join(df_train.select("id"), on="id", how="anti")

    hist_pred = hist_source.filter(pl.col(score_col).is_not_null()).select(pl.col(score_col))
    obs_scores = pl.DataFrame({score_col: []})
    hover_id_col = None
    if "score" in df_sfxi.columns:
        obs_cols = [pl.col("score").alias(score_col)]
        if "id" in df_sfxi.columns:
            obs_cols.append(pl.col("id"))
            hover_id_col = "id"
        elif "__row_id" in df_sfxi.columns:
            obs_cols.append(pl.col("__row_id"))
            hover_id_col = "__row_id"
        obs_scores = df_sfxi.filter(pl.col("score").is_not_null()).select(obs_cols)
    obs_n = int(obs_scores.height)
    obs_cap = 200
    if obs_n > obs_cap:
        obs_plot = obs_scores.sample(n=obs_cap, seed=1)
        obs_note = f"observed labels: n={obs_n} (showing {obs_cap})"
    elif obs_n > 0:
        obs_plot = obs_scores
        obs_note = f"observed labels: n={obs_n}"
    else:
        obs_plot = obs_scores
        obs_note = "observed labels: n=0"

    if hist_pred.is_empty():
        hist_df = pl.DataFrame(
            schema={
                "bin_start": pl.Float64,
                "bin_end": pl.Float64,
                "count": pl.Int64,
            }
        )
        chart = (
            alt.Chart(hist_df)
            .mark_bar()
            .encode(
                x=alt.X(
                    "bin_start:Q",
                    bin=alt.Bin(binned=True),
                    title="Predicted scalar score",
                    scale=alt.Scale(nice=True),
                ),
                x2="bin_end:Q",
                y=alt.Y(
                    "count:Q",
                    title="Count",
                    scale=alt.Scale(domainMin=0),
                ),
            )
        )
        chart = (
            _with_title(
                chart,
                title,
                f"{dataset_name or 'dataset'} · n=0 · {obs_note}",
            )
            .properties(width="container", height=240)
            .configure_view(stroke=None)
        )
        label = f"{context_label} " if context_label else ""
        return chart, f"No {label}predictions available."

    score_min = hist_pred.select(pl.col(score_col).min()).item()
    score_max = hist_pred.select(pl.col(score_col).max()).item()
    if obs_plot.height:
        obs_min = obs_plot.select(pl.col(score_col).min()).item()
        obs_max = obs_plot.select(pl.col(score_col).max()).item()
        if obs_min is not None:
            score_min = min(score_min, obs_min)
        if obs_max is not None:
            score_max = max(score_max, obs_max)
    score_span = (score_max - score_min) if score_max is not None else 0.0
    pad = max(score_span * 0.02, 1e-6)
    x_scale = alt.Scale(domain=[score_min - pad, score_max + pad], nice=True)

    hist_values = hist_pred.select(pl.col(score_col)).to_numpy()
    if score_min is None or score_max is None:
        score_min = 0.0
        score_max = 1.0
    if score_min == score_max:
        score_min -= 1e-3
        score_max += 1e-3
    hist_counts, hist_edges = np.histogram(hist_values, bins=30, range=(score_min, score_max))
    lollipop_top = float(hist_counts.max()) if hist_counts.size else 0.0
    lollipop_scale = 0.85
    lollipop_top *= lollipop_scale
    if lollipop_top <= 0:
        lollipop_top = 1.0
    if obs_plot.height:
        obs_plot = obs_plot.with_columns(
            pl.lit(0).alias("__baseline"),
            pl.lit(lollipop_top).alias("__lollipop_top"),
        )

    hist_df = pl.DataFrame(
        {
            "bin_start": hist_edges[:-1],
            "bin_end": hist_edges[1:],
            "count": hist_counts,
        }
    )
    chart = (
        alt.Chart(hist_df)
        .mark_bar(opacity=0.7)
        .encode(
            x=alt.X(
                "bin_start:Q",
                bin=alt.Bin(binned=True),
                title="Predicted scalar score",
                scale=x_scale,
                axis=alt.Axis(tickCount=7),
            ),
            x2="bin_end:Q",
            y=alt.Y(
                "count:Q",
                title="Count",
                scale=alt.Scale(domainMin=0),
            ),
            tooltip=[
                alt.Tooltip("bin_start:Q", title="bin start"),
                alt.Tooltip("bin_end:Q", title="bin end"),
                alt.Tooltip("count:Q", title="count"),
            ],
        )
    )
    if obs_plot.height:
        obs_rules = (
            alt.Chart(obs_plot)
            .mark_rule(opacity=0.25, strokeDash=[4, 4])
            .encode(
                x=alt.X(f"{score_col}:Q", scale=x_scale),
                y=alt.Y("__baseline:Q", axis=None),
                y2="__lollipop_top:Q",
            )
        )
        tooltip_fields = [alt.Tooltip(f"{score_col}:Q", title="score")]
        if hover_id_col:
            hover_title = "id" if hover_id_col == "id" else "__row_id"
            tooltip_fields.insert(0, alt.Tooltip(f"{hover_id_col}:N", title=hover_title))
        obs_lollipops = (
            alt.Chart(obs_plot)
            .mark_point(filled=True, opacity=0.35, size=20)
            .encode(
                x=alt.X(f"{score_col}:Q", scale=x_scale),
                y=alt.Y("__lollipop_top:Q", axis=None),
                tooltip=tooltip_fields,
            )
        )
        obs_note_chart = (
            alt.Chart(pl.DataFrame({"note": [obs_note]}))
            .mark_text(
                align="left",
                baseline="top",
                dx=0,
                dy=6,
                fontSize=11,
                color="#666666",
            )
            .encode(
                text="note:N",
                x=alt.value(150),
                y=alt.value(12),
            )
        )
        chart = chart + obs_rules + obs_lollipops + obs_note_chart

    chart = (
        _with_title(
            chart,
            title,
            f"{dataset_name or 'dataset'} · n={hist_pred.height} · {obs_note}",
        )
        .properties(width="container", height=240)
        .configure_view(stroke=None)
    )
    return chart, None


def compute_transient_overlay(
    *,
    df_base: pl.DataFrame,
    pred_df: pl.DataFrame | None,
    labels_current_df: pl.DataFrame,
    df_sfxi: pl.DataFrame,
    y_col: str | None,
    sfxi_params: SFXIParams,
    selection_params: dict[str, Any],
    dataset_name: str | None,
    as_of_round: int | None,
    run_id: str | None,
    id_col: str = "id",
    y_hat_col: str = "pred_y_hat",
    feature_importances: np.ndarray | None = None,
    model_params: dict[str, Any] | None = None,
    x_dim: int | None = None,
) -> TransientOverlayResult:
    df_overlay = df_base
    df_pred_scored = df_base.head(0)
    diagnostics = Diagnostics()
    feature_chart = None
    hist_chart = None
    hist_note = None

    if feature_importances is not None:
        feature_chart = build_feature_importance_chart(
            feature_importances=feature_importances,
            dataset_name=dataset_name,
            selected_round=as_of_round,
            n_labels=int(labels_current_df.height),
            x_dim=x_dim,
            model_params=model_params or {},
        )

    if pred_df is None or pred_df.is_empty():
        diagnostics = diagnostics.add_error("No stored predictions available for overlay.")
    elif y_hat_col not in pred_df.columns:
        diagnostics = diagnostics.add_error(f"Missing prediction vector column `{y_hat_col}`.")
    elif y_col is None or y_col not in labels_current_df.columns:
        diagnostics = diagnostics.add_error(f"Missing label column `{y_col}` for overlay scoring.")
    elif labels_current_df.is_empty():
        diagnostics = diagnostics.add_error("No labels available for overlay scoring.")
    else:
        vec_col = "__overlay_vec"
        pred_cols = dedupe_columns([id_col, "__row_id", y_hat_col])
        df_pred = pred_df.select([c for c in pred_cols if c in pred_df.columns]).rename({y_hat_col: vec_col})
        denom_pool = labels_current_df.select(pl.col(y_col).alias(vec_col))
        try:
            pred_result = compute_sfxi_metrics(
                df=df_pred,
                vec_col=vec_col,
                params=sfxi_params,
                denom_pool_df=denom_pool,
            )
        except Exception as exc:
            diagnostics = diagnostics.add_error(f"SFXI overlay failed: {exc}")
        else:
            df_pred_scored = pred_result.df
            if df_pred_scored.is_empty():
                diagnostics = diagnostics.add_warning("No valid predictions after overlay scoring.")
            else:
                df_pred_scored = df_pred_scored.with_columns(
                    pl.col("logic_fidelity").alias("opal__overlay__logic_fidelity").cast(pl.Float64),
                    pl.col("effect_scaled").alias("opal__overlay__effect_scaled").cast(pl.Float64),
                    pl.col("score").alias("opal__overlay__score").cast(pl.Float64),
                    pl.lit(run_id).alias("opal__overlay__run_id"),
                    pl.lit(as_of_round).alias("opal__overlay__round"),
                )

                sel_ids = np.asarray(df_pred_scored.get_column(id_col).to_list(), dtype=str)
                sel_scores = (
                    df_pred_scored.select(pl.col("opal__overlay__score").fill_null(float("nan")).cast(pl.Float64))
                    .to_numpy()
                    .ravel()
                )
                try:
                    ranks, selected, warnings = compute_selection_overlay(
                        ids=sel_ids,
                        scores=sel_scores,
                        selection_params=selection_params,
                    )
                    if warnings:
                        diagnostics = diagnostics.add_warning("Selection objective warning: " + "; ".join(warnings))
                except Exception as exc:
                    diagnostics = diagnostics.add_error(f"Selection overlay error: {exc}")
                    ranks = np.full(sel_scores.shape, None, dtype=object)
                    selected = np.full(sel_scores.shape, False, dtype=bool)
                df_pred_scored = df_pred_scored.with_columns(
                    pl.Series("opal__overlay__rank", ranks),
                    pl.Series("opal__overlay__top_k", selected),
                )

                overlay_cols = [
                    "opal__overlay__score",
                    "opal__overlay__rank",
                    "opal__overlay__logic_fidelity",
                    "opal__overlay__effect_scaled",
                    "opal__overlay__top_k",
                    "opal__overlay__run_id",
                    "opal__overlay__round",
                ]
                pred_cols = [c for c in [id_col, "__row_id", *overlay_cols] if c in df_pred_scored.columns]
                overlay_drop_cols = [col for col in overlay_cols if col in df_base.columns and col != id_col]
                df_overlay_base = df_base.drop(overlay_drop_cols) if overlay_drop_cols else df_base
                df_overlay = df_overlay_base.join(df_pred_scored.select(pred_cols), on=id_col, how="left")

                hist_chart, hist_note = build_score_histogram(
                    df_pred_scored=df_pred_scored,
                    score_col="opal__overlay__score",
                    df_sfxi=df_sfxi,
                    df_train=labels_current_df,
                    dataset_name=dataset_name,
                    context_label="overlay",
                )

    df_overlay = _ensure_overlay_cols(df_overlay)
    if as_of_round is not None:
        df_overlay = df_overlay.with_columns(pl.lit(int(as_of_round)).alias("opal__overlay__round"))
    if run_id is not None:
        df_overlay = df_overlay.with_columns(pl.lit(str(run_id)).alias("opal__overlay__run_id"))
    return TransientOverlayResult(
        df_overlay=df_overlay,
        df_pred_scored=df_pred_scored,
        diagnostics=diagnostics,
        feature_chart=feature_chart,
        hist_chart=hist_chart,
        hist_note=hist_note,
    )


def _ensure_overlay_cols(df: pl.DataFrame) -> pl.DataFrame:
    overlay_cols = [
        "opal__overlay__score",
        "opal__overlay__rank",
        "opal__overlay__logic_fidelity",
        "opal__overlay__effect_scaled",
        "opal__overlay__top_k",
        "opal__overlay__run_id",
        "opal__overlay__round",
    ]
    missing = [c for c in overlay_cols if c not in df.columns]
    if not missing:
        return df
    return df.with_columns([pl.lit(None).alias(col) for col in missing])
