"""Overlay artifact RF helpers for the promoter dashboard."""

from __future__ import annotations

import math
from collections.abc import Hashable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import altair as alt
import numpy as np
import polars as pl

from .datasets import CampaignInfo
from .diagnostics import Diagnostics
from .models import get_feature_importances, load_round_ctx_from_dir
from .selection import compute_selection_overlay
from .sfxi import SFXIParams, compute_sfxi_metrics
from .util import dedupe_columns, list_series_to_numpy
from .y_ops import apply_y_ops_inverse


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


def _build_feature_importance_chart(
    *,
    feature_importances: np.ndarray,
    dataset_name: str | None,
    selected_round: int | None,
    n_labels: int,
    x_dim: int,
    model_params: dict[str, Any],
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

    n_estimators = model_params.get("n_estimators", "default")
    max_depth = model_params.get("max_depth", "default")
    df_sorted = df_importance.sort("feature_idx")
    max_ticks = 40
    stride = max(1, math.ceil(feature_count / max_ticks))
    axis_values = list(range(0, feature_count, stride))
    if axis_values and axis_values[-1] != feature_count - 1:
        axis_values.append(feature_count - 1)

    round_label = str(selected_round) if selected_round is not None else "headless"
    subtitle = (
        f"{dataset_name or 'dataset'} · round={round_label} · "
        f"n_labels={n_labels} · x_dim={x_dim} · n_features={feature_count} · "
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


def _build_histogram_chart(
    *,
    df_pred_scored: pl.DataFrame,
    df_sfxi: pl.DataFrame,
    df_train: pl.DataFrame,
    dataset_name: str | None,
) -> tuple[alt.Chart | None, str | None]:
    score_col = "opal__overlay__score"
    if df_pred_scored.is_empty() or score_col not in df_pred_scored.columns:
        return None, "No overlay predictions available."

    hist_source = df_pred_scored
    if "__row_id" in hist_source.columns and "__row_id" in df_train.columns:
        hist_source = hist_source.join(df_train.select("__row_id"), on="__row_id", how="anti")
    elif "id" in hist_source.columns and "id" in df_train.columns:
        hist_source = hist_source.join(df_train.select("id"), on="id", how="anti")

    hist_pred = hist_source.filter(pl.col(score_col).is_not_null()).select(pl.col(score_col))
    obs_scores = pl.DataFrame({score_col: []})
    if "score" in df_sfxi.columns:
        obs_scores = df_sfxi.filter(pl.col("score").is_not_null()).select(pl.col("score").alias(score_col))
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
                "Predicted scalar score distribution",
                f"{dataset_name or 'dataset'} · n=0 · {obs_note}",
            )
            .properties(width="container", height=240)
            .configure_view(stroke=None)
        )
        return chart, "No overlay predictions available."

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
        obs_lollipops = (
            alt.Chart(obs_plot)
            .mark_point(filled=True, opacity=0.35, size=20)
            .encode(
                x=alt.X(f"{score_col}:Q", scale=x_scale),
                y=alt.Y("__lollipop_top:Q", axis=None),
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
            "Predicted scalar score distribution",
            f"{dataset_name or 'dataset'} · n={hist_pred.height} · {obs_note}",
        )
        .properties(width="container", height=240)
        .configure_view(stroke=None)
    )
    return chart, None


def compute_transient_overlay(
    *,
    df_base: pl.DataFrame,
    labels_asof_df: pl.DataFrame,
    labels_current_df: pl.DataFrame,
    df_sfxi: pl.DataFrame,
    context: object | None = None,
    campaign_info: CampaignInfo | None = None,
    campaign_slug: str | None = None,
    x_col: str | None,
    y_col: str | None,
    sfxi_params: SFXIParams,
    selected_round: int | None,
    artifact_model: Any | None,
    artifact_round_dir: Path | None,
    run_id: str | None,
    dataset_name: str | None = None,
    pred_cache: dict | None = None,
    cache_key: Hashable | None = None,
    compute_if_missing: bool = True,
) -> TransientOverlayResult:
    df_overlay = df_base
    df_pred_scored = df_base.head(0)
    diagnostics = Diagnostics()
    feature_chart = None
    hist_chart = None
    hist_note = None

    def _note(message: str) -> None:
        nonlocal diagnostics
        diagnostics = diagnostics.add_note(message)

    def _warn(message: str) -> None:
        nonlocal diagnostics
        diagnostics = diagnostics.add_warning(message)

    def _error(message: str) -> None:
        nonlocal diagnostics
        diagnostics = diagnostics.add_error(message)

    if campaign_info is None and context is not None:
        campaign_info = getattr(context, "campaign_info", None)
    if campaign_slug is None and context is not None:
        campaign_slug = getattr(getattr(context, "campaign_info", None), "slug", None)
    if dataset_name is None and context is not None:
        dataset_name = getattr(context, "dataset_name", None)

    effective_round = selected_round
    round_mode = "explicit"
    run_id_value = run_id
    source_value = "artifact"

    def _with_overlay_provenance(frame: pl.DataFrame) -> pl.DataFrame:
        return frame.with_columns(
            pl.lit(source_value).alias("opal__overlay__source"),
            pl.lit(campaign_slug).alias("opal__overlay__campaign_slug"),
            pl.lit(run_id_value).alias("opal__overlay__run_id"),
            pl.lit(effective_round).alias("opal__overlay__round"),
            pl.lit(round_mode).alias("opal__overlay__round_mode"),
        )

    if run_id is None:
        _error("Artifact overlay requires an explicit run_id selection.")
        return TransientOverlayResult(
            df_overlay=_with_overlay_provenance(df_overlay),
            df_pred_scored=df_pred_scored,
            diagnostics=diagnostics,
            feature_chart=feature_chart,
            hist_chart=hist_chart,
            hist_note=hist_note,
        )
    if effective_round is None:
        _error("Artifact overlay requires an explicit round selection.")
        return TransientOverlayResult(
            df_overlay=_with_overlay_provenance(df_overlay),
            df_pred_scored=df_pred_scored,
            diagnostics=diagnostics,
            feature_chart=feature_chart,
            hist_chart=hist_chart,
            hist_note=hist_note,
        )
    if artifact_model is None:
        _error("Artifact model unavailable; overlay disabled.")
        return TransientOverlayResult(
            df_overlay=_with_overlay_provenance(df_overlay),
            df_pred_scored=df_pred_scored,
            diagnostics=diagnostics,
            feature_chart=feature_chart,
            hist_chart=hist_chart,
            hist_note=hist_note,
        )

    if campaign_info is None:
        _error("Overlay predictions unavailable (campaign unsupported).")
        return TransientOverlayResult(
            df_overlay=_with_overlay_provenance(df_overlay),
            df_pred_scored=df_pred_scored,
            diagnostics=diagnostics,
            feature_chart=feature_chart,
            hist_chart=hist_chart,
            hist_note=hist_note,
        )

    # cache_key is a stable, hashable fingerprint (e.g. JSON string payload).
    cache_hit = False
    cached_pred = None
    cached_feature_chart = None
    if pred_cache is not None and cache_key is not None:
        cached_pred = pred_cache.get(cache_key)
        if cached_pred is not None:
            cache_hit = True
            cached_feature_chart = cached_pred.get("feature_chart")

    df_pred = None
    y_ops = list(campaign_info.y_ops or [])
    yops_ctx = None
    yops_inverse_ready = True
    df_train = labels_asof_df if labels_asof_df is not None else df_base.head(0)
    if df_train is None:
        df_train = df_base.head(0)
    if df_train.is_empty() and labels_current_df is not None and not labels_current_df.is_empty():
        df_train = labels_current_df

    if cache_hit:
        if isinstance(cached_pred, dict):
            df_pred = cached_pred.get("df_pred")
            yops_inverse_ready = bool(cached_pred.get("yops_inverse_ready", True))
            if cached_feature_chart is not None:
                feature_chart = cached_feature_chart
            cached_train_ids = cached_pred.get("train_ids")
            if isinstance(cached_train_ids, pl.DataFrame):
                df_train = cached_train_ids
        if df_pred is None or df_pred.is_empty() or "opal__overlay__y_vec" not in df_pred.columns:
            _warn("Overlay cache entry missing predictions; recompute required.")
            cache_hit = False
        else:
            _note("Overlay cache hit; reusing predictions.")

    if not cache_hit and not compute_if_missing:
        _warn("Overlay predictions not computed yet. Press 'Compute overlay' to run the model.")
        return TransientOverlayResult(
            df_overlay=_with_overlay_provenance(df_overlay),
            df_pred_scored=df_pred_scored,
            diagnostics=diagnostics,
            feature_chart=feature_chart,
            hist_chart=hist_chart,
            hist_note=hist_note,
        )

    if not cache_hit:
        if x_col is None or x_col not in df_base.columns:
            _error(f"Missing X column: `{x_col}`.")
            return TransientOverlayResult(
                df_overlay=_with_overlay_provenance(df_overlay),
                df_pred_scored=df_pred_scored,
                diagnostics=diagnostics,
                feature_chart=feature_chart,
                hist_chart=hist_chart,
                hist_note=hist_note,
            )
        df_x_non_null = df_base.filter(pl.col(x_col).is_not_null())
        df_x_non_null = df_x_non_null.filter(pl.col(x_col).list.len() > 0)
        if df_x_non_null.is_empty():
            _error("No candidate X vectors available for prediction.")
            return TransientOverlayResult(
                df_overlay=_with_overlay_provenance(df_overlay),
                df_pred_scored=df_pred_scored,
                diagnostics=diagnostics,
                feature_chart=feature_chart,
                hist_chart=hist_chart,
                hist_note=hist_note,
            )
        len_stats = df_x_non_null.select(
            pl.col(x_col).list.len().min().alias("min_len"),
            pl.col(x_col).list.len().max().alias("max_len"),
        )
        min_len, max_len = len_stats.row(0)
        if min_len is None or max_len is None or int(min_len) != int(max_len):
            _error("X vectors must be fixed-length for artifact predictions.")
            return TransientOverlayResult(
                df_overlay=_with_overlay_provenance(df_overlay),
                df_pred_scored=df_pred_scored,
                diagnostics=diagnostics,
                feature_chart=feature_chart,
                hist_chart=hist_chart,
                hist_note=hist_note,
            )

        x_dim = int(min_len)
        if artifact_round_dir is not None:
            yops_ctx, yops_err = load_round_ctx_from_dir(artifact_round_dir)
            if yops_err:
                _warn(f"Round context load failed: {yops_err}")
        if yops_ctx is None and y_ops:
            _warn("No round_ctx.json available; cannot invert Y-ops for artifact predictions.")
            yops_inverse_ready = False
        _note("Using OPAL artifact model for predictions.")

        feature_chart = _build_feature_importance_chart(
            feature_importances=get_feature_importances(artifact_model),
            dataset_name=dataset_name,
            selected_round=effective_round,
            n_labels=int(df_train.height),
            x_dim=x_dim,
            model_params=dict(campaign_info.model_params or {}),
        )

        _note("Artifact RF: predicting over full pool ...")
        df_x_all = df_base.select(dedupe_columns(["__row_id", "id", x_col]))
        df_x_valid = df_x_all.filter(pl.col(x_col).is_not_null() & (pl.col(x_col).list.len() == x_dim))
        if df_x_valid.is_empty():
            _error("No candidate X vectors available for prediction.")
            return TransientOverlayResult(
                df_overlay=_with_overlay_provenance(df_overlay),
                df_pred_scored=df_pred_scored,
                diagnostics=diagnostics,
                feature_chart=feature_chart,
                hist_chart=hist_chart,
                hist_note=hist_note,
            )

        chunk_size = 50_000
        pred_chunks = []
        failed_chunk = False
        for start in range(0, df_x_valid.height, chunk_size):
            df_chunk = df_x_valid.slice(start, chunk_size)
            x_chunk = list_series_to_numpy(df_chunk.select(pl.col(x_col)).to_series(), expected_len=x_dim)
            if x_chunk is None:
                failed_chunk = True
                break
            try:
                pred_chunks.append(artifact_model.predict(x_chunk))
            except Exception as exc:
                _error(f"Model predict failed: {exc}")
                failed_chunk = True
                break
        if failed_chunk or not pred_chunks:
            _error("Unable to build feature matrix for full pool.")
            return TransientOverlayResult(
                df_overlay=_with_overlay_provenance(df_overlay),
                df_pred_scored=df_pred_scored,
                diagnostics=diagnostics,
                feature_chart=feature_chart,
                hist_chart=hist_chart,
                hist_note=hist_note,
            )

        y_pred = np.vstack(pred_chunks)
        if yops_ctx is not None:
            try:
                y_pred = apply_y_ops_inverse(y_ops=y_ops, y=y_pred, ctx=yops_ctx)
            except Exception as exc:
                _warn(f"Y-ops inverse failed: {exc}")
                yops_inverse_ready = False
        elif y_ops:
            yops_inverse_ready = False

        df_pred = df_x_valid.with_columns(pl.Series("opal__overlay__y_vec", y_pred.tolist()))
        if pred_cache is not None and cache_key is not None:
            train_id_cols = [col for col in ["id", "__row_id"] if col in df_train.columns]
            train_ids = df_train.select(train_id_cols) if train_id_cols else df_train.head(0)
            pred_cache[cache_key] = {
                "df_pred": df_pred,
                "yops_inverse_ready": yops_inverse_ready,
                "feature_chart": feature_chart,
                "train_ids": train_ids,
            }
    if df_pred is None or df_pred.is_empty():
        _error("Overlay predictions unavailable; no cached or computed predictions found.")
        return TransientOverlayResult(
            df_overlay=_with_overlay_provenance(df_overlay),
            df_pred_scored=df_pred_scored,
            diagnostics=diagnostics,
            feature_chart=feature_chart,
            hist_chart=hist_chart,
            hist_note=hist_note,
        )

    if y_col is None or y_col not in labels_current_df.columns:
        _error(f"Missing label column `{y_col}` for scoring.")
        return TransientOverlayResult(
            df_overlay=_with_overlay_provenance(df_overlay),
            df_pred_scored=df_pred_scored,
            diagnostics=diagnostics,
            feature_chart=feature_chart,
            hist_chart=hist_chart,
            hist_note=hist_note,
        )
    if labels_current_df.is_empty():
        _error("No labels available for overlay scoring.")
        return TransientOverlayResult(
            df_overlay=_with_overlay_provenance(df_overlay),
            df_pred_scored=df_pred_scored,
            diagnostics=diagnostics,
            feature_chart=feature_chart,
            hist_chart=hist_chart,
            hist_note=hist_note,
        )
    denom_pool = labels_current_df.select(pl.col(y_col).alias("opal__overlay__y_vec"))
    pred_result = None
    if y_ops and not yops_inverse_ready:
        _warn("Y-ops inverse unavailable; SFXI scoring disabled.")
    else:
        if y_ops:
            _note("Y-ops inverse applied; scoring in objective space.")
        try:
            pred_result = compute_sfxi_metrics(
                df=df_pred,
                vec_col="opal__overlay__y_vec",
                params=sfxi_params,
                denom_pool_df=denom_pool,
            )
        except ValueError as exc:
            _error(f"SFXI scoring failed: {exc}")

    if pred_result is None:
        return TransientOverlayResult(
            df_overlay=_with_overlay_provenance(df_overlay),
            df_pred_scored=df_pred_scored,
            diagnostics=diagnostics,
            feature_chart=feature_chart,
            hist_chart=hist_chart,
            hist_note=hist_note,
        )

    df_pred_scored = pred_result.df
    if df_pred_scored.is_empty():
        _warn("No valid predictions after SFXI scoring.")
        return TransientOverlayResult(
            df_overlay=_with_overlay_provenance(df_overlay),
            df_pred_scored=df_pred_scored,
            diagnostics=diagnostics,
            feature_chart=feature_chart,
            hist_chart=hist_chart,
            hist_note=hist_note,
        )

    df_pred_scored = df_pred_scored.with_columns(
        pl.col("logic_fidelity").alias("opal__overlay__logic_fidelity").cast(pl.Float64),
        pl.col("effect_scaled").alias("opal__overlay__effect_scaled").cast(pl.Float64),
        pl.col("score").alias("opal__overlay__score").cast(pl.Float64),
        pl.lit(source_value).alias("opal__overlay__source"),
        pl.lit(campaign_slug).alias("opal__overlay__campaign_slug"),
        pl.lit(run_id_value).alias("opal__overlay__run_id"),
        pl.lit(effective_round).alias("opal__overlay__round"),
        pl.lit(round_mode).alias("opal__overlay__round_mode"),
    )

    sel_params = dict(campaign_info.selection_params or {})
    id_col = "id" if "id" in df_pred_scored.columns else "__row_id"
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
            selection_params=sel_params,
        )
        if warnings:
            _warn("Selection objective warning: " + "; ".join(warnings))
    except Exception as exc:
        _error(f"Selection overlay error: {exc}")
        ranks = np.full(sel_scores.shape, None, dtype=object)
        selected = np.full(sel_scores.shape, False, dtype=bool)
    df_pred_scored = df_pred_scored.with_columns(
        pl.Series("opal__overlay__rank", ranks),
        pl.Series("opal__overlay__top_k", selected),
    )

    pred_cols = [
        "__row_id",
        "opal__overlay__score",
        "opal__overlay__rank",
        "opal__overlay__logic_fidelity",
        "opal__overlay__effect_scaled",
        "opal__overlay__top_k",
        "opal__overlay__source",
        "opal__overlay__campaign_slug",
        "opal__overlay__run_id",
        "opal__overlay__round",
        "opal__overlay__round_mode",
    ]
    if "id" in df_pred_scored.columns:
        pred_cols.append("id")
    overlay_drop_cols = [col for col in pred_cols if col in df_base.columns and col not in {"__row_id", "id"}]
    df_overlay_base = df_base.drop(overlay_drop_cols) if overlay_drop_cols else df_base
    df_overlay = df_overlay_base.join(df_pred_scored.select(pred_cols), on="__row_id", how="left")
    if cache_hit:
        _note(f"Overlay RF done: used cached predictions; predicted `{df_pred_scored.height}` candidates.")
    else:
        _note(f"Overlay RF done: predicted `{df_pred_scored.height}` candidates from artifact model.")

    hist_chart, hist_note = _build_histogram_chart(
        df_pred_scored=df_pred_scored,
        df_sfxi=df_sfxi,
        df_train=df_train,
        dataset_name=dataset_name,
    )

    return TransientOverlayResult(
        df_overlay=df_overlay,
        df_pred_scored=df_pred_scored,
        diagnostics=diagnostics,
        feature_chart=feature_chart,
        hist_chart=hist_chart,
        hist_note=hist_note,
    )
