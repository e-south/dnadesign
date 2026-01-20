"""Transient RF overlay helpers for the promoter dashboard."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import altair as alt
import numpy as np
import polars as pl

from ...models.random_forest import RandomForestModel
from .datasets import CampaignInfo
from .diagnostics import Diagnostics
from .labels import infer_round_from_labels
from .models import get_feature_importances, load_round_ctx_from_dir
from .selection import compute_selection_overlay
from .sfxi import SFXIParams, compute_sfxi_metrics, valid_vec8_mask_expr
from .util import dedupe_columns, list_series_to_numpy
from .y_ops import apply_y_ops_fit_transform, apply_y_ops_inverse, build_round_ctx_for_notebook


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

    subtitle = (
        f"{dataset_name or 'dataset'} · round={selected_round} · "
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
    score_col = "opal__transient__score"
    if df_pred_scored.is_empty() or score_col not in df_pred_scored.columns:
        return None, "No transient predictions available."

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
        return chart, "No transient predictions available."

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
    use_artifact: bool,
    artifact_model: Any | None,
    artifact_round_dir: Path | None,
    run_id: str | None,
    rf_random_state: int | None,
    dataset_name: str | None = None,
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
    if effective_round is None:
        effective_round = infer_round_from_labels(labels_current_df) or infer_round_from_labels(labels_asof_df)
    if effective_round is None and context is not None:
        runs_df = getattr(context, "ledger_runs_df", None)
        if runs_df is not None and not runs_df.is_empty() and "as_of_round" in runs_df.columns:
            try:
                effective_round = int(runs_df.select(pl.col("as_of_round").max()).item())
            except Exception:
                effective_round = None
    if effective_round is None:
        _error(
            "Transient overlay requires a round context. Select a round or attach label events "
            "so the round can be inferred."
        )

    run_id_value = run_id if use_artifact else "notebook-transient"
    source_value = "artifact" if use_artifact else "transient"

    def _with_transient_provenance(frame: pl.DataFrame) -> pl.DataFrame:
        return frame.with_columns(
            pl.lit(source_value).alias("opal__transient__source"),
            pl.lit(campaign_slug).alias("opal__transient__campaign_slug"),
            pl.lit(run_id_value).alias("opal__transient__run_id"),
            pl.lit(effective_round).alias("opal__transient__round"),
        )

    if campaign_info is None:
        _warn("Transient predictions unavailable (campaign unsupported).")
        return TransientOverlayResult(
            df_overlay=_with_transient_provenance(df_overlay),
            df_pred_scored=df_pred_scored,
            diagnostics=diagnostics,
            feature_chart=feature_chart,
            hist_chart=hist_chart,
            hist_note=hist_note,
        )

    if x_col is None or x_col not in df_base.columns:
        _warn(f"Missing X column: `{x_col}`.")
        return TransientOverlayResult(
            df_overlay=_with_transient_provenance(df_overlay),
            df_pred_scored=df_pred_scored,
            diagnostics=diagnostics,
            feature_chart=feature_chart,
            hist_chart=hist_chart,
            hist_note=hist_note,
        )
    if labels_asof_df.is_empty():
        _warn("No labels available for transient model training.")
        return TransientOverlayResult(
            df_overlay=_with_transient_provenance(df_overlay),
            df_pred_scored=df_pred_scored,
            diagnostics=diagnostics,
            feature_chart=feature_chart,
            hist_chart=hist_chart,
            hist_note=hist_note,
        )
    if y_col is None or y_col not in labels_asof_df.columns:
        _warn(f"Missing label column `{y_col}` for training.")
        return TransientOverlayResult(
            df_overlay=_with_transient_provenance(df_overlay),
            df_pred_scored=df_pred_scored,
            diagnostics=diagnostics,
            feature_chart=feature_chart,
            hist_chart=hist_chart,
            hist_note=hist_note,
        )

    df_train = labels_asof_df.filter(pl.col(x_col).is_not_null() & pl.col(y_col).is_not_null())
    df_train = df_train.filter(pl.col(x_col).list.len() > 0)
    df_train = df_train.filter(valid_vec8_mask_expr(y_col))
    if df_train.is_empty():
        _warn("No valid training labels after filtering.")
        return TransientOverlayResult(
            df_overlay=_with_transient_provenance(df_overlay),
            df_pred_scored=df_pred_scored,
            diagnostics=diagnostics,
            feature_chart=feature_chart,
            hist_chart=hist_chart,
            hist_note=hist_note,
        )

    len_stats = df_train.select(
        pl.col(x_col).list.len().min().alias("min_len"),
        pl.col(x_col).list.len().max().alias("max_len"),
    )
    min_len, max_len = len_stats.row(0)
    if min_len is None or max_len is None or int(min_len) != int(max_len):
        _warn("X vectors must be fixed-length for transient RF training.")
        return TransientOverlayResult(
            df_overlay=_with_transient_provenance(df_overlay),
            df_pred_scored=df_pred_scored,
            diagnostics=diagnostics,
            feature_chart=feature_chart,
            hist_chart=hist_chart,
            hist_note=hist_note,
        )

    x_dim = int(min_len)
    x_train = list_series_to_numpy(df_train.select(pl.col(x_col)).to_series(), expected_len=x_dim)
    y_train = list_series_to_numpy(df_train.select(pl.col(y_col)).to_series(), expected_len=8)
    if x_train is None or y_train is None or y_train.size == 0:
        _warn("Unable to build training arrays from label data.")
        return TransientOverlayResult(
            df_overlay=_with_transient_provenance(df_overlay),
            df_pred_scored=df_pred_scored,
            diagnostics=diagnostics,
            feature_chart=feature_chart,
            hist_chart=hist_chart,
            hist_note=hist_note,
        )

    y_ops = list(campaign_info.y_ops or [])
    y_train_fit = y_train
    yops_ctx = None
    yops_inverse_ready = True
    model_ready = False

    if use_artifact and artifact_model is not None:
        model = artifact_model
        model_ready = model is not None
        if artifact_round_dir is not None:
            yops_ctx, yops_err = load_round_ctx_from_dir(artifact_round_dir)
            if yops_err:
                _warn(f"Round context load failed: {yops_err}")
        if yops_ctx is None and y_ops:
            _warn("No round_ctx.json available; cannot invert Y-ops for artifact predictions.")
            yops_inverse_ready = False
        if model_ready:
            _note("Using OPAL artifact model for predictions.")
    else:
        yops_ctx = build_round_ctx_for_notebook(
            info=campaign_info,
            run_id="notebook-transient",
            round_index=int(effective_round),
            y_dim=int(y_train.shape[1]),
            n_train=int(df_train.height),
        )
        try:
            y_train_fit = apply_y_ops_fit_transform(y_ops=y_ops, y=y_train, ctx=yops_ctx)
        except Exception as exc:
            _warn(f"Y-ops fit/transform failed: {exc}")
            y_train_fit = y_train
            yops_inverse_ready = False
        _note(f"Transient RF: training started (n_labels={df_train.height}, x_dim={x_train.shape[1]})")
        model_params = dict(campaign_info.model_params or {})
        if rf_random_state is not None:
            model_params["random_state"] = rf_random_state
        _note(f"Transient RF random_state={model_params.get('random_state')}")
        model = RandomForestModel(params=model_params)
        try:
            fit_metrics = model.fit(x_train, y_train_fit)
        except Exception as exc:
            _error(f"Model fit failed: {exc}")
        else:
            model_ready = True
            if fit_metrics is not None:
                _note(f"Transient RF fit metrics: oob_r2={fit_metrics.oob_r2}, oob_mse={fit_metrics.oob_mse}")

    if not model_ready:
        return TransientOverlayResult(
            df_overlay=_with_transient_provenance(df_overlay),
            df_pred_scored=df_pred_scored,
            diagnostics=diagnostics,
            feature_chart=feature_chart,
            hist_chart=hist_chart,
            hist_note=hist_note,
        )

    feature_chart = _build_feature_importance_chart(
        feature_importances=get_feature_importances(model),
        dataset_name=dataset_name,
        selected_round=effective_round,
        n_labels=int(df_train.height),
        x_dim=int(x_train.shape[1]),
        model_params=dict(campaign_info.model_params or {}),
    )

    _note("Transient RF: predicting over full pool ...")
    df_x_all = df_base.select(dedupe_columns(["__row_id", "id", x_col]))
    df_x_valid = df_x_all.filter(pl.col(x_col).is_not_null() & (pl.col(x_col).list.len() == x_dim))
    if df_x_valid.is_empty():
        _warn("No candidate X vectors available for prediction.")
        return TransientOverlayResult(
            df_overlay=_with_transient_provenance(df_overlay),
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
            pred_chunks.append(model.predict(x_chunk))
        except Exception as exc:
            _error(f"Model predict failed: {exc}")
            failed_chunk = True
            break
    if failed_chunk or not pred_chunks:
        _warn("Unable to build feature matrix for full pool.")
        return TransientOverlayResult(
            df_overlay=_with_transient_provenance(df_overlay),
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

    df_pred = df_x_valid.with_columns(pl.Series("opal__transient__y_vec", y_pred.tolist()))
    denom_pool = labels_current_df.select(pl.col(y_col).alias("opal__transient__y_vec"))
    pred_result = None
    if y_ops and not yops_inverse_ready:
        _warn("Y-ops inverse unavailable; SFXI scoring disabled.")
    else:
        if y_ops:
            _note("Y-ops inverse applied; scoring in objective space.")
        try:
            pred_result = compute_sfxi_metrics(
                df=df_pred,
                vec_col="opal__transient__y_vec",
                params=sfxi_params,
                denom_pool_df=denom_pool,
            )
        except ValueError as exc:
            _error(f"SFXI scoring failed: {exc}")

    if pred_result is None:
        return TransientOverlayResult(
            df_overlay=_with_transient_provenance(df_overlay),
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
            df_overlay=_with_transient_provenance(df_overlay),
            df_pred_scored=df_pred_scored,
            diagnostics=diagnostics,
            feature_chart=feature_chart,
            hist_chart=hist_chart,
            hist_note=hist_note,
        )

    df_pred_scored = df_pred_scored.with_columns(
        pl.col("logic_fidelity").alias("opal__transient__logic_fidelity").cast(pl.Float64),
        pl.col("effect_scaled").alias("opal__transient__effect_scaled").cast(pl.Float64),
        pl.col("score").alias("opal__transient__score").cast(pl.Float64),
        pl.lit(source_value).alias("opal__transient__source"),
        pl.lit(campaign_slug).alias("opal__transient__campaign_slug"),
        pl.lit(run_id_value).alias("opal__transient__run_id"),
        pl.lit(effective_round).alias("opal__transient__round"),
    )

    sel_params = dict(campaign_info.selection_params or {})
    id_col = "id" if "id" in df_pred_scored.columns else "__row_id"
    sel_ids = np.asarray(df_pred_scored.get_column(id_col).to_list(), dtype=str)
    sel_scores = (
        df_pred_scored.select(pl.col("opal__transient__score").fill_null(float("nan")).cast(pl.Float64))
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
        pl.Series("opal__transient__rank", ranks),
        pl.Series("opal__transient__top_k", selected),
    )

    pred_cols = [
        "__row_id",
        "opal__transient__score",
        "opal__transient__rank",
        "opal__transient__logic_fidelity",
        "opal__transient__effect_scaled",
        "opal__transient__top_k",
        "opal__transient__source",
        "opal__transient__campaign_slug",
        "opal__transient__run_id",
        "opal__transient__round",
    ]
    if "id" in df_pred_scored.columns:
        pred_cols.append("id")
    overlay_drop_cols = [col for col in pred_cols if col in df_base.columns and col not in {"__row_id", "id"}]
    df_overlay_base = df_base.drop(overlay_drop_cols) if overlay_drop_cols else df_base
    df_overlay = df_overlay_base.join(df_pred_scored.select(pred_cols), on="__row_id", how="left")
    _note(f"Transient RF done: trained on `{df_train.height}` labels; predicted `{df_pred_scored.height}` candidates.")

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
