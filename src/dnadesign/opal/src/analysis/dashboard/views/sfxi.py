"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/analysis/dashboard/views/sfxi.py

Computes SFXI scoring metrics for dashboard views. Provides label/prediction
SFXI view data for charts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import polars as pl

from ....objectives import sfxi_math
from ..datasets import CampaignInfo
from ..labels import observed_event_ids
from ..util import list_series_to_numpy


def fit_intensity_median_iqr(y, *, min_labels: int, eps: float):
    if y.ndim != 2 or y.shape[1] < 8:
        raise ValueError("intensity_median_iqr expects y with shape (n, 8+).")
    if y.shape[0] < int(min_labels):
        return np.zeros(4, dtype=float), np.ones(4, dtype=float), False
    block = y[:, 4:8]
    med = np.median(block, axis=0)
    q75 = np.percentile(block, 75, axis=0)
    q25 = np.percentile(block, 25, axis=0)
    iqr = q75 - q25
    iqr = np.where(iqr <= 0, float(eps), iqr)
    return med, iqr, True


def apply_intensity_median_iqr(y, med, iqr, *, eps: float, enabled: bool):
    if not enabled:
        return y
    out = y.copy()
    out[:, 4:8] = (out[:, 4:8] - med[None, :]) / np.maximum(iqr[None, :], float(eps))
    return out


def invert_intensity_median_iqr(y, med, iqr, *, enabled: bool):
    if not enabled:
        return y
    out = y.copy()
    out[:, 4:8] = out[:, 4:8] * iqr[None, :] + med[None, :]
    return out


@dataclass(frozen=True)
class SFXIParams:
    setpoint: tuple[float, float, float, float]
    weights: tuple[float, float, float, float]
    state_order: tuple[str, str, str, str]
    d: float
    beta: float
    gamma: float
    delta: float
    p: float
    fallback_p: float
    min_n: int
    eps: float


@dataclass(frozen=True)
class SFXIResult:
    df: pl.DataFrame
    denom: float
    weights: tuple[float, float, float, float]
    d: float
    pool_size: int
    denom_source: str


@dataclass(frozen=True)
class LabelSfxiView:
    df: pl.DataFrame
    notice: str | None
    table_df: pl.DataFrame
    table_cols: list[str]


@dataclass(frozen=True)
class PredSfxiView:
    df: pl.DataFrame
    notice: str | None


@dataclass(frozen=True)
class SFXIReadiness:
    ready: bool
    notice: str | None
    x_col: str | None
    y_col: str | None


def resolve_sfxi_readiness(campaign_info: CampaignInfo | None) -> SFXIReadiness:
    if campaign_info is None:
        return SFXIReadiness(
            ready=False,
            notice="Campaign config unavailable; SFXI disabled.",
            x_col=None,
            y_col=None,
        )
    if campaign_info.objective_name != "sfxi_v1":
        return SFXIReadiness(
            ready=False,
            notice=f"Objective `{campaign_info.objective_name}` is not supported here.",
            x_col=campaign_info.x_column,
            y_col=campaign_info.y_column,
        )
    if campaign_info.y_expected_length not in (None, 8):
        return SFXIReadiness(
            ready=False,
            notice="SFXI expects 8-length label vectors; campaign uses a different length.",
            x_col=campaign_info.x_column,
            y_col=campaign_info.y_column,
        )
    return SFXIReadiness(
        ready=True,
        notice=None,
        x_col=campaign_info.x_column,
        y_col=campaign_info.y_column,
    )


def compute_sfxi_params(
    *,
    setpoint: Sequence[float],
    beta: float,
    gamma: float,
    delta: float,
    p: float,
    fallback_p: float,
    min_n: int,
    eps: float,
    state_order: Sequence[str] | None,
) -> SFXIParams:
    if state_order is None:
        raise ValueError("state_order is required and must be [00, 10, 01, 11].")
    sfxi_math.assert_state_order(state_order)
    if len(setpoint) != 4:
        raise ValueError("setpoint must have length 4")
    p0, p1, p2, p3 = (float(x) for x in setpoint)
    weights_arr = sfxi_math.weights_from_setpoint(np.array([p0, p1, p2, p3], dtype=float), eps=eps)
    weights = tuple(float(x) for x in weights_arr.tolist())
    d = float(sfxi_math.worst_corner_distance(np.array([p0, p1, p2, p3], dtype=float)))
    if not math.isfinite(d) or d <= 0:
        d = float(eps)
    return SFXIParams(
        setpoint=(p0, p1, p2, p3),
        weights=weights,
        state_order=tuple(state_order),
        d=d,
        beta=float(beta),
        gamma=float(gamma),
        delta=float(delta),
        p=float(p),
        fallback_p=float(fallback_p),
        min_n=int(min_n),
        eps=float(eps),
    )


def valid_vec8_mask_expr(vec_col: str) -> pl.Expr:
    vec = pl.col(vec_col)
    len_ok = vec.list.len() == 8
    finite_ok = vec.list.eval(pl.element().is_finite()).list.all()
    return vec.is_not_null() & len_ok & finite_ok


def _coerce_vec8_column(df: pl.DataFrame, vec_col: str) -> pl.DataFrame:
    if vec_col not in df.columns or df.is_empty():
        return df
    dtype = df.schema.get(vec_col, pl.Null)
    if dtype == pl.Object:
        return df.with_columns(
            pl.col(vec_col)
            .map_elements(
                lambda v: None if v is None else [float(x) for x in v],
                return_dtype=pl.List(pl.Float64),
            )
            .alias(vec_col)
        )
    inner = getattr(dtype, "inner", None)
    if inner is not None and inner != pl.Float64:
        return df.with_columns(pl.col(vec_col).cast(pl.List(pl.Float64), strict=False))
    return df


def compute_sfxi_metrics(
    *,
    df: pl.DataFrame,
    vec_col: str,
    params: SFXIParams,
    denom_pool_df: pl.DataFrame,
) -> SFXIResult:
    sfxi_math.assert_state_order(params.state_order)
    df = _coerce_vec8_column(df, vec_col)
    denom_pool_df = _coerce_vec8_column(denom_pool_df, vec_col)
    if vec_col not in df.columns:
        return SFXIResult(
            df=df.head(0),
            denom=params.eps,
            weights=params.weights,
            d=params.d,
            pool_size=0,
            denom_source="empty",
        )
    valid_mask = valid_vec8_mask_expr(vec_col)
    df_valid = df.filter(valid_mask)
    if df_valid.is_empty():
        return SFXIResult(
            df=df.head(0),
            denom=params.eps,
            weights=params.weights,
            d=params.d,
            pool_size=0,
            denom_source="empty",
        )

    p0, p1, p2, p3 = params.setpoint
    setpoint_sum = p0 + p1 + p2 + p3
    intensity_disabled = not math.isfinite(setpoint_sum) or setpoint_sum <= 1.0e-12
    vec = list_series_to_numpy(df_valid.get_column(vec_col), expected_len=8)
    if vec is None:
        raise ValueError("Invalid SFXI vectors: expected length-8 lists of finite values.")
    v_hat = np.clip(vec[:, 0:4], 0.0, 1.0)
    y_star = vec[:, 4:8]
    setpoint = np.array([p0, p1, p2, p3], dtype=float)
    F_logic = sfxi_math.logic_fidelity(v_hat, setpoint)

    if intensity_disabled:
        E_raw = np.zeros(v_hat.shape[0], dtype=float)
        E_scaled = np.ones(v_hat.shape[0], dtype=float)
        score = np.power(F_logic, params.beta)
        df_sfxi = df_valid.with_columns(
            [
                pl.Series("logic_fidelity", F_logic),
                pl.Series("effect_raw", E_raw),
                pl.Series("effect_scaled", E_scaled),
                pl.Series("score", score),
            ]
        )
        return SFXIResult(
            df=df_sfxi,
            denom=1.0,
            weights=params.weights,
            d=params.d,
            pool_size=0,
            denom_source="disabled",
        )

    pool_size = 0
    denom_source = "p"
    if vec_col not in denom_pool_df.columns or denom_pool_df.is_empty():
        raise ValueError(f"Need at least min_n={params.min_n} labels in current round to scale intensity; got 0.")

    pool_dtype = denom_pool_df.schema.get(vec_col, pl.Null)
    if pool_dtype == pl.Null:
        raise ValueError(f"Need at least min_n={params.min_n} labels in current round to scale intensity; got 0.")

    pool_valid = denom_pool_df.filter(valid_vec8_mask_expr(vec_col))
    pool_vec = list_series_to_numpy(pool_valid.get_column(vec_col), expected_len=8)
    if pool_vec is None:
        raise ValueError("Invalid SFXI label vectors: expected length-8 lists of finite values.")
    pool_size = int(pool_vec.shape[0])
    if pool_size < params.min_n:
        raise ValueError(
            f"Need at least min_n={params.min_n} labels in current round to scale intensity; got {pool_size}."
        )

    denom = sfxi_math.denom_from_labels(
        pool_vec[:, 4:8],
        setpoint,
        delta=params.delta,
        percentile=int(params.p),
        min_n=int(params.min_n),
        eps=float(params.eps),
        state_order=params.state_order,
    )

    E_raw, _weights = sfxi_math.effect_raw_from_y_star(
        y_star,
        setpoint,
        delta=params.delta,
        eps=params.eps,
        state_order=params.state_order,
    )
    E_scaled = sfxi_math.effect_scaled(E_raw, float(denom))
    score = np.power(F_logic, params.beta) * np.power(E_scaled, params.gamma)

    df_sfxi = df_valid.with_columns(
        [
            pl.Series("logic_fidelity", F_logic),
            pl.Series("effect_raw", E_raw),
            pl.Series("effect_scaled", E_scaled),
            pl.Series("score", score),
        ]
    )

    return SFXIResult(
        df=df_sfxi,
        denom=denom,
        weights=params.weights,
        d=params.d,
        pool_size=pool_size,
        denom_source=denom_source,
    )


def compute_label_sfxi_view(
    *,
    labels_view_df: pl.DataFrame,
    labels_current_df: pl.DataFrame,
    y_col: str | None,
    params: SFXIParams,
) -> LabelSfxiView:
    notice = None
    df_sfxi = labels_view_df.head(0)
    if y_col is None or y_col not in labels_view_df.columns:
        notice = f"Missing SFXI labels: `{y_col}` not found." if y_col else "Missing SFXI labels."
    elif labels_view_df.is_empty():
        notice = "No label events available for the selected filters."
    else:
        try:
            sfxi_result = compute_sfxi_metrics(
                df=labels_view_df,
                vec_col=y_col,
                params=params,
                denom_pool_df=labels_current_df,
            )
        except ValueError as exc:
            notice = str(exc)
        else:
            df_sfxi = sfxi_result.df
            if sfxi_result.denom_source != "disabled" and sfxi_result.pool_size < params.min_n:
                notice = (
                    f"Insufficient labels in current round for scaling; denom source: `{sfxi_result.denom_source}`."
                )
            elif df_sfxi.is_empty():
                notice = "No valid SFXI vectors after filtering."

    table_df = df_sfxi
    display_col = y_col
    if y_col and y_col in df_sfxi.columns:
        preview_col = f"{y_col}_preview"

        def _preview_vec(val) -> str | None:
            if val is None:
                return None
            text = str(val)
            if len(text) > 120:
                return text[:117] + "..."
            return text

        table_df = df_sfxi.with_columns(
            pl.col(y_col).map_elements(_preview_vec, return_dtype=pl.Utf8).alias(preview_col)
        )
        display_col = preview_col

    table_cols = [
        col
        for col in [
            "__row_id",
            "id",
            "observed_round",
            "label_src",
            "logic_fidelity",
            "effect_scaled",
            "score",
            display_col,
        ]
        if col and col in table_df.columns
    ]
    return LabelSfxiView(df=df_sfxi, notice=notice, table_df=table_df, table_cols=table_cols)


def build_label_sfxi_view(
    *,
    readiness: SFXIReadiness,
    selected_round: int | None,
    labels_view_df: pl.DataFrame,
    labels_current_df: pl.DataFrame,
    params: SFXIParams,
) -> LabelSfxiView:
    if not readiness.ready:
        notice = readiness.notice or "SFXI disabled."
        empty_df = labels_view_df.head(0)
        return LabelSfxiView(df=empty_df, notice=notice, table_df=empty_df, table_cols=[])
    if selected_round is None:
        if labels_view_df.is_empty():
            empty_df = labels_view_df.head(0)
            return LabelSfxiView(
                df=empty_df,
                notice="No label events available for the selected filters.",
                table_df=empty_df,
                table_cols=[],
            )
        sfxi_view = compute_label_sfxi_view(
            labels_view_df=labels_view_df,
            labels_current_df=labels_view_df,
            y_col=readiness.y_col,
            params=params,
        )
        if sfxi_view.notice:
            return sfxi_view
        return LabelSfxiView(
            df=sfxi_view.df,
            notice="Using all rounds for SFXI scaling.",
            table_df=sfxi_view.table_df,
            table_cols=sfxi_view.table_cols,
        )
    return compute_label_sfxi_view(
        labels_view_df=labels_view_df,
        labels_current_df=labels_current_df,
        y_col=readiness.y_col,
        params=params,
    )


def build_pred_sfxi_view(
    *,
    pred_df: pl.DataFrame,
    labels_current_df: pl.DataFrame,
    y_col: str | None,
    params: SFXIParams,
    mode: str,
    y_hat_col: str = "pred_y_hat",
) -> PredSfxiView:
    notice = None
    if pred_df is None or pred_df.is_empty():
        return PredSfxiView(df=pl.DataFrame(), notice="No stored predictions available.")

    mode_val = str(mode or "canonical").strip().lower()
    if mode_val == "canonical":
        required = ["pred_score", "pred_logic_fidelity", "pred_effect_scaled"]
        missing = [c for c in required if c not in pred_df.columns]
        if missing:
            return PredSfxiView(
                df=pred_df.head(0),
                notice=f"Prediction history missing required metrics: {sorted(missing)}.",
            )
        df_out = pred_df.with_columns(
            [
                pl.col("pred_logic_fidelity").cast(pl.Float64).alias("logic_fidelity"),
                pl.col("pred_effect_scaled").cast(pl.Float64).alias("effect_scaled"),
                pl.col("pred_score").cast(pl.Float64).alias("score"),
            ]
        )
        mask = (
            pl.col("pred_score").is_not_null()
            & pl.col("pred_logic_fidelity").is_not_null()
            & pl.col("pred_effect_scaled").is_not_null()
        )
        df_out = df_out.filter(mask)
        if df_out.is_empty():
            notice = "No valid prediction metrics available for the selected run."
        return PredSfxiView(df=df_out, notice=notice)

    if y_hat_col not in pred_df.columns:
        return PredSfxiView(
            df=pred_df.head(0),
            notice=f"Missing prediction vector column `{y_hat_col}`.",
        )
    if y_col is None or y_col not in labels_current_df.columns:
        return PredSfxiView(
            df=pred_df.head(0),
            notice=f"Missing label column `{y_col}` for overlay scoring.",
        )
    if labels_current_df.is_empty():
        return PredSfxiView(df=pred_df.head(0), notice="No labels available for overlay scoring.")

    vec_col = "__overlay_vec"
    df_pred = pred_df.select([c for c in ["id", "__row_id", y_hat_col] if c in pred_df.columns]).rename(
        {y_hat_col: vec_col}
    )
    denom_pool = labels_current_df.select(pl.col(y_col).alias(vec_col))
    try:
        result = compute_sfxi_metrics(df=df_pred, vec_col=vec_col, params=params, denom_pool_df=denom_pool)
    except Exception as exc:
        notice = f"SFXI overlay failed: {exc}"
        return PredSfxiView(df=pred_df.head(0), notice=notice)
    if result.df.is_empty():
        notice = "No valid predictions after overlay scoring."
    return PredSfxiView(df=result.df, notice=notice)


def apply_overlay_label_flags(
    *,
    df_overlay: pl.DataFrame,
    labels_view_df: pl.DataFrame,
    df_sfxi: pl.DataFrame,
    label_src: str = "ingest_y",
    id_col: str = "id",
) -> pl.DataFrame:
    if df_overlay.is_empty():
        return df_overlay

    observed_ids = []
    if labels_view_df is not None and not labels_view_df.is_empty():
        observed_ids = observed_event_ids(labels_view_df, label_src=label_src)
    if observed_ids and id_col in df_overlay.columns:
        df_overlay = df_overlay.with_columns(
            pl.col(id_col).cast(pl.Utf8).is_in(observed_ids).alias("opal__overlay__observed_event")
        )
    elif "opal__overlay__observed_event" not in df_overlay.columns:
        df_overlay = df_overlay.with_columns(pl.lit(False).alias("opal__overlay__observed_event"))

    sfxi_scored_col = "opal__overlay__sfxi_scored_label"
    if "__row_id" in df_overlay.columns and "__row_id" in df_sfxi.columns and not df_sfxi.is_empty():
        sfxi_ids = df_sfxi.select(pl.col("__row_id").drop_nulls().unique()).to_series().to_list()
        df_overlay = df_overlay.with_columns(pl.col("__row_id").is_in(sfxi_ids).alias(sfxi_scored_col))
    elif sfxi_scored_col not in df_overlay.columns:
        df_overlay = df_overlay.with_columns(pl.lit(False).alias(sfxi_scored_col))
    return df_overlay
