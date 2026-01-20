"""SFXI scoring utilities."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import polars as pl

from .datasets import CampaignInfo


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
) -> SFXIParams:
    if len(setpoint) != 4:
        raise ValueError("setpoint must have length 4")
    p0, p1, p2, p3 = (float(x) for x in setpoint)
    total = p0 + p1 + p2 + p3
    if total <= eps:
        weights = (0.0, 0.0, 0.0, 0.0)
    else:
        weights = (p0 / total, p1 / total, p2 / total, p3 / total)
    d = math.sqrt(sum(max(v * v, (1.0 - v) * (1.0 - v)) for v in (p0, p1, p2, p3)))
    if d <= 0:
        d = eps
    return SFXIParams(
        setpoint=(p0, p1, p2, p3),
        weights=weights,
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


def _effect_raw_expr(vec_col: str, weights: Sequence[float], delta: float) -> pl.Expr:
    y0 = pl.col(vec_col).list.get(4)
    y1 = pl.col(vec_col).list.get(5)
    y2 = pl.col(vec_col).list.get(6)
    y3 = pl.col(vec_col).list.get(7)
    y0_lin = pl.max_horizontal(pl.lit(0.0), (pl.lit(2.0) ** y0) - delta)
    y1_lin = pl.max_horizontal(pl.lit(0.0), (pl.lit(2.0) ** y1) - delta)
    y2_lin = pl.max_horizontal(pl.lit(0.0), (pl.lit(2.0) ** y2) - delta)
    y3_lin = pl.max_horizontal(pl.lit(0.0), (pl.lit(2.0) ** y3) - delta)
    return weights[0] * y0_lin + weights[1] * y1_lin + weights[2] * y2_lin + weights[3] * y3_lin


def compute_sfxi_metrics(
    *,
    df: pl.DataFrame,
    vec_col: str,
    params: SFXIParams,
    denom_pool_df: pl.DataFrame,
) -> SFXIResult:
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

    p0, p1, p2, p3 = params.setpoint
    setpoint_sum = p0 + p1 + p2 + p3
    intensity_disabled = not math.isfinite(setpoint_sum) or setpoint_sum <= 1.0e-12
    v0 = pl.col(vec_col).list.get(0)
    v1 = pl.col(vec_col).list.get(1)
    v2 = pl.col(vec_col).list.get(2)
    v3 = pl.col(vec_col).list.get(3)
    dist = ((v0 - p0) ** 2 + (v1 - p1) ** 2 + (v2 - p2) ** 2 + (v3 - p3) ** 2) ** 0.5
    logic_fidelity = (1.0 - dist / params.d).clip(0.0, 1.0)

    if intensity_disabled:
        df_sfxi = df_valid.with_columns(
            [
                logic_fidelity.alias("logic_fidelity"),
                pl.lit(0.0).alias("effect_raw"),
                pl.lit(1.0).alias("effect_scaled"),
            ]
        ).with_columns((pl.col("logic_fidelity") ** params.beta).alias("score"))
        return SFXIResult(
            df=df_sfxi,
            denom=1.0,
            weights=params.weights,
            d=params.d,
            pool_size=0,
            denom_source="disabled",
        )

    effect_raw_expr = _effect_raw_expr(vec_col, params.weights, params.delta)

    pool_size = 0
    denom_source = "p"
    if vec_col not in denom_pool_df.columns or denom_pool_df.is_empty():
        raise ValueError(f"Need at least min_n={params.min_n} labels in current round to scale intensity; got 0.")

    pool_dtype = denom_pool_df.schema.get(vec_col, pl.Null)
    if pool_dtype == pl.Null:
        raise ValueError(f"Need at least min_n={params.min_n} labels in current round to scale intensity; got 0.")

    pool_valid = denom_pool_df.filter(valid_vec8_mask_expr(vec_col))
    pool_effect = pool_valid.select(effect_raw_expr.alias("effect_raw"))
    pool_size = pool_effect.height
    if pool_size < params.min_n:
        raise ValueError(
            f"Need at least min_n={params.min_n} labels in current round to scale intensity; got {pool_size}."
        )

    denom = float(pool_effect["effect_raw"].quantile(params.p / 100.0, interpolation="nearest"))
    if not math.isfinite(denom):
        raise ValueError("Invalid denom computed (non-finite). Check labels and scaling config.")
    if denom < 0.0:
        raise ValueError("Invalid denom computed (negative). Check labels and scaling config.")
    if not math.isfinite(params.eps) or params.eps <= 0.0:
        raise ValueError(f"eps must be positive and finite; got {params.eps}.")
    denom = max(denom, params.eps)

    df_sfxi = (
        df_valid.with_columns(
            [
                logic_fidelity.alias("logic_fidelity"),
                effect_raw_expr.alias("effect_raw"),
            ]
        )
        .with_columns(((pl.col("effect_raw") / pl.lit(denom)).clip(0.0, 1.0).alias("effect_scaled")))
        .with_columns(
            ((pl.col("logic_fidelity") ** params.beta) * (pl.col("effect_scaled") ** params.gamma)).alias("score")
        )
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
        empty_df = labels_view_df.head(0)
        return LabelSfxiView(
            df=empty_df,
            notice="No label rounds available for the selected campaign.",
            table_df=empty_df,
            table_cols=[],
        )
    return compute_label_sfxi_view(
        labels_view_df=labels_view_df,
        labels_current_df=labels_current_df,
        y_col=readiness.y_col,
        params=params,
    )
