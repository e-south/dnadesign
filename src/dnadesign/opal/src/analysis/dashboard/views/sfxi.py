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
from ...sfxi.gates import nearest_gate
from ..datasets import CampaignInfo
from ..labels import observed_event_ids
from ..util import list_series_to_numpy


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
class Nearest2FactorCounts:
    df: pl.DataFrame
    note: str | None


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
    min_n: int,
    eps: float,
    state_order: Sequence[str] | None,
) -> SFXIParams:
    if state_order is None:
        raise ValueError("state_order is required and must be [00, 10, 01, 11].")
    sfxi_math.assert_state_order(state_order)
    parsed_setpoint = sfxi_math.parse_setpoint_vector({"setpoint_vector": list(setpoint)})
    p0, p1, p2, p3 = (float(x) for x in parsed_setpoint.tolist())
    weights_arr = sfxi_math.weights_from_setpoint(parsed_setpoint, eps=eps)
    weights = tuple(float(x) for x in weights_arr.tolist())
    d = float(sfxi_math.worst_corner_distance(parsed_setpoint))
    if not math.isfinite(d) or d <= 0:
        raise ValueError("setpoint produced invalid distance; check setpoint values.")
    return SFXIParams(
        setpoint=(p0, p1, p2, p3),
        weights=weights,
        state_order=tuple(state_order),
        d=d,
        beta=float(beta),
        gamma=float(gamma),
        delta=float(delta),
        p=float(p),
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
        raise ValueError(f"Missing SFXI vector column: `{vec_col}`.")
    if df.is_empty():
        raise ValueError("No rows available for SFXI metrics.")
    valid_mask = valid_vec8_mask_expr(vec_col)
    invalid_count = int(df.select((~valid_mask).sum()).item())
    if invalid_count:
        raise ValueError(f"SFXI vectors in `{vec_col}` must be length-8 finite values (invalid rows: {invalid_count}).")
    df_valid = df

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
            pool_size=df_valid.height,
            denom_source="disabled",
        )

    if vec_col not in denom_pool_df.columns:
        raise ValueError(f"Denom pool missing vector column: `{vec_col}`.")
    if denom_pool_df.is_empty():
        raise ValueError("Denom pool is empty.")
    pool_invalid = int(denom_pool_df.select((~valid_mask).sum()).item())
    if pool_invalid:
        raise ValueError(
            f"Denom pool vectors in `{vec_col}` must be length-8 finite values (invalid rows: {pool_invalid})."
        )
    pool_vec = list_series_to_numpy(denom_pool_df.get_column(vec_col), expected_len=8)
    if pool_vec is None:
        raise ValueError("Invalid SFXI label vectors: expected length-8 lists of finite values.")
    pool_size = int(pool_vec.shape[0])

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
        denom_source="labels",
    )


def compute_label_sfxi_view(
    *,
    labels_view_df: pl.DataFrame,
    labels_current_df: pl.DataFrame,
    y_col: str | None,
    params: SFXIParams,
) -> LabelSfxiView:
    if y_col is None or y_col not in labels_view_df.columns:
        raise ValueError(f"Missing SFXI label column: `{y_col}`.")
    if labels_view_df.is_empty():
        raise ValueError("No label events available for the selected filters.")
    sfxi_result = compute_sfxi_metrics(
        df=labels_view_df,
        vec_col=y_col,
        params=params,
        denom_pool_df=labels_current_df,
    )
    df_sfxi = sfxi_result.df

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
    return LabelSfxiView(df=df_sfxi, notice=None, table_df=table_df, table_cols=table_cols)


def build_label_sfxi_view(
    *,
    readiness: SFXIReadiness,
    selected_round: int | None,
    labels_view_df: pl.DataFrame,
    labels_current_df: pl.DataFrame,
    params: SFXIParams,
) -> LabelSfxiView:
    if not readiness.ready:
        raise ValueError(readiness.notice or "SFXI disabled.")
    if selected_round is None:
        sfxi_view = compute_label_sfxi_view(
            labels_view_df=labels_view_df,
            labels_current_df=labels_view_df,
            y_col=readiness.y_col,
            params=params,
        )
        return sfxi_view
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
    if pred_df is None or pred_df.is_empty():
        raise ValueError("No stored predictions available.")

    mode_val = str(mode or "canonical").strip().lower()
    if mode_val == "canonical":
        required = ["pred_score", "pred_logic_fidelity", "pred_effect_scaled"]
        missing = [c for c in required if c not in pred_df.columns]
        if missing:
            raise ValueError(f"Prediction history missing required metrics: {sorted(missing)}.")
        df_out = pred_df.with_columns(
            [
                pl.col("pred_logic_fidelity").cast(pl.Float64).alias("logic_fidelity"),
                pl.col("pred_effect_scaled").cast(pl.Float64).alias("effect_scaled"),
                pl.col("pred_score").cast(pl.Float64).alias("score"),
            ]
        )
        invalid_count = int(
            df_out.select(
                (~pl.col("logic_fidelity").is_finite()).sum()
                + (~pl.col("effect_scaled").is_finite()).sum()
                + (~pl.col("score").is_finite()).sum()
            ).item()
        )
        if invalid_count:
            raise ValueError("Prediction metrics must be finite for all rows.")
        return PredSfxiView(df=df_out, notice=None)

    if mode_val != "overlay":
        raise ValueError(f"Unknown SFXI mode: {mode}.")
    if y_hat_col not in pred_df.columns:
        raise ValueError(f"Missing prediction vector column `{y_hat_col}`.")
    if y_col is None or y_col not in labels_current_df.columns:
        raise ValueError(f"Missing label column `{y_col}` for overlay scoring.")
    if labels_current_df.is_empty():
        raise ValueError("No labels available for overlay scoring.")

    vec_col = "__overlay_vec"
    df_pred = pred_df.select([c for c in ["id", "__row_id", y_hat_col] if c in pred_df.columns]).rename(
        {y_hat_col: vec_col}
    )
    denom_pool = labels_current_df.select(pl.col(y_col).alias(vec_col))
    result = compute_sfxi_metrics(df=df_pred, vec_col=vec_col, params=params, denom_pool_df=denom_pool)
    if result.df.is_empty():
        raise ValueError("No valid predictions after overlay scoring.")
    return PredSfxiView(df=result.df, notice=None)


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


def build_nearest_2_factor_counts(
    *,
    label_events_df: pl.DataFrame | None,
    pred_events_df: pl.DataFrame | None,
    y_col: str | None,
    pred_vec_col: str = "pred_y_hat",
) -> Nearest2FactorCounts:
    if y_col is None:
        raise ValueError("Nearest-logic counts require a label vector column.")
    if label_events_df is None or label_events_df.is_empty():
        raise ValueError("Nearest-logic counts require label history.")
    if y_col not in label_events_df.columns:
        raise ValueError(f"Nearest-logic counts require `{y_col}` in label history.")
    if pred_events_df is None or pred_events_df.is_empty():
        raise ValueError("Nearest-logic counts require prediction history.")
    if pred_vec_col not in pred_events_df.columns:
        raise ValueError(f"Nearest-logic counts require `{pred_vec_col}` in prediction history.")

    labels_vec8 = list_series_to_numpy(label_events_df.get_column(y_col), expected_len=8)
    if labels_vec8 is None:
        raise ValueError("Nearest-logic counts require valid label vectors.")
    pred_vec8 = list_series_to_numpy(pred_events_df.get_column(pred_vec_col), expected_len=8)
    if pred_vec8 is None:
        raise ValueError("Nearest-logic counts require valid pred_y_hat vectors.")

    label_classes, _ = nearest_gate(labels_vec8[:, 0:4], state_order=sfxi_math.STATE_ORDER)
    pred_classes, _ = nearest_gate(pred_vec8[:, 0:4], state_order=sfxi_math.STATE_ORDER)
    label_unique, label_counts = np.unique(label_classes, return_counts=True)
    pred_unique, pred_counts = np.unique(pred_classes, return_counts=True)
    label_map = {str(k): int(v) for k, v in zip(label_unique.tolist(), label_counts.tolist())}
    pred_map = {str(k): int(v) for k, v in zip(pred_unique.tolist(), pred_counts.tolist())}
    all_classes = sorted(set(label_map) | set(pred_map))
    rows = []
    for cls in all_classes:
        obs_count = int(label_map.get(cls, 0))
        pred_count = int(pred_map.get(cls, 0))
        if obs_count == 0 and pred_count == 0:
            continue
        rows.append(
            {
                "opal__nearest_2_factor_logic": cls,
                "observed_count": obs_count,
                "predicted_count": pred_count,
            }
        )
    return Nearest2FactorCounts(df=pl.DataFrame(rows), note=None)
