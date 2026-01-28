"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/analysis/dashboard/views/derived_metrics.py

Builds derived dashboard metrics that join prediction history with view data.
Includes nearest logic class, logic-space support distance, and uncertainty.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from ....objectives import sfxi_math
from ...sfxi.gates import nearest_gate
from ...sfxi.support import dist_to_labeled_logic
from ...sfxi.uncertainty import UncertaintyContext, compute_uncertainty, supports_uncertainty
from ..datasets import CampaignInfo
from ..models import load_round_ctx_from_dir, resolve_artifact_state
from ..util import list_series_to_numpy
from .sfxi import SFXIParams


@dataclass(frozen=True)
class DerivedMetricsResult:
    df: pl.DataFrame
    uncertainty_available: bool


def _is_unique(df: pl.DataFrame, key: str) -> bool:
    if key not in df.columns or df.is_empty():
        return False
    return not bool(df.select(pl.col(key).is_duplicated().any()).item())


def _resolve_join_key(df_left: pl.DataFrame, df_right: pl.DataFrame) -> str:
    if (
        "id" in df_left.columns
        and "id" in df_right.columns
        and _is_unique(df_left, "id")
        and _is_unique(df_right, "id")
    ):
        return "id"
    if (
        "__row_id" in df_left.columns
        and "__row_id" in df_right.columns
        and _is_unique(df_left, "__row_id")
        and _is_unique(df_right, "__row_id")
    ):
        return "__row_id"
    raise ValueError("Derived metrics require a unique join key (`id` or `__row_id`) in both frames.")


def _require_vec8(df: pl.DataFrame, col: str, *, label: str) -> np.ndarray:
    if col not in df.columns:
        raise ValueError(f"{label} requires `{col}`.")
    if df.is_empty():
        raise ValueError(f"{label} requires non-empty data.")
    vec = list_series_to_numpy(df.get_column(col), expected_len=8)
    if vec is None:
        raise ValueError(f"{label} requires length-8 finite vectors in `{col}`.")
    return vec


def attach_diagnostics_metrics(
    *,
    df_view: pl.DataFrame,
    df_pred_selected: pl.DataFrame,
    labels_asof_df: pl.DataFrame,
    labels_current_df: pl.DataFrame,
    campaign_info: CampaignInfo,
    pred_selected_round: int,
    pred_selected_run_id: str | None,
    sfxi_params: SFXIParams,
    pred_vec_col: str = "pred_y_hat",
) -> DerivedMetricsResult:
    if df_view.is_empty():
        raise ValueError("Dashboard view is empty; cannot attach derived metrics.")
    if df_pred_selected.is_empty():
        raise ValueError("Derived metrics require prediction history for the selected run.")

    join_key = _resolve_join_key(df_pred_selected, df_view)
    pred_vec8 = _require_vec8(df_pred_selected, pred_vec_col, label="Derived metrics")

    if "opal__view__observed" not in df_view.columns:
        raise ValueError("Derived metrics require `opal__view__observed` in view.")
    if "id" not in df_view.columns:
        raise ValueError("Derived metrics require `id` in view for observed logic mapping.")

    gate_cls_pred, _ = nearest_gate(pred_vec8[:, 0:4], state_order=sfxi_math.STATE_ORDER)
    gate_cls_pred = np.asarray([str(code).zfill(4) for code in gate_cls_pred], dtype=object)
    gate_pred_df = df_pred_selected.select([join_key]).with_columns(
        pl.Series("opal__nearest_2_factor_logic_pred", gate_cls_pred, dtype=pl.Utf8),
    )
    df_view = df_view.join(gate_pred_df, on=join_key, how="left")

    if labels_asof_df.is_empty():
        raise ValueError("Derived metrics require labels-as-of data.")
    if "id" not in labels_asof_df.columns:
        raise ValueError("Derived metrics require `id` in labels-as-of for observed logic mapping.")
    if not _is_unique(labels_asof_df, "id"):
        raise ValueError("Derived metrics require unique `id` in labels-as-of for observed logic mapping.")
    labels_y_col = campaign_info.y_column if campaign_info.y_column in labels_asof_df.columns else None
    if labels_y_col is None:
        raise ValueError("Derived metrics require label vectors in labels-as-of.")
    label_vec8 = _require_vec8(labels_asof_df, labels_y_col, label="Derived metrics (observed logic)")
    gate_cls_obs, _ = nearest_gate(label_vec8[:, 0:4], state_order=sfxi_math.STATE_ORDER)
    gate_cls_obs = np.asarray([str(code).zfill(4) for code in gate_cls_obs], dtype=object)
    gate_obs_df = labels_asof_df.select([pl.col("id").cast(pl.Utf8)]).with_columns(
        pl.Series("opal__nearest_2_factor_logic_obs", gate_cls_obs, dtype=pl.Utf8),
    )
    df_view = df_view.join(gate_obs_df, on="id", how="left")
    df_view = df_view.with_columns(
        pl.when(pl.col("opal__view__observed").fill_null(False))
        .then(pl.col("opal__nearest_2_factor_logic_obs"))
        .otherwise(pl.col("opal__nearest_2_factor_logic_pred"))
        .alias("opal__nearest_2_factor_logic")
    )
    missing_gate = int(df_view.select(pl.col("opal__nearest_2_factor_logic").is_null().sum()).item())
    if missing_gate:
        raise ValueError(
            "Derived metrics require nearest 2-factor logic for all rows; "
            f"missing for {missing_gate} rows. Ensure predictions cover all rows "
            "and labels-as-of cover observed rows."
        )
    df_view = df_view.drop(["opal__nearest_2_factor_logic_pred", "opal__nearest_2_factor_logic_obs"])

    if labels_asof_df.is_empty():
        raise ValueError("dist_to_labeled_logic requires labels-as-of data.")
    if labels_y_col is None:
        raise ValueError("dist_to_labeled_logic requires label vectors in labels-as-of.")
    dists = dist_to_labeled_logic(
        pred_vec8[:, 0:4],
        label_vec8[:, 0:4],
        state_order=sfxi_math.STATE_ORDER,
    )
    dist_df = df_pred_selected.select([join_key]).with_columns(
        pl.Series("opal__sfxi__dist_to_labeled_logic", dists),
    )
    df_view = df_view.join(dist_df, on=join_key, how="left")

    if campaign_info.x_column is None or campaign_info.x_column not in df_view.columns:
        raise ValueError("Uncertainty requires the campaign x_column in the dashboard view.")
    if pred_selected_round is None:
        raise ValueError("Uncertainty requires a selected round.")

    artifact_state = resolve_artifact_state(
        campaign_info=campaign_info,
        as_of_round=pred_selected_round,
        run_id=pred_selected_run_id,
    )
    if artifact_state is None or not artifact_state.use_artifact or artifact_state.model is None:
        raise ValueError("Uncertainty requires an artifact model for the selected run.")
    if not supports_uncertainty(model=artifact_state.model):
        raise ValueError("Uncertainty requires a model that supports ensemble predictions.")

    X = list_series_to_numpy(df_view.get_column(campaign_info.x_column), expected_len=None)
    if X is None:
        raise ValueError("Uncertainty requires finite X vectors for all records.")

    w = sfxi_math.weights_from_setpoint(np.array(sfxi_params.setpoint, dtype=float), eps=1e-12)
    intensity_disabled = not np.any(w)
    denom = None
    if not intensity_disabled:
        if labels_current_df.is_empty():
            raise ValueError("Uncertainty requires current-round labels to compute denom.")
        if campaign_info.y_column not in labels_current_df.columns:
            raise ValueError("Uncertainty requires label vectors in current-round labels.")
        labels_vec8 = _require_vec8(labels_current_df, campaign_info.y_column, label="Uncertainty denom")
        denom = sfxi_math.denom_from_labels(
            labels_vec8[:, 4:8],
            np.array(sfxi_params.setpoint, dtype=float),
            delta=float(sfxi_params.delta),
            percentile=int(sfxi_params.p),
            min_n=int(sfxi_params.min_n),
            eps=float(sfxi_params.eps),
            state_order=sfxi_math.STATE_ORDER,
        )

    round_ctx = None
    if campaign_info.y_ops:
        if artifact_state.round_dir is None:
            raise ValueError("Uncertainty requires round_ctx to invert y-ops.")
        round_ctx, ctx_err = load_round_ctx_from_dir(artifact_state.round_dir)
        if round_ctx is None:
            raise ValueError(f"Uncertainty requires round_ctx ({ctx_err}).")

    ctx = UncertaintyContext(
        setpoint=np.array(sfxi_params.setpoint, dtype=float),
        beta=float(sfxi_params.beta),
        gamma=float(sfxi_params.gamma),
        delta=float(sfxi_params.delta),
        denom=denom,
        y_ops=campaign_info.y_ops or [],
        round_ctx=round_ctx,
    )
    result = compute_uncertainty(
        artifact_state.model,
        X,
        ctx=ctx,
        batch_size=2048,
    )
    uncertainty_join_key = "id" if "id" in df_view.columns else "__row_id"
    uncertainty_df = df_view.select([uncertainty_join_key]).with_columns(
        pl.Series("opal__sfxi__uncertainty", result.values),
    )
    df_view = df_view.join(uncertainty_df, on=uncertainty_join_key, how="left")

    return DerivedMetricsResult(df=df_view, uncertainty_available=True)
