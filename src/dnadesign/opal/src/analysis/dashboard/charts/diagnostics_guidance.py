"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/analysis/dashboard/charts/diagnostics_guidance.py

Diagnostics chart assembly helpers for the Promoter60 dashboard.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import polars as pl

from ....objectives import sfxi_math
from ...sfxi import setpoint_sweep as sfxi_setpoint_sweep
from ..datasets import CampaignInfo
from ..hues import HueOption
from ..theme import DNAD_DIAGNOSTICS_PLOT_SIZE
from ..util import list_series_to_numpy, safe_is_numeric
from ..views.sfxi import SFXIParams
from .sfxi_diagnostics_altair import (
    make_factorial_effects_chart,
    make_support_diagnostics_chart,
    make_uncertainty_chart,
)
from .sfxi_intensity_scaling import make_intensity_scaling_figure
from .sfxi_setpoint_sweep import make_setpoint_sweep_figure


@dataclass(frozen=True)
class ChartPanel:
    chart: object | None
    note: str | None
    kind: str = "mpl"


@dataclass(frozen=True)
class DiagnosticsPanels:
    factorial: ChartPanel
    support: ChartPanel
    uncertainty: ChartPanel
    sweep: ChartPanel
    intensity: ChartPanel


@dataclass(frozen=True)
class _SweepData:
    sweep_df: pl.DataFrame
    label_effect_raw: np.ndarray
    pool_effect_raw: np.ndarray | None
    subtitle: str | None
    note: str | None


@dataclass(frozen=True)
class ActiveVec8:
    vec8: np.ndarray | None
    source: str | None
    note: str | None


def _note_panel(message: str) -> ChartPanel:
    return ChartPanel(chart=None, note=message, kind="mpl")


def _resolve_join_key(df_left: pl.DataFrame, df_right: pl.DataFrame) -> str | None:
    if "id" in df_left.columns and "id" in df_right.columns:
        return "id"
    if "__row_id" in df_left.columns and "__row_id" in df_right.columns:
        return "__row_id"
    return None


def _has_duplicate_key(df: pl.DataFrame, key: str) -> bool:
    if key not in df.columns or df.is_empty():
        return False
    return bool(df.select(pl.col(key).is_duplicated().any()).item())


def resolve_active_vec8(
    *,
    active_record_id: str | None,
    pred_events_df: pl.DataFrame | None,
    label_events_df: pl.DataFrame | None,
    y_col: str,
    pred_vec_col: str = "pred_y_hat",
) -> ActiveVec8:
    if not active_record_id:
        return ActiveVec8(vec8=None, source=None, note="Active record id missing.")
    active_id = str(active_record_id)

    if pred_events_df is not None and not pred_events_df.is_empty():
        if "id" in pred_events_df.columns and pred_vec_col in pred_events_df.columns:
            pred_rows = pred_events_df.filter(pl.col("id").cast(pl.Utf8) == active_id)
            if not pred_rows.is_empty():
                pred_series = pred_rows.get_column(pred_vec_col).drop_nulls()
                pred_vec = list_series_to_numpy(pred_series, expected_len=8)
                if pred_vec is not None and pred_vec.shape[0] > 0:
                    return ActiveVec8(vec8=pred_vec[0], source="pred_history", note=None)
                return ActiveVec8(vec8=None, source=None, note="Invalid pred_y_hat for active record.")

    if label_events_df is None or label_events_df.is_empty():
        return ActiveVec8(vec8=None, source=None, note="Label history missing.")
    if "id" not in label_events_df.columns or y_col not in label_events_df.columns:
        return ActiveVec8(vec8=None, source=None, note="Label history missing required columns.")
    label_rows = label_events_df.filter(pl.col("id").cast(pl.Utf8) == active_id)
    if label_rows.is_empty():
        return ActiveVec8(vec8=None, source=None, note="Active record missing from label history.")
    sort_cols = []
    if "observed_round" in label_rows.columns:
        sort_cols.append("observed_round")
    if "label_ts" in label_rows.columns:
        sort_cols.append("label_ts")
    if sort_cols:
        label_rows = label_rows.sort(sort_cols, descending=True)
    label_series = label_rows.get_column(y_col).drop_nulls()
    label_vec = list_series_to_numpy(label_series, expected_len=8)
    if label_vec is None or label_vec.shape[0] == 0:
        return ActiveVec8(vec8=None, source=None, note="Invalid label vector for active record.")
    return ActiveVec8(vec8=label_vec[0], source="label_history", note=None)


def _build_diag_view(
    *,
    df_pred_selected: pl.DataFrame | None,
    df_view: pl.DataFrame | None,
    required_cols: Sequence[str],
    ctx: str,
) -> tuple[pl.DataFrame | None, str | None]:
    if df_pred_selected is None or df_pred_selected.is_empty():
        return None, f"{ctx} unavailable (missing predictions)."
    if df_view is None or df_view.is_empty():
        return None, f"{ctx} unavailable (dashboard view missing)."
    join_key = _resolve_join_key(df_pred_selected, df_view)
    if join_key is None:
        return None, f"{ctx} unavailable (missing join key: id or __row_id)."
    if _has_duplicate_key(df_view, join_key):
        return None, f"{ctx} unavailable (join key is not unique in view)."

    required = [col for col in required_cols if col]
    missing = [col for col in required if col not in df_view.columns]
    if missing:
        return None, f"{ctx} unavailable (missing view columns: {', '.join(missing)})."

    df_diag = df_pred_selected.join(
        df_view.select([join_key, *required]),
        on=join_key,
        how="left",
    )
    for col in required:
        dtype = df_diag.schema.get(col, pl.Null)
        if safe_is_numeric(dtype):
            expr = pl.col(col).cast(pl.Float64, strict=False).is_finite().fill_null(False)
            bad = int(df_diag.select((~expr).sum()).item())
            if bad:
                return None, f"{ctx} unavailable ({col} must be finite)."
    return df_diag, None


def _build_factorial_panel(
    *,
    df_pred_selected: pl.DataFrame | None,
    df_view: pl.DataFrame | None,
) -> ChartPanel:
    if df_pred_selected is None or df_pred_selected.is_empty():
        return _note_panel("Factorial effects unavailable (missing predictions).")
    if df_view is None or df_view.is_empty():
        return _note_panel("Factorial effects unavailable (dashboard view missing).")
    if "pred_y_hat" not in df_pred_selected.columns:
        return _note_panel("Factorial effects unavailable (missing pred_y_hat).")
    join_key = _resolve_join_key(df_pred_selected, df_view)
    if join_key is None:
        return _note_panel("Factorial effects unavailable (missing join key: id or __row_id).")
    required_view_cols = ["opal__view__effect_scaled", "opal__view__observed", "opal__view__top_k"]
    missing = [col for col in required_view_cols if col not in df_view.columns]
    if missing:
        return _note_panel(f"Factorial effects unavailable (missing view columns: {', '.join(missing)}).")
    if _has_duplicate_key(df_view, join_key):
        return _note_panel("Factorial effects unavailable (join key is not unique in view).")

    df_plot = df_pred_selected.join(
        df_view.select([join_key, *required_view_cols]),
        on=join_key,
        how="left",
    )
    try:
        chart = make_factorial_effects_chart(
            df_plot,
            logic_col="pred_y_hat",
            size_col="opal__view__effect_scaled",
            label_col="opal__view__observed",
            selected_col="opal__view__top_k",
            subtitle="Predicted logic vectors",
            plot_size=DNAD_DIAGNOSTICS_PLOT_SIZE,
        )
    except Exception as exc:
        return _note_panel(f"Factorial effects unavailable ({exc}).")
    return ChartPanel(chart=chart, note=None, kind="altair")


def _build_support_panel(
    *,
    df_view: pl.DataFrame | None,
    df_pred_selected: pl.DataFrame | None,
    y_col: str,
    hue: HueOption | None,
) -> ChartPanel:
    required_cols = ["opal__sfxi__dist_to_labeled_logic", y_col]
    if hue is not None:
        required_cols.append(hue.key)
    df_support, note = _build_diag_view(
        df_pred_selected=df_pred_selected,
        df_view=df_view,
        required_cols=required_cols,
        ctx="Support diagnostics",
    )
    if note:
        return _note_panel(note)

    try:
        chart = make_support_diagnostics_chart(
            df_support,
            x_col="opal__sfxi__dist_to_labeled_logic",
            y_col=y_col,
            hue=hue,
            label_col="opal__view__observed",
            selected_col="opal__view__top_k",
            subtitle="Logic-space support",
            plot_size=DNAD_DIAGNOSTICS_PLOT_SIZE,
        )
    except Exception as exc:
        return _note_panel(f"Support diagnostics unavailable ({exc}).")
    return ChartPanel(chart=chart, note=None, kind="altair")


def _build_uncertainty_panel(
    *,
    df_view: pl.DataFrame | None,
    df_pred_selected: pl.DataFrame | None,
    uncertainty_available: bool,
    hue: HueOption | None,
) -> ChartPanel:
    if not uncertainty_available:
        return _note_panel("Uncertainty plot unavailable.")
    required_cols = ["opal__sfxi__uncertainty", "opal__view__score"]
    if hue is not None:
        required_cols.append(hue.key)
    df_unc, note = _build_diag_view(
        df_pred_selected=df_pred_selected,
        df_view=df_view,
        required_cols=required_cols,
        ctx="Uncertainty plot",
    )
    if note:
        return _note_panel(note)

    try:
        chart = make_uncertainty_chart(
            df_unc,
            x_col="opal__sfxi__uncertainty",
            y_col="opal__view__score",
            hue=hue,
            label_col="opal__view__observed",
            selected_col="opal__view__top_k",
            subtitle="Uncertainty vs score",
            plot_size=DNAD_DIAGNOSTICS_PLOT_SIZE,
        )
    except Exception as exc:
        return _note_panel(f"Uncertainty plot unavailable ({exc}).")
    return ChartPanel(chart=chart, note=None, kind="altair")


def _build_sweep_data(
    *,
    df_pred_selected: pl.DataFrame | None,
    opal_campaign_info: CampaignInfo | None,
    opal_labels_current_df: pl.DataFrame | None,
    sfxi_params: SFXIParams,
) -> _SweepData:
    labels_df = opal_labels_current_df if opal_labels_current_df is not None else pl.DataFrame()
    if labels_df.is_empty():
        raise ValueError("Setpoint sweep unavailable (current-round labels missing).")

    labels_y_col = None
    if opal_campaign_info is not None and opal_campaign_info.y_column in labels_df.columns:
        labels_y_col = opal_campaign_info.y_column
    elif "y_obs" in labels_df.columns:
        labels_y_col = "y_obs"
    if labels_y_col is None:
        raise ValueError("Setpoint sweep unavailable (label vectors missing).")

    labels_vec8 = list_series_to_numpy(labels_df.get_column(labels_y_col), expected_len=8)
    if labels_vec8 is None:
        raise ValueError("Setpoint sweep unavailable (invalid label vectors).")

    pool_vec = None
    pool_note = None
    if df_pred_selected is not None and not df_pred_selected.is_empty() and "pred_y_hat" in df_pred_selected.columns:
        pool_vec = list_series_to_numpy(df_pred_selected.get_column("pred_y_hat"), expected_len=8)
        if pool_vec is None:
            pool_note = "Pool effect_raw omitted (invalid pred_y_hat vectors)."

    sweep_df = sfxi_setpoint_sweep.sweep_setpoints(
        labels_vec8=labels_vec8,
        current_setpoint=sfxi_params.setpoint,
        percentile=int(sfxi_params.p),
        min_n=int(sfxi_params.min_n),
        eps=float(sfxi_params.eps),
        delta=float(sfxi_params.delta),
        top_k=5,
        tau=0.8,
        pool_vec8=pool_vec,
        state_order=sfxi_math.STATE_ORDER,
    )
    denom_note = f"denom={int(sfxi_params.p)}th pct E_raw (min_n={int(sfxi_params.min_n)})"

    label_effect_raw, _ = sfxi_math.effect_raw_from_y_star(
        labels_vec8[:, 4:8],
        np.array(sfxi_params.setpoint, dtype=float),
        delta=float(sfxi_params.delta),
        eps=float(sfxi_params.eps),
        state_order=sfxi_math.STATE_ORDER,
    )
    pool_effect_raw = None
    if pool_vec is not None:
        pool_effect_raw, _ = sfxi_math.effect_raw_from_y_star(
            pool_vec[:, 4:8],
            np.array(sfxi_params.setpoint, dtype=float),
            delta=float(sfxi_params.delta),
            eps=float(sfxi_params.eps),
            state_order=sfxi_math.STATE_ORDER,
        )

    return _SweepData(
        sweep_df=sweep_df,
        label_effect_raw=label_effect_raw,
        pool_effect_raw=pool_effect_raw,
        subtitle=denom_note,
        note=pool_note,
    )


def build_diagnostics_panels(
    *,
    df_pred_selected: pl.DataFrame | None,
    df_view: pl.DataFrame | None,
    opal_campaign_info: CampaignInfo | None,
    opal_labels_current_df: pl.DataFrame | None,
    sfxi_params: SFXIParams,
    sweep_metrics: Sequence[str],
    support_y_col: str,
    support_color: HueOption | None,
    uncertainty_color: HueOption | None,
    uncertainty_available: bool,
) -> DiagnosticsPanels:
    factorial = _build_factorial_panel(
        df_pred_selected=df_pred_selected,
        df_view=df_view,
    )
    support = _build_support_panel(
        df_view=df_view,
        df_pred_selected=df_pred_selected,
        y_col=support_y_col,
        hue=support_color,
    )
    uncertainty = _build_uncertainty_panel(
        df_view=df_view,
        df_pred_selected=df_pred_selected,
        uncertainty_available=uncertainty_available,
        hue=uncertainty_color,
    )

    try:
        sweep_data = _build_sweep_data(
            df_pred_selected=df_pred_selected,
            opal_campaign_info=opal_campaign_info,
            opal_labels_current_df=opal_labels_current_df,
            sfxi_params=sfxi_params,
        )
    except Exception as exc:
        sweep = _note_panel(str(exc))
        intensity = _note_panel(str(exc))
    else:
        try:
            fig = make_setpoint_sweep_figure(
                sweep_data.sweep_df,
                metrics=list(sweep_metrics),
                subtitle=f"labels={sweep_data.label_effect_raw.shape[0]} · {sweep_data.subtitle}",
            )
        except Exception as exc:
            sweep = _note_panel(f"Setpoint sweep unavailable ({exc}).")
        else:
            sweep = ChartPanel(chart=fig, note=sweep_data.note, kind="mpl")

        try:
            fig = make_intensity_scaling_figure(
                sweep_data.sweep_df,
                label_effect_raw=sweep_data.label_effect_raw,
                pool_effect_raw=sweep_data.pool_effect_raw,
                subtitle=sweep_data.subtitle,
            )
        except Exception as exc:
            intensity = _note_panel(f"Intensity scaling unavailable ({exc}).")
        else:
            intensity = ChartPanel(chart=fig, note=sweep_data.note, kind="mpl")

    return DiagnosticsPanels(
        factorial=factorial,
        support=support,
        uncertainty=uncertainty,
        sweep=sweep,
        intensity=intensity,
    )
