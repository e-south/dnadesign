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
from ..util import list_series_to_numpy
from ..views.sfxi import SFXIParams
from .sfxi_diagnostics_altair import (
    make_factorial_effects_chart,
    make_support_diagnostics_chart,
    make_uncertainty_chart,
)
from .sfxi_intensity_scaling import make_intensity_scaling_figure
from .sfxi_setpoint_decomposition import make_setpoint_decomposition_figure
from .sfxi_setpoint_sweep import make_setpoint_sweep_figure


@dataclass(frozen=True)
class ChartPanel:
    chart: object | None
    note: str | None
    kind: str = "mpl"


@dataclass(frozen=True)
class DiagnosticsPanels:
    sample_note: str | None
    factorial: ChartPanel
    decomposition: ChartPanel
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
class ResolvedVec8:
    vec8: np.ndarray | None
    source: str | None
    note: str | None


def resolve_active_vec8(
    *,
    active_record_id: str | None,
    pred_events_df: pl.DataFrame | None,
    label_events_df: pl.DataFrame | None,
    y_col: str | None,
) -> ResolvedVec8:
    if not active_record_id:
        return ResolvedVec8(vec8=None, source=None, note="Select an active record to view setpoint decomposition.")
    if pred_events_df is not None and not pred_events_df.is_empty():
        if "id" not in pred_events_df.columns or "pred_y_hat" not in pred_events_df.columns:
            return ResolvedVec8(
                vec8=None,
                source="pred_history",
                note="Prediction history missing required columns for decomposition.",
            )
        df_pred = pred_events_df.filter(pl.col("id") == str(active_record_id))
        if not df_pred.is_empty():
            sort_cols = [col for col in ["as_of_round", "pred_ts"] if col in df_pred.columns]
            if sort_cols:
                df_pred = df_pred.sort(sort_cols)
            pred_vec = list_series_to_numpy(df_pred.get_column("pred_y_hat"), expected_len=8)
            if pred_vec is None:
                return ResolvedVec8(
                    vec8=None,
                    source="pred_history",
                    note="Prediction history vec8 is invalid for active record.",
                )
            return ResolvedVec8(vec8=pred_vec[-1], source="pred_history", note="Source: prediction history.")
    if label_events_df is None or label_events_df.is_empty():
        return ResolvedVec8(vec8=None, source=None, note="No label history available for active record.")
    if "id" not in label_events_df.columns:
        return ResolvedVec8(vec8=None, source="label_history", note="Label history missing id column.")
    if not y_col or y_col not in label_events_df.columns:
        return ResolvedVec8(vec8=None, source="label_history", note="Label history missing label vector column.")
    df_labels = label_events_df.filter(pl.col("id") == str(active_record_id))
    if df_labels.is_empty():
        return ResolvedVec8(vec8=None, source="label_history", note="No label history for active record.")
    sort_cols = [col for col in ["observed_round", "label_ts"] if col in df_labels.columns]
    if sort_cols:
        df_labels = df_labels.sort(sort_cols)
    label_vec = list_series_to_numpy(df_labels.get_column(y_col), expected_len=8)
    if label_vec is None:
        return ResolvedVec8(vec8=None, source="label_history", note="Label history vec8 is invalid for active record.")
    return ResolvedVec8(
        vec8=label_vec[-1],
        source="label_history",
        note="Source: label history (no pred history for active record).",
    )


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


def _sample_df(df: pl.DataFrame, *, sample_n: int, seed: int) -> pl.DataFrame:
    if sample_n <= 0:
        return df
    if df.height <= sample_n:
        return df
    return df.sample(n=int(sample_n), seed=int(seed), shuffle=True)


def _build_factorial_panel(
    *,
    df_pred_selected: pl.DataFrame | None,
    df_view: pl.DataFrame | None,
    sample_n: int,
    seed: int,
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
    df_plot = _sample_df(df_plot, sample_n=sample_n, seed=seed)
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


def _build_decomposition_panel(
    *,
    active_record_id: str | None,
    pred_events_df: pl.DataFrame | None,
    label_events_df: pl.DataFrame | None,
    y_col: str | None,
    sfxi_params: SFXIParams,
) -> ChartPanel:
    resolved = resolve_active_vec8(
        active_record_id=active_record_id,
        pred_events_df=pred_events_df,
        label_events_df=label_events_df,
        y_col=y_col,
    )
    if resolved.vec8 is None:
        return _note_panel(resolved.note or "Setpoint decomposition unavailable (missing vec8).")
    if resolved.vec8.shape[0] < 8:
        return _note_panel("Setpoint decomposition unavailable (vec8 too short).")

    y_hat = resolved.vec8
    try:
        fig = make_setpoint_decomposition_figure(
            v_hat=y_hat[0:4],
            y_star=y_hat[4:8],
            setpoint=np.array(sfxi_params.setpoint, dtype=float),
            delta=float(sfxi_params.delta),
            subtitle=f"id={active_record_id or 'n/a'} · {resolved.source or 'unknown'}",
        )
    except Exception as exc:
        return _note_panel(f"Setpoint decomposition unavailable ({exc}).")
    return ChartPanel(chart=fig, note=resolved.note, kind="mpl")


def _build_support_panel(
    *,
    df_view: pl.DataFrame | None,
    sample_n: int,
    seed: int,
    y_col: str,
    hue: HueOption | None,
) -> ChartPanel:
    if df_view is None or df_view.is_empty():
        return _note_panel("Support diagnostics unavailable (dashboard view missing).")
    if "opal__sfxi__dist_to_labeled_logic" not in df_view.columns:
        return _note_panel("Support diagnostics unavailable (dist_to_labeled_logic missing).")
    if y_col not in df_view.columns:
        return _note_panel(f"Support diagnostics unavailable (missing y column: {y_col}).")
    if hue is not None and hue.key not in df_view.columns:
        return _note_panel(f"Support diagnostics unavailable (missing hue column: {hue.key}).")

    df_support = _sample_df(df_view, sample_n=sample_n, seed=seed)
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
    uncertainty_available: bool,
    sample_n: int,
    seed: int,
    hue: HueOption | None,
) -> ChartPanel:
    if not uncertainty_available:
        return _note_panel("Uncertainty plot unavailable.")
    if df_view is None or df_view.is_empty():
        return _note_panel("Uncertainty plot unavailable (dashboard view missing).")
    if "opal__sfxi__uncertainty" not in df_view.columns:
        return _note_panel("Uncertainty plot unavailable (missing uncertainty column).")
    if hue is not None and hue.key not in df_view.columns:
        return _note_panel(f"Uncertainty plot unavailable (missing hue column: {hue.key}).")

    df_unc = df_view.filter(pl.col("opal__sfxi__uncertainty").is_not_null())
    df_unc = _sample_df(df_unc, sample_n=sample_n, seed=seed)
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
    active_record_id: str | None,
    opal_campaign_info: CampaignInfo | None,
    label_events_df: pl.DataFrame | None,
    opal_labels_current_df: pl.DataFrame | None,
    sfxi_params: SFXIParams,
    sweep_metrics: Sequence[str],
    support_y_col: str,
    support_color: HueOption | None,
    uncertainty_color: HueOption | None,
    uncertainty_available: bool,
    sample_n: int,
    seed: int,
) -> DiagnosticsPanels:
    sample_note = None
    if sample_n > 0:
        sample_note = f"Displaying up to {int(sample_n)} sampled points (seed={int(seed)})."

    factorial = _build_factorial_panel(
        df_pred_selected=df_pred_selected,
        df_view=df_view,
        sample_n=sample_n,
        seed=seed,
    )
    decomposition = _build_decomposition_panel(
        active_record_id=active_record_id,
        pred_events_df=df_pred_selected,
        label_events_df=label_events_df,
        y_col=opal_campaign_info.y_column if opal_campaign_info is not None else None,
        sfxi_params=sfxi_params,
    )
    support = _build_support_panel(
        df_view=df_view,
        sample_n=sample_n,
        seed=seed,
        y_col=support_y_col,
        hue=support_color,
    )
    uncertainty = _build_uncertainty_panel(
        df_view=df_view,
        uncertainty_available=uncertainty_available,
        sample_n=sample_n,
        seed=seed,
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
        sample_note=sample_note,
        factorial=factorial,
        decomposition=decomposition,
        support=support,
        uncertainty=uncertainty,
        sweep=sweep,
        intensity=intensity,
    )
