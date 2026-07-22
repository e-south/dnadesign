"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy/evaluation/response_uncertainty.py

Study target uncertainty over Reader-owned joint bootstrap draws.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd

from dnadesign.opal import ResponseMagnitudeFeasibilityComponents, response_magnitude_feasibility_components

from ..core.contracts import STRESS_STATE_IDS, StressTargetView
from .response_magnitude import build_response_separation_rows

_COMPONENTS = ("response_separation", "on_magnitude_floor", "off_magnitude_ceiling")


class ResponseCalibration(NamedTuple):
    rows: pd.DataFrame
    calibration: pd.DataFrame


def estimate_response_calibration_from_reader_draws(
    labels: pd.DataFrame,
    draws: pd.DataFrame,
    *,
    target_views: tuple[StressTargetView, ...],
    scale_quantile: float,
    expected_bootstrap_samples: int,
) -> ResponseCalibration:
    """Apply study masks to Reader-owned state draws without reopening trajectories."""

    raw_columns = tuple(f"r{state}" for state in STRESS_STATE_IDS) + tuple(f"b{state}" for state in STRESS_STATE_IDS)
    label_required = {
        "id",
        "design_id",
        "reader_experiment_id",
        *raw_columns,
        *(f"r{state}_event_half_range" for state in STRESS_STATE_IDS),
        *(f"b{state}_event_half_range" for state in STRESS_STATE_IDS),
    }
    draw_required = {"id", "draw_index", *raw_columns}
    label_missing = sorted(label_required - set(labels.columns))
    draw_missing = sorted(draw_required - set(draws.columns))
    if label_missing:
        raise ValueError(f"Reader response labels lack uncertainty fields: {label_missing}")
    if draw_missing:
        raise ValueError(f"Reader response draws lack fields: {draw_missing}")
    if expected_bootstrap_samples < 100:
        raise ValueError("Reader uncertainty requires at least 100 bootstrap draws.")
    if not 0.5 <= scale_quantile < 1.0:
        raise ValueError("Reader uncertainty scale_quantile must be in [0.5, 1).")

    base = build_response_separation_rows(labels, target_views=target_views)
    uncertainty = _component_uncertainty_rows(
        labels,
        draws,
        target_views=target_views,
        expected_bootstrap_samples=expected_bootstrap_samples,
        raw_columns=raw_columns,
    )
    rows = base.merge(
        uncertainty,
        on=["id", "selection_view_id", "reader_experiment_id"],
        how="left",
        validate="one_to_one",
    )
    if rows[[f"{component}__bootstrap_sd" for component in _COMPONENTS]].isna().any().any():
        raise ValueError("Reader response labels lack one or more joint bootstrap summaries.")
    for component in _COMPONENTS:
        rows[f"{component}__combined_sd"] = np.hypot(
            rows[f"{component}__bootstrap_sd"].to_numpy(dtype=float),
            rows[f"{component}__event_half_range"].to_numpy(dtype=float),
        )

    calibration = build_calibration_table(
        rows,
        scale_quantile=scale_quantile,
        bootstrap_samples=expected_bootstrap_samples,
    )
    scales = calibration.set_index(["selection_view_id", "component"])["scale"]
    rows["response_constraint_margin"] = [
        float(row.response_separation) / float(scales.loc[(str(row.selection_view_id), "response_separation")])
        for row in rows.itertuples(index=False)
    ]
    rows["on_magnitude_constraint_margin"] = [
        float(row.on_magnitude_floor) / float(scales.loc[(str(row.selection_view_id), "on_magnitude_floor")])
        for row in rows.itertuples(index=False)
    ]
    rows["off_magnitude_constraint_margin"] = [
        -float(row.off_magnitude_ceiling) / float(scales.loc[(str(row.selection_view_id), "off_magnitude_ceiling")])
        for row in rows.itertuples(index=False)
    ]
    rows["feasibility_margin"] = rows[
        ["response_constraint_margin", "on_magnitude_constraint_margin", "off_magnitude_constraint_margin"]
    ].min(axis=1)
    return ResponseCalibration(rows=rows, calibration=calibration)


def _component_uncertainty_rows(
    labels: pd.DataFrame,
    draws: pd.DataFrame,
    *,
    target_views: tuple[StressTargetView, ...],
    expected_bootstrap_samples: int,
    raw_columns: tuple[str, ...],
) -> pd.DataFrame:
    labels_by_id = labels.set_index("id")
    if not labels_by_id.index.is_unique:
        raise ValueError("Reader response labels must have unique ids.")
    records: list[dict[str, object]] = []
    for candidate_id, candidate_draws in draws.groupby("id", sort=True):
        if len(candidate_draws) != expected_bootstrap_samples:
            raise ValueError(
                f"Reader candidate {candidate_id!r} has {len(candidate_draws)} bootstrap draws; "
                f"expected {expected_bootstrap_samples}."
            )
        if candidate_id not in labels_by_id.index:
            raise ValueError(f"Reader bootstrap candidate {candidate_id!r} has no selected label row.")
        label = labels_by_id.loc[candidate_id]
        values = candidate_draws.loc[:, list(raw_columns)].to_numpy(dtype=float)
        for target_view in target_views:
            components = response_magnitude_feasibility_components(values, target_mask=target_view.target_mask)
            records.append(_uncertainty_record(candidate_id, label, target_view, components))
    return pd.DataFrame.from_records(records)


def _uncertainty_record(
    candidate_id: object,
    label: pd.Series,
    target_view: StressTargetView,
    components: ResponseMagnitudeFeasibilityComponents,
) -> dict[str, object]:
    on = np.asarray(target_view.target_mask, dtype=float) == 1.0
    off = ~on
    response_event = np.asarray([float(label[f"r{state}_event_half_range"]) for state in STRESS_STATE_IDS], dtype=float)
    magnitude_event = np.asarray(
        [float(label[f"b{state}_event_half_range"]) for state in STRESS_STATE_IDS], dtype=float
    )
    return {
        "id": str(candidate_id),
        "selection_view_id": target_view.id,
        "reader_experiment_id": str(label["reader_experiment_id"]),
        "response_separation__bootstrap_sd": float(np.std(components.response_separation, ddof=1)),
        "on_magnitude_floor__bootstrap_sd": float(np.std(components.on_magnitude_floor, ddof=1)),
        "off_magnitude_ceiling__bootstrap_sd": float(np.std(components.off_magnitude_ceiling, ddof=1)),
        "response_separation__event_half_range": float(response_event[on].max() + response_event[off].max()),
        "on_magnitude_floor__event_half_range": float(magnitude_event[on].max()),
        "off_magnitude_ceiling__event_half_range": float(magnitude_event[off].max()),
    }


def build_calibration_table(
    rows: pd.DataFrame,
    *,
    scale_quantile: float,
    bootstrap_samples: int,
    exclude_experiment: str | None = None,
) -> pd.DataFrame:
    required = {
        "selection_view_id",
        "reader_experiment_id",
        *(f"{component}__combined_sd" for component in _COMPONENTS),
    }
    missing = sorted(required - set(rows.columns))
    if missing:
        raise ValueError(f"response uncertainty rows are missing columns: {missing}")
    work = rows
    if exclude_experiment is not None:
        work = rows.loc[~rows["reader_experiment_id"].astype(str).eq(str(exclude_experiment))]
    records: list[dict[str, object]] = []
    for selection_view_id, view_rows in work.groupby("selection_view_id", sort=True):
        for component in _COMPONENTS:
            scale = float(view_rows[f"{component}__combined_sd"].quantile(scale_quantile))
            if not np.isfinite(scale) or scale <= 0.0:
                raise ValueError(f"Reader uncertainty produced invalid {selection_view_id}/{component} scale {scale}.")
            records.append(
                {
                    "selection_view_id": str(selection_view_id),
                    "component": component,
                    "threshold": 0.0,
                    "scale": scale,
                    "scale_quantile": scale_quantile,
                    "bootstrap_samples": bootstrap_samples,
                    "excluded_experiment": exclude_experiment,
                    "scale_basis": "reader_joint_bootstrap_plus_conservative_event_bound",
                }
            )
    return pd.DataFrame.from_records(records)


__all__ = [
    "ResponseCalibration",
    "build_calibration_table",
    "estimate_response_calibration_from_reader_draws",
]
