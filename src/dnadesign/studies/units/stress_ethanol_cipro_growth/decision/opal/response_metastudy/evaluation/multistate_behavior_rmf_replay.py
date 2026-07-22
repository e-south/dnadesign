"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy/evaluation/multistate_behavior_rmf_replay.py

Fair RMF replay on corrected Reader calibration and fixed raw predictions.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from dnadesign.opal import score_response_magnitude_feasibility

from ..core.contracts import StressTargetView
from .multistate_behavior_cohort import behavior_component_columns
from .multistate_behavior_protocol import MultistateBehaviorShadowProtocol

_EVIDENCE_ROLE = "corrected_reader_calibration_replay_same_fixed_raw_prediction_matrix"


def build_current_rmf_prediction_scores(
    *,
    predictions: pd.DataFrame,
    calibration: pd.DataFrame,
    protocol: MultistateBehaviorShadowProtocol,
    target_views: tuple[StressTargetView, ...],
) -> pd.DataFrame:
    """Replay RMF instead of comparing against stale persisted objective scores."""

    components = behavior_component_columns(protocol)
    required = {"id", "prediction_run_id", "prediction_source_sha256", *components}
    if missing := sorted(required - set(predictions.columns)):
        raise ValueError(f"RMF replay predictions lack fields: {missing}")
    rows = predictions.loc[:, ["id", "prediction_run_id", "prediction_source_sha256", *components]].copy()
    if rows.empty or rows["id"].astype(str).duplicated().any():
        raise ValueError("RMF replay requires unique fixed-prediction candidate IDs.")
    values = rows.loc[:, list(components)].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("RMF replay raw predictions must be finite.")
    records: list[pd.DataFrame] = []
    for view in target_views:
        view_calibration = _view_calibration(calibration, view_id=view.id)
        score = score_response_magnitude_feasibility(
            values,
            target_mask=view.target_mask,
            calibration=view_calibration,
        )
        frame = rows.loc[:, ["id", "prediction_run_id", "prediction_source_sha256"]].copy()
        frame["selection_view_id"] = view.id
        frame["hard_score"] = score.feasibility_margin
        records.append(frame)
    return pd.concat(records, ignore_index=True)


def bind_current_rmf_calibration(
    calibration: pd.DataFrame,
    *,
    reader_bundle_manifest_sha256: str,
    normalization_source_rows_sha256: str,
) -> pd.DataFrame:
    """Attach the corrected source receipt to the persisted RMF replay scales."""

    required = {
        "selection_view_id",
        "component",
        "threshold",
        "scale",
        "scale_quantile",
        "bootstrap_samples",
        "excluded_experiment",
        "scale_basis",
    }
    if set(calibration.columns) != required:
        raise ValueError("RMF replay calibration columns are incomplete or unexpected.")
    rows = calibration.loc[:, sorted(required)].copy()
    rows["reader_bundle_manifest_sha256"] = _digest(reader_bundle_manifest_sha256)
    rows["normalization_source_rows_sha256"] = _digest(normalization_source_rows_sha256)
    rows["evidence_role"] = _EVIDENCE_ROLE
    return (
        rows.loc[
            :,
            [
                "selection_view_id",
                "component",
                "threshold",
                "scale",
                "scale_quantile",
                "bootstrap_samples",
                "excluded_experiment",
                "scale_basis",
                "reader_bundle_manifest_sha256",
                "normalization_source_rows_sha256",
                "evidence_role",
            ],
        ]
        .sort_values(["selection_view_id", "component"], kind="mergesort")
        .reset_index(drop=True)
    )


def _view_calibration(frame: pd.DataFrame, *, view_id: str) -> dict[str, float]:
    rows = frame.loc[frame["selection_view_id"].astype(str).eq(view_id)].set_index("component")
    expected = {"response_separation", "on_magnitude_floor", "off_magnitude_ceiling"}
    if set(rows.index.astype(str)) != expected or not rows.index.is_unique:
        raise ValueError(f"RMF replay calibration coverage drifted for {view_id!r}.")
    return {
        "response_separation_min": 0.0,
        "on_magnitude_min": 0.0,
        "off_magnitude_max": 0.0,
        "response_separation_scale": float(rows.loc["response_separation", "scale"]),
        "on_magnitude_scale": float(rows.loc["on_magnitude_floor", "scale"]),
        "off_magnitude_scale": float(rows.loc["off_magnitude_ceiling", "scale"]),
    }


def _digest(value: str) -> str:
    digest = str(value).removeprefix("sha256:")
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError("RMF replay provenance must use a lowercase SHA-256 digest.")
    return "sha256:" + digest


__all__ = ["bind_current_rmf_calibration", "build_current_rmf_prediction_scores"]
