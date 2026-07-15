"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/response_window_observations/artifact_validation.py

Cross-record scientific validation for observation bundles.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .artifact_contract import ResponseWindowObservationArtifactError
from .artifact_manifest import is_sha256
from .artifact_recomputation import validate_recomputed_observations
from .artifact_repeat_validation import validate_repeat_records
from .artifact_uncertainty_validation import validate_uncertainty_records
from .contracts import CANDIDATE_METADATA_COLUMNS, VALUE_COLUMNS, ResponseWindowObservationPreview


def validate_frames(preview: ResponseWindowObservationPreview, *, bootstrap_samples: int) -> None:
    observations = preview.observations
    required_observations = {
        "candidate_id",
        "reader_design_ids",
        "experiment_count",
        "aggregation_method",
        *CANDIDATE_METADATA_COLUMNS,
        *VALUE_COLUMNS,
    }
    missing = sorted(required_observations - set(observations.columns))
    if observations.empty or missing:
        raise ResponseWindowObservationArtifactError(
            f"observation rows are empty or missing required columns: {missing}"
        )
    if observations["candidate_id"].duplicated().any():
        raise ResponseWindowObservationArtifactError("observation candidate IDs must be unique.")
    _finite_values(observations, label="observations")
    if not observations["sequence_sha256"].map(is_sha256).all():
        raise ResponseWindowObservationArtifactError("observation sequence digests are invalid.")
    candidate_ids = set(observations["candidate_id"].astype(str))
    _validate_contributions(preview.contributions, candidate_ids=candidate_ids)
    _validate_draws(preview.bootstrap_draws, candidate_ids=candidate_ids, samples=bootstrap_samples)
    expected_pairs = {(candidate_id, component) for candidate_id in candidate_ids for component in VALUE_COLUMNS}
    for label, frame in (
        ("uncertainty", preview.uncertainty),
        ("event-time sensitivity", preview.event_time_sensitivity),
    ):
        if not {"candidate_id", "component"}.issubset(frame.columns):
            raise ResponseWindowObservationArtifactError(f"{label} rows lack candidate/component identity.")
        observed_pairs = set(frame[["candidate_id", "component"]].astype(str).itertuples(index=False, name=None))
        if observed_pairs != expected_pairs or frame.duplicated(subset=["candidate_id", "component"]).any():
            raise ResponseWindowObservationArtifactError(f"{label} coverage disagrees with observations.")
    if set(preview.reduction_sensitivity["candidate_id"].astype(str)) != candidate_ids:
        raise ResponseWindowObservationArtifactError("reduction-sensitivity coverage disagrees with observations.")
    validate_uncertainty_records(
        preview.uncertainty,
        observations=observations,
        bootstrap_samples=bootstrap_samples,
    )
    validate_repeat_records(preview.repeat_diagnostics, contributions=preview.contributions)
    validate_recomputed_observations(observations, preview.contributions, preview.uncertainty)


def _validate_contributions(frame: pd.DataFrame, *, candidate_ids: set[str]) -> None:
    required = {
        "candidate_id",
        "design_id",
        "reader_experiment_id",
        "reduction_id",
        "repeat_decision",
        "repeat_decision_reason",
        "repeat_classification",
        "repeat_evidence_artifact",
        "repeat_evidence_sha256",
        "repeat_adjudicated_by",
        "repeat_adjudicated_at",
        "included_in_label",
        "experiment_weight",
        *VALUE_COLUMNS,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ResponseWindowObservationArtifactError(f"observation contributions disagree: missing={missing}")
    if not frame["included_in_label"].map(lambda value: isinstance(value, (bool, np.bool_))).all():
        raise ResponseWindowObservationArtifactError("contribution inclusion flags must be boolean.")
    inclusion = frame.groupby("candidate_id")["included_in_label"].agg(lambda values: set(map(bool, values)))
    if inclusion.map(len).gt(1).any():
        raise ResponseWindowObservationArtifactError("candidate contributions mix inclusion decisions.")
    included_ids = {str(candidate_id) for candidate_id, values in inclusion.items() if True in values}
    if included_ids != candidate_ids:
        raise ResponseWindowObservationArtifactError("included contribution candidates disagree with observations.")
    excluded = frame.loc[~frame["included_in_label"].astype(bool)]
    if not excluded.empty and not np.allclose(
        pd.to_numeric(excluded["experiment_weight"], errors="coerce").to_numpy(dtype=float),
        0.0,
    ):
        raise ResponseWindowObservationArtifactError("excluded contributions must have zero experiment weight.")
    weights = pd.to_numeric(frame["experiment_weight"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(weights).all() or (weights < 0.0).any():
        raise ResponseWindowObservationArtifactError("contribution experiment weights must be finite and nonnegative.")
    _finite_values(frame, label="contributions")


def _validate_draws(frame: pd.DataFrame, *, candidate_ids: set[str], samples: int) -> None:
    missing = sorted({"candidate_id", "draw_index", *VALUE_COLUMNS} - set(frame.columns))
    if missing or set(frame["candidate_id"].astype(str)) != candidate_ids:
        raise ResponseWindowObservationArtifactError(f"hierarchical bootstrap draws disagree: missing={missing}")
    if isinstance(samples, bool) or samples < 100:
        raise ResponseWindowObservationArtifactError("hierarchical bootstrap sample count is invalid.")
    expected_indices = tuple(range(samples))
    indices = frame.groupby("candidate_id")["draw_index"].agg(lambda values: tuple(sorted(map(int, values))))
    if indices.empty or not indices.map(lambda value: value == expected_indices).all():
        raise ResponseWindowObservationArtifactError("hierarchical bootstrap draw indices are incomplete.")
    _finite_values(frame, label="hierarchical bootstrap draws")


def _finite_values(frame: pd.DataFrame, *, label: str) -> None:
    values = frame.loc[:, VALUE_COLUMNS].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ResponseWindowObservationArtifactError(f"{label} contain non-finite response vectors.")


__all__ = ["validate_frames"]
