"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/response_window_observations/artifact_uncertainty_validation.py

Validate descriptive low-n uncertainty records without inferential overclaim.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .artifact_contract import ResponseWindowObservationArtifactError


def validate_uncertainty_records(
    frame: pd.DataFrame,
    *,
    observations: pd.DataFrame,
    bootstrap_samples: int,
) -> None:
    required = {
        "candidate_id",
        "component",
        "label_source_reader_experiment_id",
        "point_estimate",
        "bootstrap_sd",
        "descriptive_interval_low",
        "descriptive_interval_high",
        "nominal_interval_mass",
        "interval_scope",
        "population_coverage_claimed",
        "bootstrap_samples",
    }
    if missing := sorted(required - set(frame.columns)):
        raise ResponseWindowObservationArtifactError(f"uncertainty semantics are incomplete: {missing}")
    source_ids = (
        observations.assign(candidate_id=observations["candidate_id"].astype(str))
        .set_index("candidate_id")["label_source_reader_experiment_id"]
        .astype(str)
    )
    candidate_ids = frame["candidate_id"].astype(str)
    grouped_sources = frame.assign(candidate_id=candidate_ids).groupby("candidate_id")[
        "label_source_reader_experiment_id"
    ]
    if grouped_sources.nunique().gt(1).any() or grouped_sources.first().astype(str).to_dict() != source_ids.to_dict():
        raise ResponseWindowObservationArtifactError("uncertainty label sources disagree with observations.")
    numeric = frame[
        [
            "point_estimate",
            "bootstrap_sd",
            "descriptive_interval_low",
            "descriptive_interval_high",
            "nominal_interval_mass",
        ]
    ].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ResponseWindowObservationArtifactError("uncertainty records contain non-finite values.")
    if (numeric["bootstrap_sd"] < 0.0).any():
        raise ResponseWindowObservationArtifactError("selected-source bootstrap spread must be nonnegative.")
    if (numeric["descriptive_interval_low"] > numeric["descriptive_interval_high"]).any():
        raise ResponseWindowObservationArtifactError("descriptive uncertainty interval bounds are reversed.")
    masses = numeric["nominal_interval_mass"]
    if masses.nunique() != 1 or not masses.between(0.0, 1.0, inclusive="neither").all():
        raise ResponseWindowObservationArtifactError("nominal interval mass is invalid or inconsistent.")
    if set(frame["interval_scope"].astype(str)) != {"descriptive_selected_source_joint_bootstrap"}:
        raise ResponseWindowObservationArtifactError("uncertainty interval scope is not descriptive.")
    if not frame["population_coverage_claimed"].map(lambda value: isinstance(value, (bool, np.bool_))).all():
        raise ResponseWindowObservationArtifactError("population coverage flags must be boolean.")
    if frame["population_coverage_claimed"].astype(bool).any():
        raise ResponseWindowObservationArtifactError("selected-source bootstrap rows cannot claim population coverage.")
    sample_counts = pd.to_numeric(frame["bootstrap_samples"], errors="coerce")
    if not sample_counts.eq(bootstrap_samples).all():
        raise ResponseWindowObservationArtifactError("uncertainty bootstrap sample counts disagree with the manifest.")


__all__ = ["validate_uncertainty_records"]
