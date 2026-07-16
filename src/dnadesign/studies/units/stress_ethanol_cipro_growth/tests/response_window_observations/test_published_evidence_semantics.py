"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/tests/response_window_observations/test_published_evidence_semantics.py

Tamper tests for published repeat and uncertainty semantics.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import pandas as pd
import pytest

from dnadesign.studies.units.stress_ethanol_cipro_growth.response_window_observations import (
    artifact_repeat_validation,
    artifact_uncertainty_validation,
)
from dnadesign.studies.units.stress_ethanol_cipro_growth.response_window_observations.artifact_contract import (
    ResponseWindowObservationArtifactError,
)
from dnadesign.studies.units.stress_ethanol_cipro_growth.response_window_observations.contracts import (
    VALUE_COLUMNS,
)
from dnadesign.studies.units.stress_ethanol_cipro_growth.response_window_observations.repeat_diagnostics import (
    repeat_diagnostic_rows,
)


def test_repeat_diagnostics_are_bound_to_typed_final_decision() -> None:
    contributions, diagnostics = _repeat_records()
    diagnostics.loc[diagnostics["component"].eq("b01"), "classification"] = "unresolved"

    with pytest.raises(ResponseWindowObservationArtifactError, match="classification.*disagrees"):
        artifact_repeat_validation.validate_repeat_records(diagnostics, contributions=contributions)


def test_unresolved_repeat_cannot_appear_in_published_bundle() -> None:
    contributions, diagnostics = _repeat_records()
    contributions["repeat_decision"] = "review_required"
    contributions["repeat_classification"] = "unresolved"
    contributions[
        ["repeat_evidence_artifact", "repeat_evidence_sha256", "repeat_adjudicated_by", "repeat_adjudicated_at"]
    ] = None

    with pytest.raises(ResponseWindowObservationArtifactError, match="unresolved repeat decisions"):
        artifact_repeat_validation.validate_repeat_records(diagnostics, contributions=contributions)


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("population_coverage_claimed", "False", "must be boolean"),
        ("bootstrap_sd", -0.1, "must be nonnegative"),
        ("bootstrap_samples", 99, "disagree with the manifest"),
    ],
)
def test_low_n_uncertainty_claims_fail_closed(column: str, value: object, message: str) -> None:
    observations, uncertainty = _uncertainty_records()
    uncertainty[column] = value

    with pytest.raises(ResponseWindowObservationArtifactError, match=message):
        artifact_uncertainty_validation.validate_uncertainty_records(
            uncertainty,
            observations=observations,
            bootstrap_samples=100,
        )


def _repeat_records() -> tuple[pd.DataFrame, pd.DataFrame]:
    evidence = {
        "repeat_decision": "label_source_selected",
        "repeat_decision_reason": "latest reviewed Reader experiment selected",
        "repeat_classification": "source_agreement_accepted",
        "repeat_evidence_artifact": "reviews/candidate-a.json",
        "repeat_evidence_sha256": "a" * 64,
        "repeat_adjudicated_by": "study-reviewer",
        "repeat_adjudicated_at": "2026-07-15T12:00:00+00:00",
        "label_source_reader_experiment_id": "experiment-b",
    }
    contributions = pd.DataFrame.from_records(
        [
            {
                "candidate_id": "candidate-a",
                "design_id": "design-a",
                "reader_experiment_id": experiment_id,
                "selected_as_label_source": experiment_id == "experiment-b",
                "included_in_label": experiment_id == "experiment-b",
                "label_exclusion_reason": None if experiment_id == "experiment-b" else "not_selected_repeat_evidence",
                **evidence,
                **{component: offset for component in VALUE_COLUMNS},
            }
            for experiment_id, offset in (("experiment-a", 0.0), ("experiment-b", 2.0))
        ]
    )
    decisions = pd.DataFrame.from_records(
        [
            {
                "candidate_id": "candidate-a",
                "label_source_reader_experiment_id": "experiment-b",
                "status": "label_source_selected",
                "classification": "source_agreement_accepted",
                "evidence_artifact": "reviews/candidate-a.json",
                "evidence_sha256": "a" * 64,
                "adjudicated_by": "study-reviewer",
                "adjudicated_at": "2026-07-15T12:00:00+00:00",
                "reason": "latest reviewed Reader experiment selected",
            }
        ]
    )
    return contributions, repeat_diagnostic_rows(contributions, decisions=decisions)


def _uncertainty_records() -> tuple[pd.DataFrame, pd.DataFrame]:
    observations = pd.DataFrame(
        {"candidate_id": ["candidate-a"], "label_source_reader_experiment_id": ["experiment-b"]}
    )
    uncertainty = pd.DataFrame.from_records(
        [
            {
                "candidate_id": "candidate-a",
                "component": component,
                "label_source_reader_experiment_id": "experiment-b",
                "point_estimate": 1.0,
                "bootstrap_sd": 0.2,
                "descriptive_interval_low": 0.5,
                "descriptive_interval_high": 1.5,
                "nominal_interval_mass": 0.9,
                "interval_scope": "descriptive_selected_source_joint_bootstrap",
                "population_coverage_claimed": False,
                "bootstrap_samples": 100,
            }
            for component in VALUE_COLUMNS
        ]
    )
    return observations, uncertainty
