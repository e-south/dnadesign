"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/tests/response_window_observations/test_repeat_evidence.py

Contract tests for typed repeated-experiment evidence.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from dnadesign.studies.units.stress_ethanol_cipro_growth.response_window_observations.contracts import (
    VALUE_COLUMNS,
)
from dnadesign.studies.units.stress_ethanol_cipro_growth.response_window_observations.repeat_evidence import (
    RepeatEvidenceContractError,
    validate_repeat_evidence_artifact,
)

READER_SHA256 = "a" * 64
PRIMARY_REDUCTION_ID = "event_logmean_4_8h_post"
EXPERIMENT_IDS = ("experiment-a", "experiment-b")


def test_repeat_evidence_binds_the_exact_decision_and_allows_large_disagreement(tmp_path: Path) -> None:
    payload = _payload()
    payload["candidate_reviews"][0]["comparison_evidence"]["component_ranges"]["r11"] = 1000.0
    payload["candidate_reviews"][0]["comparison_evidence"]["maximum_component_range"] = 1000.0
    payload["candidate_reviews"][0]["comparison_evidence"]["maximum_range_components"] = ["r11"]
    path = _write(tmp_path / "repeat-evidence.json", payload)

    validate_repeat_evidence_artifact(
        path,
        expected_reader_bundle_sha256=READER_SHA256,
        expected_primary_reduction_id=PRIMARY_REDUCTION_ID,
        candidate_id="candidate-a",
        reader_experiment_ids=EXPERIMENT_IDS,
        label_source_reader_experiment_id="experiment-b",
        status="label_source_selected",
        classification="source_agreement_accepted",
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda payload: payload.update(reader_bundle_sha256="b" * 64),
            "Reader bundle digest disagrees",
        ),
        (
            lambda payload: payload.update(primary_reduction_id="event_logmean_6_12h_post"),
            "primary reduction disagrees",
        ),
        (
            lambda payload: payload["candidate_reviews"][0].update(candidate_id="candidate-b"),
            "has no entry for candidate",
        ),
        (
            lambda payload: payload["candidate_reviews"][0].update(
                reader_experiment_ids=["experiment-a", "experiment-c"],
                label_source_reader_experiment_id="experiment-a",
            ),
            "experiment identities disagree",
        ),
        (
            lambda payload: payload["candidate_reviews"][0].update(label_source_reader_experiment_id="experiment-a"),
            "label source disagrees",
        ),
        (
            lambda payload: payload["candidate_reviews"][0].update(classification="corrected_technical_error"),
            "classification disagrees",
        ),
    ],
)
def test_repeat_evidence_rejects_decision_or_source_drift(tmp_path: Path, mutation, message: str) -> None:
    payload = _payload()
    mutation(payload)
    path = _write(tmp_path / "repeat-evidence.json", payload)

    with pytest.raises(RepeatEvidenceContractError, match=message):
        validate_repeat_evidence_artifact(
            path,
            expected_reader_bundle_sha256=READER_SHA256,
            expected_primary_reduction_id=PRIMARY_REDUCTION_ID,
            candidate_id="candidate-a",
            reader_experiment_ids=EXPERIMENT_IDS,
            label_source_reader_experiment_id="experiment-b",
            status="label_source_selected",
            classification="source_agreement_accepted",
        )


def test_repeat_evidence_binds_status_independently_of_classification(tmp_path: Path) -> None:
    payload = _payload()
    payload["candidate_reviews"][0].update(
        status="label_source_excluded",
        classification="noncomparable_assay_context",
        label_source_reader_experiment_id=None,
    )
    path = _write(tmp_path / "repeat-evidence.json", payload)

    with pytest.raises(RepeatEvidenceContractError, match="status disagrees"):
        validate_repeat_evidence_artifact(
            path,
            expected_reader_bundle_sha256=READER_SHA256,
            expected_primary_reduction_id=PRIMARY_REDUCTION_ID,
            candidate_id="candidate-a",
            reader_experiment_ids=EXPERIMENT_IDS,
            label_source_reader_experiment_id=None,
            status="label_source_selected",
            classification="noncomparable_assay_context",
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda comparison: comparison["component_ranges"].pop("r00"),
            "component ranges must be exactly",
        ),
        (
            lambda comparison: comparison.update(maximum_component_range=4.0),
            "maximum component range disagrees",
        ),
        (
            lambda comparison: comparison.update(maximum_range_components=["r00"]),
            "maximum-range components disagree",
        ),
        (
            lambda comparison: comparison["component_ranges"].update(r00=-1.0),
            "finite and nonnegative",
        ),
    ],
)
def test_repeat_evidence_recomputes_comparison_summary(tmp_path: Path, mutation, message: str) -> None:
    payload = _payload()
    mutation(payload["candidate_reviews"][0]["comparison_evidence"])
    path = _write(tmp_path / "repeat-evidence.json", payload)

    with pytest.raises(RepeatEvidenceContractError, match=message):
        validate_repeat_evidence_artifact(
            path,
            expected_reader_bundle_sha256=READER_SHA256,
            expected_primary_reduction_id=PRIMARY_REDUCTION_ID,
            candidate_id="candidate-a",
            reader_experiment_ids=EXPERIMENT_IDS,
            label_source_reader_experiment_id="experiment-b",
            status="label_source_selected",
            classification="source_agreement_accepted",
        )


def test_repeat_evidence_rejects_duplicate_candidate_entries(tmp_path: Path) -> None:
    payload = _payload()
    payload["candidate_reviews"].append(deepcopy(payload["candidate_reviews"][0]))
    path = _write(tmp_path / "repeat-evidence.json", payload)

    with pytest.raises(RepeatEvidenceContractError, match="duplicate candidate IDs"):
        validate_repeat_evidence_artifact(
            path,
            expected_reader_bundle_sha256=READER_SHA256,
            expected_primary_reduction_id=PRIMARY_REDUCTION_ID,
            candidate_id="candidate-a",
            reader_experiment_ids=EXPERIMENT_IDS,
            label_source_reader_experiment_id="experiment-b",
            status="label_source_selected",
            classification="source_agreement_accepted",
        )


def _payload() -> dict[str, object]:
    ranges = {component: float(index + 1) / 10.0 for index, component in enumerate(VALUE_COLUMNS)}
    return {
        "schema_id": "stress_ethanol_cipro_growth.repeat_adjudication_evidence.v1",
        "schema_version": "1",
        "study_id": "stress_ethanol_cipro_growth",
        "reader_bundle_sha256": READER_SHA256,
        "primary_reduction_id": PRIMARY_REDUCTION_ID,
        "candidate_reviews": [
            {
                "candidate_id": "candidate-a",
                "reader_experiment_ids": list(EXPERIMENT_IDS),
                "label_source_reader_experiment_id": "experiment-b",
                "status": "label_source_selected",
                "classification": "source_agreement_accepted",
                "comparison_evidence": {
                    "component_ranges": ranges,
                    "maximum_component_range": ranges["b11"],
                    "maximum_range_components": ["b11"],
                },
            }
        ],
    }


def _write(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
