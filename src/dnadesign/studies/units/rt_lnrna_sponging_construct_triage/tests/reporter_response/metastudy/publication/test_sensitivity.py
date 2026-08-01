"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/tests/reporter_response/metastudy/publication/test_sensitivity.py

Tests sensitivity evidence coverage required for publication.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reporter_response.metastudy import (
    MaterializationOmission,
    MetastudyContractError,
    evaluate_sensitivity,
    publish_metastudy,
    verify_publication,
)
from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reporter_response.metastudy import (
    evaluate_metastudy as evaluate_metastudy_with_attempts,
)

from .._builders import (
    KINETIC_IDS,
    _attempts,
    _evidence,
    _ready,
    evaluate_metastudy,
)
from ..evidence._builders import (
    SENSITIVITY_COVERAGE_CONTRACT_ID,
    SensitivityCoverageEntry,
    SensitivityCoverageLedger,
    _complete_sensitivity_evidence,
    _sensitivity_coverages,
    _sensitivity_evidence,
)


def test_publication_requires_sensitivity_or_omission_for_each_ready_attempt(tmp_path: Path) -> None:
    selected = evaluate_metastudy(_evidence(), readiness=_ready())
    sensitivity = tuple(
        row for row in _sensitivity_evidence() if row.profile.provenance.reader_experiment_id != KINETIC_IDS[-1]
    )

    with pytest.raises(MetastudyContractError, match="exact ready-attempt set"):
        publish_metastudy(
            selected,
            tmp_path / "missing-one-experiment",
            primary_evidence=_evidence(),
            sensitivity_evidence=sensitivity,
            sensitivity_evaluations=evaluate_sensitivity(sensitivity),
        )


def test_publication_rejects_endpoint_8_only_as_incomplete_sensitivity_coverage(tmp_path: Path) -> None:
    selected = evaluate_metastudy(_evidence(), readiness=_ready())
    endpoint_8_only = _sensitivity_evidence()

    with pytest.raises(MetastudyContractError, match="sensitivity coverage"):
        publish_metastudy(
            selected,
            tmp_path / "endpoint-8-only",
            primary_evidence=_evidence(),
            sensitivity_evidence=endpoint_8_only,
            sensitivity_evaluations=evaluate_sensitivity(endpoint_8_only),
        )


def test_publication_allows_empty_sensitivity_only_with_typed_ready_attempt_omissions(
    tmp_path: Path,
) -> None:
    primary = _evidence()
    complete = _complete_sensitivity_evidence(primary)
    attempts = _attempts(primary)
    omitted_coverages = tuple(
        SensitivityCoverageLedger(
            contract_id=SENSITIVITY_COVERAGE_CONTRACT_ID,
            experiment_id=coverage.experiment_id,
            materialization_attempt_digest=coverage.materialization_attempt_digest,
            reader_record_identity=coverage.reader_record_identity,
            evidence_binding_artifact_id=coverage.evidence_binding_artifact_id,
            evidence_binding_artifact_digest=coverage.evidence_binding_artifact_digest,
            expected_subjects=coverage.expected_subjects,
            expected_reduction_ids=coverage.expected_reduction_ids,
            entries=tuple(
                SensitivityCoverageEntry(
                    subject=entry.subject,
                    reduction_id=entry.reduction_id,
                    outcome="omission",
                    profile_digest=None,
                    omission=MaterializationOmission(
                        code="synthetic_sensitivity_unavailable",
                        subject_id=entry.subject.subject_id,
                        reduction_id=entry.reduction_id,
                    ),
                )
                for entry in coverage.entries
            ),
        )
        for coverage in _sensitivity_coverages(complete, attempts)
    )
    selected = evaluate_metastudy_with_attempts(
        primary,
        readiness=_ready(),
        attempts=attempts,
    )

    destination = publish_metastudy(
        selected,
        tmp_path / "omitted-sensitivity",
        primary_evidence=primary,
        sensitivity_coverages=omitted_coverages,
    )

    payload = json.loads((destination / "sensitivity.json").read_text(encoding="utf-8"))
    assert payload["evaluations"] == []
    assert payload["profiles"] == []
    verify_publication(destination)
