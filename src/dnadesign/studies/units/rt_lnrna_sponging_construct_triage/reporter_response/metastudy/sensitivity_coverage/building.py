"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/reporter_response/metastudy/sensitivity_coverage/building.py

Source-closed construction of sensitivity-coverage ledgers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Iterable

from ....reader_evidence import ReaderEvidenceBindingSet
from ..contracts._values import MetastudyContractError
from ..contracts.materialization import MaterializationAttemptReceipt, MaterializationOmission
from ..contracts.profile import ProfileEvidence
from .contracts import (
    SENSITIVITY_COVERAGE_CONTRACT_ID,
    SensitivityCoverageEntry,
    SensitivityCoverageLedger,
    SensitivitySubjectCoordinate,
    declared_sensitivity_reduction_ids,
    subject_key,
)
from .validation import sensitivity_profile_reduction_id, validate_sensitivity_coverage


def build_sensitivity_coverage(
    *,
    attempt: MaterializationAttemptReceipt,
    bindings: ReaderEvidenceBindingSet,
    expected_subjects: Iterable[SensitivitySubjectCoordinate],
    evidence: Iterable[ProfileEvidence],
    omissions: Iterable[MaterializationOmission],
) -> SensitivityCoverageLedger:
    if attempt.status not in {"complete", "partial"} or attempt.reader_record_identity is None:
        raise MetastudyContractError("sensitivity coverage requires one usable materialization attempt")
    reader_record_identity = attempt.reader_record_identity
    if not isinstance(bindings, ReaderEvidenceBindingSet) or not bindings.is_source_closed:
        raise MetastudyContractError("sensitivity coverage requires source-closed evidence bindings")
    evidence_rows = tuple(evidence)
    subjects = tuple(sorted(expected_subjects, key=subject_key))
    bound_coordinates = {
        SensitivitySubjectCoordinate(row.raw_design_id, row.raw_assay_subject_id, row.subject_id)
        for row in bindings.rows
        if row.binding_state == "bound" and row.subject_id is not None
    }
    if not set(subjects) <= bound_coordinates:
        raise MetastudyContractError("sensitivity coverage subjects are not declared by evidence bindings")
    evidence_by_coordinate: dict[tuple[SensitivitySubjectCoordinate, str], ProfileEvidence] = {}
    for row in evidence_rows:
        provenance = row.profile.provenance
        subject = SensitivitySubjectCoordinate(
            provenance.raw_design_id, provenance.raw_assay_subject_id, row.profile.subject_id
        )
        key = (subject, sensitivity_profile_reduction_id(row))
        if key in evidence_by_coordinate:
            raise MetastudyContractError("sensitivity coverage contains duplicate profile coordinates")
        evidence_by_coordinate[key] = row
    omissions_by_coordinate: dict[tuple[str, str], MaterializationOmission] = {}
    for omission in omissions:
        key = (omission.subject_id, omission.reduction_id)
        if key in omissions_by_coordinate:
            raise MetastudyContractError("sensitivity coverage contains duplicate omission coordinates")
        omissions_by_coordinate[key] = omission
    entries: list[SensitivityCoverageEntry] = []
    for subject in subjects:
        for reduction_id in declared_sensitivity_reduction_ids():
            profile = evidence_by_coordinate.pop((subject, reduction_id), None)
            omission = omissions_by_coordinate.pop((subject.subject_id, reduction_id), None)
            if (profile is None) == (omission is None):
                raise MetastudyContractError(
                    "sensitivity coverage requires exactly one profile or omission for every coordinate"
                )
            entries.append(
                SensitivityCoverageEntry(
                    subject=subject,
                    reduction_id=reduction_id,
                    outcome="profile" if profile is not None else "omission",
                    profile_digest=profile.audit.profile_digest if profile is not None else None,
                    omission=omission,
                )
            )
    if evidence_by_coordinate or omissions_by_coordinate:
        raise MetastudyContractError("sensitivity coverage contains undeclared coordinates")
    ledger = SensitivityCoverageLedger(
        contract_id=SENSITIVITY_COVERAGE_CONTRACT_ID,
        experiment_id=reader_record_identity.reader_experiment_id,
        materialization_attempt_digest=attempt.attempt_digest,
        reader_record_identity=reader_record_identity,
        evidence_binding_artifact_id=bindings.artifact_id,
        evidence_binding_artifact_digest=bindings.artifact_digest,
        expected_subjects=subjects,
        expected_reduction_ids=declared_sensitivity_reduction_ids(),
        entries=tuple(entries),
    )
    validate_sensitivity_coverage(ledger, evidence=evidence_rows)
    return ledger


__all__ = ["build_sensitivity_coverage"]
