"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/reporter_response/metastudy/sensitivity_coverage/serialization.py

Canonical payload codecs for sensitivity-coverage ledgers and receipts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict

from ..contracts._materialization.lineage import (
    reader_record_identity_from_payload,
    reader_record_identity_payload,
)
from ..contracts._values import MetastudyContractError
from ..contracts.materialization import MaterializationAttemptReceipt, MaterializationOmission
from ._values import require_digest
from .contracts import (
    SENSITIVITY_COVERAGE_CONTRACT_ID,
    SensitivityCoverageEntry,
    SensitivityCoverageLedger,
    SensitivitySubjectCoordinate,
    coverage_payload,
    declared_sensitivity_reduction_ids,
    subject_key,
)


def sensitivity_coverage_payload(
    coverage: SensitivityCoverageLedger,
    include_digest: bool = True,
) -> dict[str, object]:
    return coverage_payload(coverage, include_digest=include_digest)


def sensitivity_coverage_receipt_payload(coverage: SensitivityCoverageLedger) -> dict[str, object]:
    """Project one full ledger into the compact source-controlled parity receipt."""

    return {
        "contract_id": coverage.contract_id,
        "experiment_id": coverage.experiment_id,
        "materialization_attempt_digest": coverage.materialization_attempt_digest,
        "reader_record_identity": reader_record_identity_payload(coverage.reader_record_identity),
        "evidence_binding_artifact_id": coverage.evidence_binding_artifact_id,
        "evidence_binding_artifact_digest": coverage.evidence_binding_artifact_digest,
        "expected_subjects": [asdict(row) for row in coverage.expected_subjects],
        "expected_reduction_ids": list(coverage.expected_reduction_ids),
        "profile_count": sum(entry.outcome == "profile" for entry in coverage.entries),
        "omission_count": sum(entry.outcome == "omission" for entry in coverage.entries),
        "coverage_digest": coverage.coverage_digest,
    }


def validate_sensitivity_coverage_receipt_payloads(
    payloads: object,
    *,
    attempts: Iterable[MaterializationAttemptReceipt],
) -> None:
    """Validate compact receipts against the exact primary attempt ledger."""

    if not isinstance(payloads, list):
        raise MetastudyContractError("sensitivity coverage receipts must be an array")
    ready = tuple(
        sorted((row for row in attempts if row.status in {"complete", "partial"}), key=lambda row: row.experiment_id)
    )
    if len(payloads) != len(ready):
        raise MetastudyContractError("sensitivity coverage receipts must equal the ready-attempt set")
    expected_fields = {
        "contract_id",
        "experiment_id",
        "materialization_attempt_digest",
        "reader_record_identity",
        "evidence_binding_artifact_id",
        "evidence_binding_artifact_digest",
        "expected_subjects",
        "expected_reduction_ids",
        "profile_count",
        "omission_count",
        "coverage_digest",
    }
    for index, (payload, attempt) in enumerate(zip(payloads, ready, strict=True)):
        if not isinstance(payload, Mapping) or set(payload) != expected_fields:
            raise MetastudyContractError(
                f"sensitivity coverage receipts[{index}] fields do not match the exact contract"
            )
        if payload["contract_id"] != SENSITIVITY_COVERAGE_CONTRACT_ID:
            raise MetastudyContractError(f"sensitivity coverage receipts[{index}] contract_id changed")
        identity_payload = payload["reader_record_identity"]
        subjects_payload = payload["expected_subjects"]
        reductions = payload["expected_reduction_ids"]
        if (
            not isinstance(identity_payload, Mapping)
            or not isinstance(subjects_payload, list)
            or not isinstance(reductions, list)
        ):
            raise MetastudyContractError(f"sensitivity coverage receipts[{index}] structure changed")
        try:
            identity = reader_record_identity_from_payload(identity_payload, index=index)
            subjects = tuple(SensitivitySubjectCoordinate(**row) for row in subjects_payload)
        except (TypeError, KeyError) as exc:
            raise MetastudyContractError(f"sensitivity coverage receipts[{index}] is malformed") from exc
        if not subjects or subjects != tuple(sorted(subjects, key=subject_key)) or len(subjects) != len(set(subjects)):
            raise MetastudyContractError(f"sensitivity coverage receipts[{index}] subjects are not canonical")
        if tuple(reductions) != declared_sensitivity_reduction_ids():
            raise MetastudyContractError(f"sensitivity coverage receipts[{index}] reductions changed")
        profile_count = payload["profile_count"]
        omission_count = payload["omission_count"]
        if (
            type(profile_count) is not int
            or profile_count < 0
            or type(omission_count) is not int
            or omission_count < 0
            or profile_count + omission_count != len(subjects) * len(reductions)
        ):
            raise MetastudyContractError(f"sensitivity coverage receipts[{index}] coordinate counts changed")
        if not isinstance(payload["evidence_binding_artifact_id"], str) or not payload["evidence_binding_artifact_id"]:
            raise MetastudyContractError(f"sensitivity coverage receipts[{index}] binding artifact_id changed")
        require_digest(
            payload["evidence_binding_artifact_digest"],
            label=f"sensitivity coverage receipts[{index}] binding digest",
        )
        require_digest(
            payload["materialization_attempt_digest"],
            label=f"sensitivity coverage receipts[{index}] attempt digest",
        )
        require_digest(
            payload["coverage_digest"],
            label=f"sensitivity coverage receipts[{index}] coverage digest",
        )
        if (
            payload["experiment_id"] != attempt.experiment_id
            or identity != attempt.reader_record_identity
            or payload["materialization_attempt_digest"] != attempt.attempt_digest
            or payload["evidence_binding_artifact_id"] != attempt.evidence_binding_artifact_id
            or payload["evidence_binding_artifact_digest"] != attempt.evidence_binding_artifact_digest
            or tuple(subject.subject_id for subject in subjects) != attempt.expected_subject_ids
        ):
            raise MetastudyContractError(
                f"sensitivity coverage receipts[{index}] differs from its exact materialization attempt"
            )


def parse_sensitivity_coverage(payload: object, *, index: int) -> SensitivityCoverageLedger:
    expected = {
        "contract_id",
        "experiment_id",
        "materialization_attempt_digest",
        "reader_record_identity",
        "evidence_binding_artifact_id",
        "evidence_binding_artifact_digest",
        "expected_subjects",
        "expected_reduction_ids",
        "entries",
        "coverage_digest",
    }
    if not isinstance(payload, Mapping) or set(payload) != expected:
        raise MetastudyContractError(f"sensitivity coverages[{index}] fields do not match the exact contract")
    try:
        identity_payload = payload["reader_record_identity"]
        if not isinstance(identity_payload, Mapping):
            raise TypeError("reader_record_identity must be an object")
        identity = reader_record_identity_from_payload(identity_payload, index=index)
        subjects = tuple(SensitivitySubjectCoordinate(**row) for row in payload["expected_subjects"])
        entries = tuple(
            SensitivityCoverageEntry(
                subject=SensitivitySubjectCoordinate(**row["subject"]),
                reduction_id=row["reduction_id"],
                outcome=row["outcome"],
                profile_digest=row["profile_digest"],
                omission=MaterializationOmission(**row["omission"]) if row["omission"] is not None else None,
            )
            for row in payload["entries"]
        )
    except (TypeError, KeyError) as exc:
        raise MetastudyContractError(f"sensitivity coverages[{index}] is malformed") from exc
    ledger = SensitivityCoverageLedger(
        contract_id=payload["contract_id"],
        experiment_id=payload["experiment_id"],
        materialization_attempt_digest=payload["materialization_attempt_digest"],
        reader_record_identity=identity,
        evidence_binding_artifact_id=payload["evidence_binding_artifact_id"],
        evidence_binding_artifact_digest=payload["evidence_binding_artifact_digest"],
        expected_subjects=subjects,
        expected_reduction_ids=tuple(payload["expected_reduction_ids"]),
        entries=entries,
    )
    if payload["coverage_digest"] != ledger.coverage_digest:
        raise MetastudyContractError(f"sensitivity coverages[{index}] digest mismatch")
    return ledger


__all__ = [
    "parse_sensitivity_coverage",
    "sensitivity_coverage_payload",
    "sensitivity_coverage_receipt_payload",
    "validate_sensitivity_coverage_receipt_payloads",
]
