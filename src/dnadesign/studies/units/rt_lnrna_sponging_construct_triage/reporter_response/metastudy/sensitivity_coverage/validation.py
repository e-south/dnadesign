"""Evidence and attempt validation for sensitivity-coverage ledgers."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

from ...profile.measurement import EndpointReduction
from ..contracts._values import MetastudyContractError
from ..contracts.materialization import MaterializationAttemptReceipt
from ..contracts.profile import ProfileEvidence
from ..evidence_projection.contracts import ProfileEvidenceProjection
from .contracts import SensitivityCoverageLedger, SensitivitySubjectCoordinate


def validate_sensitivity_coverage(
    coverage: SensitivityCoverageLedger,
    *,
    evidence: Iterable[ProfileEvidence | ProfileEvidenceProjection],
) -> None:
    declared_profiles = {
        (entry.subject, entry.reduction_id): entry.profile_digest
        for entry in coverage.entries
        if entry.profile_digest is not None
    }
    observed: dict[tuple[SensitivitySubjectCoordinate, str], str] = {}
    identity = coverage.reader_record_identity
    for row in evidence:
        provenance = row.profile.provenance
        subject = SensitivitySubjectCoordinate(
            provenance.raw_design_id, provenance.raw_assay_subject_id, row.profile.subject_id
        )
        key = (subject, sensitivity_profile_reduction_id(row))
        if key in observed:
            raise MetastudyContractError("sensitivity evidence contains duplicate coordinates")
        expected_provenance = (
            identity.reader_experiment_id,
            identity.reader_protocol_id,
            identity.reader_record_id,
            identity.reader_record_kind,
            identity.reader_record_revision,
            identity.reader_record_revision_digest,
            identity.reader_record_content_digest,
            identity.reader_record_schema_version,
            identity.reader_record_contract_id,
            identity.reader_record_path,
            coverage.evidence_binding_artifact_id,
            coverage.evidence_binding_artifact_digest,
        )
        actual_provenance = (
            provenance.reader_experiment_id,
            provenance.reader_protocol_id,
            provenance.reader_record_id,
            provenance.reader_record_kind,
            provenance.reader_record_revision,
            provenance.reader_record_revision_digest,
            provenance.reader_record_content_digest,
            provenance.reader_record_schema_version,
            provenance.reader_record_contract_id,
            provenance.reader_record_path,
            provenance.evidence_binding_artifact_id,
            provenance.evidence_binding_artifact_digest,
        )
        if actual_provenance != expected_provenance:
            raise MetastudyContractError("sensitivity profile provenance differs from its coverage ledger")
        observed[key] = row.audit.profile_digest
    if observed != declared_profiles:
        raise MetastudyContractError("sensitivity profile digests differ from exact coverage entries")


def validate_sensitivity_coverage_set(
    coverages: Iterable[SensitivityCoverageLedger],
    *,
    evidence: Iterable[ProfileEvidence | ProfileEvidenceProjection],
    attempts: Iterable[MaterializationAttemptReceipt],
) -> None:
    coverage_rows = tuple(coverages)
    evidence_rows = tuple(evidence)
    attempt_rows = tuple(attempts)
    _validate_coverage_receipts(coverage_rows, attempts=attempt_rows)
    ready = tuple(
        sorted(
            (row for row in attempt_rows if row.status in {"complete", "partial"}),
            key=lambda row: row.experiment_id,
        )
    )
    ready_ids = {row.experiment_id for row in ready}
    blocked_ids = {row.experiment_id for row in attempt_rows if row.status == "blocked"}
    evidence_by_experiment: dict[str, list[ProfileEvidence | ProfileEvidenceProjection]] = defaultdict(list)
    for row in evidence_rows:
        experiment_id = row.profile.provenance.reader_experiment_id
        if experiment_id in blocked_ids or experiment_id not in ready_ids:
            raise MetastudyContractError("sensitivity evidence belongs to a blocked or unplanned attempt")
        evidence_by_experiment[experiment_id].append(row)
    for coverage in coverage_rows:
        validate_sensitivity_coverage(coverage, evidence=tuple(evidence_by_experiment.get(coverage.experiment_id, ())))


def sensitivity_profile_reduction_id(row: ProfileEvidence | ProfileEvidenceProjection) -> str:
    reduction = row.profile.reduction
    if isinstance(reduction, EndpointReduction):
        return f"endpoint-{reduction.recorded_time_h:g}h"
    return f"window-{reduction.recorded_start_time_h:g}-{reduction.recorded_end_time_h:g}h"


def sensitivity_profile_coordinate_key(
    row: ProfileEvidence | ProfileEvidenceProjection,
) -> tuple[str, str, str, str, str]:
    provenance = row.profile.provenance
    return (
        provenance.reader_experiment_id,
        row.profile.subject_id,
        provenance.raw_design_id or "",
        provenance.raw_assay_subject_id or "",
        sensitivity_profile_reduction_id(row),
    )


def _validate_coverage_receipts(
    coverages: tuple[SensitivityCoverageLedger, ...],
    *,
    attempts: tuple[MaterializationAttemptReceipt, ...],
) -> None:
    ready = tuple(
        sorted((row for row in attempts if row.status in {"complete", "partial"}), key=lambda row: row.experiment_id)
    )
    if tuple(row.experiment_id for row in coverages) != tuple(row.experiment_id for row in ready):
        raise MetastudyContractError("sensitivity coverage must equal the exact ready-attempt set")
    for coverage, attempt in zip(coverages, ready, strict=True):
        if (
            attempt.reader_record_identity != coverage.reader_record_identity
            or attempt.attempt_digest != coverage.materialization_attempt_digest
        ):
            raise MetastudyContractError("sensitivity coverage differs from its exact materialization attempt")


__all__ = [
    "sensitivity_profile_coordinate_key",
    "sensitivity_profile_reduction_id",
    "validate_sensitivity_coverage",
    "validate_sensitivity_coverage_set",
]
