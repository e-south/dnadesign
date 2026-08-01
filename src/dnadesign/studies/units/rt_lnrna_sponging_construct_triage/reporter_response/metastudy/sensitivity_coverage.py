"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/reporter_response/metastudy/sensitivity_coverage.py

Exact subject-by-reduction coverage for non-selectable sensitivity evidence.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from typing import Literal

from ...reader_evidence import ReaderEvidenceBindingSet
from ..profile import EndpointReduction
from .contracts import (
    DEFAULT_PROTOCOL,
    MaterializationAttemptReceipt,
    MaterializationOmission,
    MetastudyContractError,
    ProfileEvidence,
    ReaderRecordIdentity,
    canonical_digest,
)
from .evidence_projection import ProfileEvidenceProjection

SENSITIVITY_COVERAGE_CONTRACT_ID = "rt_lnrna_reporter_response_sensitivity_coverage.v1"


@dataclass(frozen=True, slots=True)
class SensitivitySubjectCoordinate:
    """One exact study subject and its raw Reader identity coordinate."""

    raw_design_id: str | None
    raw_assay_subject_id: str | None
    subject_id: str

    def __post_init__(self) -> None:
        if self.raw_design_id is None and self.raw_assay_subject_id is None:
            raise MetastudyContractError("sensitivity subject requires a raw Reader identity")
        for value in (self.raw_design_id, self.raw_assay_subject_id, self.subject_id):
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise MetastudyContractError("sensitivity subject identity values must be non-empty text")


@dataclass(frozen=True, slots=True)
class SensitivityCoverageEntry:
    """Exactly one profile or typed omission for one subject/reduction coordinate."""

    subject: SensitivitySubjectCoordinate
    reduction_id: str
    outcome: Literal["profile", "omission"]
    profile_digest: str | None
    omission: MaterializationOmission | None

    def __post_init__(self) -> None:
        if not isinstance(self.subject, SensitivitySubjectCoordinate):
            raise MetastudyContractError("sensitivity coverage subject must be typed")
        if not isinstance(self.reduction_id, str) or not self.reduction_id:
            raise MetastudyContractError("sensitivity coverage reduction_id must be non-empty text")
        if self.outcome == "profile":
            _require_digest(self.profile_digest, label="sensitivity coverage profile digest")
            if self.omission is not None:
                raise MetastudyContractError("profile sensitivity coverage cannot contain an omission")
        elif self.outcome == "omission":
            if self.profile_digest is not None or not isinstance(self.omission, MaterializationOmission):
                raise MetastudyContractError("omitted sensitivity coverage requires one typed omission")
            if self.omission.subject_id != self.subject.subject_id or self.omission.reduction_id != self.reduction_id:
                raise MetastudyContractError("sensitivity omission coordinate does not match its coverage entry")
        else:
            raise MetastudyContractError("sensitivity coverage outcome must be profile or omission")


@dataclass(frozen=True, slots=True)
class SensitivityCoverageLedger:
    """Canonical Cartesian coverage for one ready Reader materialization attempt."""

    contract_id: Literal["rt_lnrna_reporter_response_sensitivity_coverage.v1"]
    experiment_id: str
    materialization_attempt_digest: str
    reader_record_identity: ReaderRecordIdentity
    evidence_binding_artifact_id: str
    evidence_binding_artifact_digest: str
    expected_subjects: tuple[SensitivitySubjectCoordinate, ...]
    expected_reduction_ids: tuple[str, ...]
    entries: tuple[SensitivityCoverageEntry, ...]
    coverage_digest: str = field(default="", init=False)

    def __post_init__(self) -> None:
        if self.contract_id != SENSITIVITY_COVERAGE_CONTRACT_ID:
            raise MetastudyContractError("sensitivity coverage contract_id changed")
        if not isinstance(self.reader_record_identity, ReaderRecordIdentity):
            raise MetastudyContractError("sensitivity coverage Reader identity must be typed")
        if self.experiment_id != self.reader_record_identity.reader_experiment_id:
            raise MetastudyContractError("sensitivity coverage experiment identity mismatch")
        _require_digest(
            self.materialization_attempt_digest,
            label="sensitivity coverage materialization attempt digest",
        )
        if not isinstance(self.evidence_binding_artifact_id, str) or not self.evidence_binding_artifact_id:
            raise MetastudyContractError("sensitivity coverage binding artifact_id must be non-empty text")
        _require_digest(self.evidence_binding_artifact_digest, label="sensitivity coverage binding digest")
        if self.expected_subjects != tuple(sorted(self.expected_subjects, key=_subject_key)):
            raise MetastudyContractError("sensitivity coverage subjects are not canonically ordered")
        if len(self.expected_subjects) != len(set(self.expected_subjects)) or not self.expected_subjects:
            raise MetastudyContractError("sensitivity coverage subjects must be non-empty and unique")
        if self.expected_reduction_ids != declared_sensitivity_reduction_ids():
            raise MetastudyContractError("sensitivity coverage reductions differ from the declared set")
        coordinates = tuple(
            (subject, reduction_id)
            for subject in self.expected_subjects
            for reduction_id in self.expected_reduction_ids
        )
        if tuple((entry.subject, entry.reduction_id) for entry in self.entries) != coordinates:
            raise MetastudyContractError(
                "sensitivity coverage entries must equal the canonical subject-by-reduction Cartesian product"
            )
        object.__setattr__(self, "coverage_digest", canonical_digest(sensitivity_coverage_payload(self, False)))

    @property
    def omissions(self) -> tuple[MaterializationOmission, ...]:
        return tuple(entry.omission for entry in self.entries if entry.omission is not None)


def declared_sensitivity_reduction_ids() -> tuple[str, ...]:
    endpoints = tuple(f"endpoint-{value:g}h" for value in DEFAULT_PROTOCOL.endpoint_sensitivity_h)
    centered = tuple(
        f"window-{(start + end) / 2.0 - width / 2.0:g}-{(start + end) / 2.0 + width / 2.0:g}h"
        for start, end in DEFAULT_PROTOCOL.candidate_windows_h
        for width in DEFAULT_PROTOCOL.centered_window_sensitivity_widths_h
    )
    return endpoints + centered


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
    subjects = tuple(sorted(expected_subjects, key=_subject_key))
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


def sensitivity_coverage_payload(coverage: SensitivityCoverageLedger, include_digest: bool = True) -> dict[str, object]:
    # ``dataclasses.asdict`` preserves tuple containers, while the publication
    # contract is JSON-shaped and YAML round-trips sequences as lists. Normalize
    # at this serialization boundary so the in-memory and persisted payloads are
    # exactly equal rather than format-dependent.
    payload = json.loads(json.dumps(asdict(coverage), allow_nan=False))
    if not include_digest:
        payload.pop("coverage_digest", None)
    return payload


def sensitivity_coverage_receipt_payload(coverage: SensitivityCoverageLedger) -> dict[str, object]:
    """Project one full ledger into the compact source-controlled parity receipt."""

    return {
        "contract_id": coverage.contract_id,
        "experiment_id": coverage.experiment_id,
        "materialization_attempt_digest": coverage.materialization_attempt_digest,
        "reader_record_identity": asdict(coverage.reader_record_identity),
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
            identity = ReaderRecordIdentity(**identity_payload)
            subjects = tuple(SensitivitySubjectCoordinate(**row) for row in subjects_payload)
        except (TypeError, KeyError) as exc:
            raise MetastudyContractError(f"sensitivity coverage receipts[{index}] is malformed") from exc
        if not subjects or subjects != tuple(sorted(subjects, key=_subject_key)) or len(subjects) != len(set(subjects)):
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
        _require_digest(
            payload["evidence_binding_artifact_digest"],
            label=f"sensitivity coverage receipts[{index}] binding digest",
        )
        _require_digest(
            payload["materialization_attempt_digest"],
            label=f"sensitivity coverage receipts[{index}] attempt digest",
        )
        _require_digest(
            payload["coverage_digest"],
            label=f"sensitivity coverage receipts[{index}] coverage digest",
        )
        if (
            payload["experiment_id"] != attempt.experiment_id
            or identity != attempt.reader_record_identity
            or payload["materialization_attempt_digest"] != attempt.attempt_digest
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
        identity = ReaderRecordIdentity(**payload["reader_record_identity"])
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


def _subject_key(subject: SensitivitySubjectCoordinate) -> tuple[str, str, str]:
    return (subject.subject_id, subject.raw_design_id or "", subject.raw_assay_subject_id or "")


def _require_digest(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 71
        or not value.startswith("sha256:")
        or any(character not in "0123456789abcdef" for character in value[7:])
    ):
        raise MetastudyContractError(f"{label} must be a lowercase sha256 digest")
    return value


__all__ = [
    "SENSITIVITY_COVERAGE_CONTRACT_ID",
    "SensitivityCoverageEntry",
    "SensitivityCoverageLedger",
    "SensitivitySubjectCoordinate",
    "build_sensitivity_coverage",
    "declared_sensitivity_reduction_ids",
    "parse_sensitivity_coverage",
    "sensitivity_coverage_payload",
    "sensitivity_coverage_receipt_payload",
    "sensitivity_profile_coordinate_key",
    "validate_sensitivity_coverage",
    "validate_sensitivity_coverage_receipt_payloads",
    "validate_sensitivity_coverage_set",
]
