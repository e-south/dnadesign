"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/reporter_response/metastudy/sensitivity_coverage/contracts.py

Typed Cartesian coverage for sensitivity evidence.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

from ..contracts._values import MetastudyContractError, canonical_digest
from ..contracts.materialization import (
    MaterializationOmission,
    ReaderRecordIdentity,
    reader_record_identity_payload,
)
from ..contracts.protocol import DEFAULT_PROTOCOL
from ._values import require_digest

SENSITIVITY_COVERAGE_CONTRACT_ID = "rt_lnrna_reporter_response_sensitivity_coverage.v2"


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
            require_digest(self.profile_digest, label="sensitivity coverage profile digest")
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

    contract_id: Literal["rt_lnrna_reporter_response_sensitivity_coverage.v2"]
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
        require_digest(
            self.materialization_attempt_digest,
            label="sensitivity coverage materialization attempt digest",
        )
        if not isinstance(self.evidence_binding_artifact_id, str) or not self.evidence_binding_artifact_id:
            raise MetastudyContractError("sensitivity coverage binding artifact_id must be non-empty text")
        require_digest(self.evidence_binding_artifact_digest, label="sensitivity coverage binding digest")
        if self.expected_subjects != tuple(sorted(self.expected_subjects, key=subject_key)):
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
        object.__setattr__(self, "coverage_digest", canonical_digest(coverage_payload(self, include_digest=False)))

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


def coverage_payload(coverage: SensitivityCoverageLedger, *, include_digest: bool) -> dict[str, object]:
    payload = {
        "contract_id": coverage.contract_id,
        "experiment_id": coverage.experiment_id,
        "materialization_attempt_digest": coverage.materialization_attempt_digest,
        "reader_record_identity": reader_record_identity_payload(coverage.reader_record_identity),
        "evidence_binding_artifact_id": coverage.evidence_binding_artifact_id,
        "evidence_binding_artifact_digest": coverage.evidence_binding_artifact_digest,
        "expected_subjects": [asdict(item) for item in coverage.expected_subjects],
        "expected_reduction_ids": list(coverage.expected_reduction_ids),
        "entries": [asdict(item) for item in coverage.entries],
        "coverage_digest": coverage.coverage_digest,
    }
    if not include_digest:
        payload.pop("coverage_digest", None)
    return payload


def subject_key(subject: SensitivitySubjectCoordinate) -> tuple[str, str, str]:
    return (subject.subject_id, subject.raw_design_id or "", subject.raw_assay_subject_id or "")


__all__ = [
    "SENSITIVITY_COVERAGE_CONTRACT_ID",
    "SensitivityCoverageEntry",
    "SensitivityCoverageLedger",
    "SensitivitySubjectCoordinate",
    "coverage_payload",
    "declared_sensitivity_reduction_ids",
    "subject_key",
]
