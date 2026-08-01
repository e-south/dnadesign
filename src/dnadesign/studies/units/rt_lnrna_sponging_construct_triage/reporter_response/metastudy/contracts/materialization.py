"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/reporter_response/metastudy/contracts/materialization.py

Reader evidence readiness and materialization receipt contracts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from dataclasses import fields as dataclass_fields
from typing import Literal

from ._values import MetastudyContractError, _digest, _required_text, _unique_text, canonical_digest
from .protocol import DEFAULT_PROTOCOL

_RECEIPT_CLOSURE_TOKEN = object()
_OWNER_BRIDGE_CLOSURE_TOKEN = object()


@dataclass(frozen=True, slots=True)
class EvidenceReadiness:
    """Read-only summary of exact selected evidence readiness."""

    selected_experiment_count: int
    ready_experiment_count: int
    ready_experiment_ids: tuple[str, ...]
    blocked_experiment_ids: tuple[str, ...]
    receipt_digest: str
    _receipt_closure: object | None = field(default=None, init=False, repr=False, compare=False)
    _owner_bridge_closure: object | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if isinstance(self.selected_experiment_count, bool) or self.selected_experiment_count < 1:
            raise MetastudyContractError("selected_experiment_count must be positive")
        if (
            isinstance(self.ready_experiment_count, bool)
            or not 0 <= self.ready_experiment_count <= self.selected_experiment_count
        ):
            raise MetastudyContractError("ready_experiment_count must be between zero and selected_experiment_count")
        _unique_text(self.ready_experiment_ids, label="ready_experiment_ids", allow_empty=True)
        _unique_text(self.blocked_experiment_ids, label="blocked_experiment_ids", allow_empty=True)
        if len(self.ready_experiment_ids) != self.ready_experiment_count:
            raise MetastudyContractError("ready experiment identities must match ready_experiment_count")
        if len(self.blocked_experiment_ids) != self.selected_experiment_count - self.ready_experiment_count:
            raise MetastudyContractError(
                "blocked experiment identities must account for every selected experiment not ready"
            )
        if set(self.ready_experiment_ids) & set(self.blocked_experiment_ids):
            raise MetastudyContractError("ready and blocked experiment identities must not overlap")
        _digest(self.receipt_digest, label="receipt_digest")

    @classmethod
    def _from_validated_receipt(cls, **values: object) -> EvidenceReadiness:
        readiness = cls(**values)
        object.__setattr__(readiness, "_receipt_closure", _RECEIPT_CLOSURE_TOKEN)
        return readiness

    @property
    def is_receipt_validated(self) -> bool:
        return self._receipt_closure is _RECEIPT_CLOSURE_TOKEN

    @classmethod
    def _from_owner_bridge_receipt(cls, **values: object) -> EvidenceReadiness:
        readiness = cls._from_validated_receipt(**values)
        object.__setattr__(readiness, "_owner_bridge_closure", _OWNER_BRIDGE_CLOSURE_TOKEN)
        return readiness

    @property
    def is_selection_authorized(self) -> bool:
        return self._owner_bridge_closure is _OWNER_BRIDGE_CLOSURE_TOKEN


@dataclass(frozen=True, slots=True)
class ReaderRecordIdentity:
    """Exact public Reader record identity preserved by one materialization attempt."""

    reader_experiment_id: str
    reader_protocol_id: str
    reader_record_id: str
    reader_record_kind: str
    reader_record_schema_version: int
    reader_record_revision: int
    reader_record_revision_digest: str
    reader_record_contract_id: str
    reader_record_content_digest: str
    reader_record_path: str

    def __post_init__(self) -> None:
        for name in (
            "reader_experiment_id",
            "reader_protocol_id",
            "reader_record_id",
            "reader_record_kind",
            "reader_record_contract_id",
            "reader_record_path",
        ):
            _required_text(getattr(self, name), label=name)
        if self.reader_record_schema_version != 6:
            raise MetastudyContractError("attempt Reader record schema version must equal 6")
        if type(self.reader_record_revision) is not int or self.reader_record_revision < 1:
            raise MetastudyContractError("attempt Reader record revision must be positive")
        _digest(self.reader_record_revision_digest, label="attempt Reader revision digest")
        _digest(self.reader_record_content_digest, label="attempt Reader content digest")


@dataclass(frozen=True, slots=True)
class MaterializationBlocker:
    """One source- or experiment-level failure that prevents materialization."""

    code: str

    def __post_init__(self) -> None:
        _required_text(self.code, label="materialization blocker code")


@dataclass(frozen=True, slots=True)
class MaterializationOmission:
    """One unusable subject/window coordinate within an otherwise usable record."""

    code: str
    subject_id: str
    reduction_id: str

    def __post_init__(self) -> None:
        _required_text(self.code, label="materialization omission code")
        _required_text(self.subject_id, label="materialization omission subject_id")
        _required_text(self.reduction_id, label="materialization omission reduction_id")


@dataclass(frozen=True, slots=True)
class MaterializationAttemptReceipt:
    """Digest-bound result of attempting one selected Reader experiment."""

    contract_id: Literal["rt_lnrna_reporter_response_materialization_attempt.v4"]
    experiment_id: str
    reader_record_identity: ReaderRecordIdentity | None
    evidence_binding_artifact_id: str | None
    evidence_binding_artifact_digest: str | None
    expected_subject_ids: tuple[str, ...]
    status: Literal["complete", "partial", "blocked"]
    candidate_profile_count: int
    candidate_profile_digests: tuple[str, ...]
    candidate_omissions: tuple[MaterializationOmission, ...]
    blockers: tuple[MaterializationBlocker, ...]
    attempt_digest: str = field(default="", init=False)

    def __post_init__(self) -> None:
        if self.contract_id != "rt_lnrna_reporter_response_materialization_attempt.v4":
            raise MetastudyContractError("materialization attempt contract_id changed")
        _required_text(self.experiment_id, label="materialization attempt experiment_id")
        if self.reader_record_identity is not None and not isinstance(
            self.reader_record_identity, ReaderRecordIdentity
        ):
            raise MetastudyContractError("materialization attempt Reader record identity must be typed or null")
        if (
            self.reader_record_identity is not None
            and self.reader_record_identity.reader_experiment_id != self.experiment_id
        ):
            raise MetastudyContractError("materialization attempt experiment identity mismatch")
        if (self.evidence_binding_artifact_id is None) != (self.evidence_binding_artifact_digest is None):
            raise MetastudyContractError("materialization attempt binding identity and digest must be paired")
        if self.evidence_binding_artifact_id is not None:
            _required_text(self.evidence_binding_artifact_id, label="materialization binding artifact id")
            _digest(self.evidence_binding_artifact_digest, label="materialization binding artifact digest")
        _unique_text(self.expected_subject_ids, label="expected_subject_ids", allow_empty=self.status == "blocked")
        if self.expected_subject_ids != tuple(sorted(self.expected_subject_ids)):
            raise MetastudyContractError("expected_subject_ids must use canonical subject order")
        if type(self.candidate_profile_count) is not int or self.candidate_profile_count < 0:
            raise MetastudyContractError("candidate_profile_count must be non-negative")
        for digest in self.candidate_profile_digests:
            _digest(digest, label="candidate profile digest")
        if len(self.candidate_profile_digests) != self.candidate_profile_count:
            raise MetastudyContractError("candidate profile count and digests differ")
        if len(set(self.candidate_profile_digests)) != len(self.candidate_profile_digests):
            raise MetastudyContractError("candidate profile digests must be unique")
        if self.candidate_profile_digests != tuple(sorted(self.candidate_profile_digests)):
            raise MetastudyContractError("candidate profile digests must use canonical digest order")
        if not all(isinstance(row, MaterializationBlocker) for row in self.blockers):
            raise MetastudyContractError("materialization attempt blockers must be typed")
        if not all(isinstance(row, MaterializationOmission) for row in self.candidate_omissions):
            raise MetastudyContractError("materialization attempt omissions must be typed")
        omission_order = tuple(
            sorted(
                self.candidate_omissions,
                key=lambda row: (row.subject_id, row.reduction_id, row.code),
            )
        )
        if self.candidate_omissions != omission_order or len(set(self.candidate_omissions)) != len(
            self.candidate_omissions
        ):
            raise MetastudyContractError("materialization attempt omissions must be unique and canonical")
        if self.status == "complete":
            if (
                self.reader_record_identity is None
                or self.evidence_binding_artifact_id is None
                or not self.expected_subject_ids
                or self.blockers
                or self.candidate_omissions
                or self.candidate_profile_count < 1
            ):
                raise MetastudyContractError(
                    "complete materialization requires source identities, profiles, and no issues"
                )
        elif self.status == "partial":
            if (
                self.reader_record_identity is None
                or self.evidence_binding_artifact_id is None
                or not self.expected_subject_ids
                or self.blockers
                or not self.candidate_omissions
                or self.candidate_profile_count < 1
            ):
                raise MetastudyContractError(
                    "partial materialization requires profiles and coordinate omissions without fatal blockers"
                )
        elif self.status == "blocked":
            if (
                not (self.blockers or self.candidate_omissions)
                or self.candidate_profile_count
                or self.candidate_profile_digests
            ):
                raise MetastudyContractError("blocked materialization requires issues and no profiles")
            if self.reader_record_identity is None and self.blockers != (
                MaterializationBlocker("reader_records_not_ready"),
            ):
                raise MetastudyContractError(
                    "a blocked attempt without a Reader record must report reader_records_not_ready"
                )
            if not self.blockers:
                expected_coordinates = {
                    (subject_id, f"window-{start:g}-{end:g}h")
                    for subject_id in self.expected_subject_ids
                    for start, end in DEFAULT_PROTOCOL.candidate_windows_h
                }
                omission_coordinates = {(row.subject_id, row.reduction_id) for row in self.candidate_omissions}
                if (
                    not expected_coordinates
                    or len(omission_coordinates) != len(self.candidate_omissions)
                    or omission_coordinates != expected_coordinates
                ):
                    raise MetastudyContractError(
                        "omission-only blocked materialization requires complete expected coordinate closure"
                    )
        else:
            raise MetastudyContractError("materialization attempt status is invalid")
        object.__setattr__(self, "attempt_digest", canonical_digest(materialization_attempt_payload(self, False)))


def materialization_attempt_from_payload(value: object, *, index: int) -> MaterializationAttemptReceipt:
    """Strictly parse one attempt receipt without granting source authority."""

    if not isinstance(value, Mapping):
        raise MetastudyContractError(f"materialization_attempts[{index}] must be an object")
    expected = {
        "contract_id",
        "experiment_id",
        "reader_record_identity",
        "evidence_binding_artifact_id",
        "evidence_binding_artifact_digest",
        "expected_subject_ids",
        "status",
        "candidate_profile_count",
        "candidate_profile_digests",
        "candidate_omissions",
        "blockers",
        "attempt_digest",
    }
    if set(value) != expected:
        raise MetastudyContractError(f"materialization_attempts[{index}] fields do not match the exact contract")
    identity = value["reader_record_identity"]
    if identity is not None:
        if not isinstance(identity, Mapping):
            raise MetastudyContractError(
                f"materialization_attempts[{index}].reader_record_identity must be an object or null"
            )
        identity_fields = {item.name for item in dataclass_fields(ReaderRecordIdentity)}
        if set(identity) != identity_fields:
            raise MetastudyContractError(f"materialization_attempts[{index}] Reader identity fields changed")
    blockers = value["blockers"]
    omissions = value["candidate_omissions"]
    digests = value["candidate_profile_digests"]
    subjects = value["expected_subject_ids"]
    if not all(isinstance(rows, (list, tuple)) for rows in (blockers, omissions, digests, subjects)):
        raise MetastudyContractError(f"materialization_attempts[{index}] array fields are malformed")
    parsed_blockers = _materialization_blockers_from_payload(blockers, index=index, field="blockers")
    parsed_omissions = _materialization_omissions_from_payload(omissions, index=index)
    attempt = MaterializationAttemptReceipt(
        contract_id=value["contract_id"],
        experiment_id=value["experiment_id"],
        reader_record_identity=ReaderRecordIdentity(**identity) if identity is not None else None,
        evidence_binding_artifact_id=value["evidence_binding_artifact_id"],
        evidence_binding_artifact_digest=value["evidence_binding_artifact_digest"],
        expected_subject_ids=tuple(subjects),
        status=value["status"],
        candidate_profile_count=value["candidate_profile_count"],
        candidate_profile_digests=tuple(digests),
        candidate_omissions=parsed_omissions,
        blockers=tuple(parsed_blockers),
    )
    if value["attempt_digest"] != attempt.attempt_digest:
        raise MetastudyContractError(f"materialization_attempts[{index}] digest mismatch")
    return attempt


def _materialization_blockers_from_payload(
    rows: object,
    *,
    index: int,
    field: str,
) -> tuple[MaterializationBlocker, ...]:
    assert isinstance(rows, (list, tuple))
    parsed: list[MaterializationBlocker] = []
    for blocker_index, blocker in enumerate(rows):
        if not isinstance(blocker, Mapping) or set(blocker) != {"code"}:
            raise MetastudyContractError(f"materialization_attempts[{index}].{field}[{blocker_index}] fields changed")
        parsed.append(MaterializationBlocker(**blocker))
    return tuple(parsed)


def _materialization_omissions_from_payload(
    rows: object,
    *,
    index: int,
) -> tuple[MaterializationOmission, ...]:
    assert isinstance(rows, (list, tuple))
    parsed: list[MaterializationOmission] = []
    for omission_index, omission in enumerate(rows):
        if not isinstance(omission, Mapping) or set(omission) != {"code", "subject_id", "reduction_id"}:
            raise MetastudyContractError(
                f"materialization_attempts[{index}].candidate_omissions[{omission_index}] fields changed"
            )
        parsed.append(MaterializationOmission(**omission))
    return tuple(parsed)


def materialization_attempt_payload(
    attempt: MaterializationAttemptReceipt,
    include_digest: bool = True,
) -> dict[str, object]:
    """Serialize one typed attempt receipt without trusting caller-authored fields."""

    if not isinstance(attempt, MaterializationAttemptReceipt):
        raise MetastudyContractError("attempt must be MaterializationAttemptReceipt")
    payload = asdict(attempt)
    if not include_digest:
        payload.pop("attempt_digest", None)
    return payload


__all__ = [
    "EvidenceReadiness",
    "MaterializationAttemptReceipt",
    "MaterializationBlocker",
    "MaterializationOmission",
    "ReaderRecordIdentity",
    "materialization_attempt_from_payload",
    "materialization_attempt_payload",
]
