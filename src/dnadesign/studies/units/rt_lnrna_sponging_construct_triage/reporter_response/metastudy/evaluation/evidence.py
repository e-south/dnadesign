"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/reporter_response/metastudy/evaluation/evidence.py

Evidence payloads, attempt closure, and cross-window source identity.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping

from ...profile.measurement import TimeWindowReduction
from ...serialization import profile_to_dict
from ..audits import profile_audit_payload, profile_source_identity_payload
from ..contracts._values import MetastudyContractError, canonical_digest
from ..contracts.decision import MetastudyDecision
from ..contracts.materialization import (
    EvidenceReadiness,
    MaterializationAttemptReceipt,
    materialization_attempt_payload,
)
from ..contracts.profile import ProfileEvidence
from ..contracts.protocol import DEFAULT_PROTOCOL, MetastudyProtocol, Window
from ..evidence_projection.contracts import (
    ProfileEvidenceProjection,
    profile_source_identity_projection,
)


def canonical_evidence_digest(
    rows: tuple[ProfileEvidence, ...],
    readiness: EvidenceReadiness,
    attempts: tuple[MaterializationAttemptReceipt, ...],
) -> str:
    profiles = [
        {"profile": profile_to_dict(row.profile), "audit": profile_audit_payload(row.audit)}
        for row in sorted(rows, key=lambda item: item.profile.profile_id)
    ]
    return canonical_digest(_evidence_payload(profiles=profiles, readiness=readiness, attempts=attempts))


def decision_evidence_payload(
    evidence: Iterable[ProfileEvidence],
    *,
    decision: MetastudyDecision,
) -> dict[str, object]:
    """Build the canonical evidence-bearing payload for one evaluated decision."""

    rows = tuple(evidence)
    require_attempt_ledger(decision.materialization_attempts, rows=rows, protocol=DEFAULT_PROTOCOL)
    profiles = [
        {"profile": profile_to_dict(row.profile), "audit": profile_audit_payload(row.audit)}
        for row in sorted(rows, key=lambda item: item.profile.profile_id)
    ]
    payload = _evidence_payload(
        profiles=profiles,
        readiness=decision.readiness,
        attempts=decision.materialization_attempts,
    )
    if canonical_digest(payload) != decision.evidence_digest:
        raise MetastudyContractError("publication evidence does not match the evaluated decision digest")
    return payload


def _evidence_payload(
    *,
    profiles: list[dict[str, object]],
    readiness: EvidenceReadiness,
    attempts: tuple[MaterializationAttemptReceipt, ...],
) -> dict[str, object]:
    return {
        "readiness_receipt_digest": readiness.receipt_digest,
        "materialization_attempts": [materialization_attempt_payload(row) for row in attempts],
        "profiles": profiles,
    }


def require_attempt_ledger(
    attempts: tuple[MaterializationAttemptReceipt, ...],
    *,
    rows: tuple[ProfileEvidence | ProfileEvidenceProjection, ...],
    protocol: MetastudyProtocol,
) -> None:
    """Require exact closure between source attempts and candidate evidence."""

    if not attempts or not all(isinstance(row, MaterializationAttemptReceipt) for row in attempts):
        raise MetastudyContractError("selection requires typed materialization attempts")
    if tuple(row.experiment_id for row in attempts) != protocol.planned_kinetic_experiment_ids:
        raise MetastudyContractError("materialization attempts must use canonical selected-experiment order")
    attempt_by_id = {row.experiment_id: row for row in attempts}
    evidence_by_experiment: dict[str, list[ProfileEvidence | ProfileEvidenceProjection]] = defaultdict(list)
    for row in rows:
        evidence_by_experiment[row.profile.provenance.reader_experiment_id].append(row)
    for experiment_id, attempt in attempt_by_id.items():
        experiment_rows = evidence_by_experiment.get(experiment_id, [])
        observed_digests = tuple(sorted(row.audit.profile_digest for row in experiment_rows))
        if attempt.status in {"complete", "partial"}:
            _require_complete_attempt(
                attempt,
                experiment_id=experiment_id,
                experiment_rows=experiment_rows,
                observed_digests=observed_digests,
                protocol=protocol,
            )
        if attempt.status == "blocked" and experiment_rows:
            raise MetastudyContractError(f"blocked materialization attempt cannot contribute profiles: {experiment_id}")


def _require_complete_attempt(
    attempt: MaterializationAttemptReceipt,
    *,
    experiment_id: str,
    experiment_rows: list[ProfileEvidence | ProfileEvidenceProjection],
    observed_digests: tuple[str, ...],
    protocol: MetastudyProtocol,
) -> None:
    if observed_digests != attempt.candidate_profile_digests:
        raise MetastudyContractError(f"materialization attempt profile digests differ for {experiment_id}")
    assert attempt.reader_record_identity is not None
    expected_identity = attempt.reader_record_identity
    expected_provenance = (
        expected_identity.reader_experiment_id,
        expected_identity.reader_protocol_id,
        expected_identity.reader_record_id,
        expected_identity.reader_record_kind,
        expected_identity.reader_record_schema_version,
        expected_identity.reader_record_revision,
        expected_identity.reader_record_revision_digest,
        expected_identity.reader_record_contract_id,
        expected_identity.reader_record_content_digest,
        expected_identity.reader_record_path,
    )
    for row in experiment_rows:
        provenance = row.profile.provenance
        observed_provenance = (
            provenance.reader_experiment_id,
            provenance.reader_protocol_id,
            provenance.reader_record_id,
            provenance.reader_record_kind,
            provenance.reader_record_schema_version,
            provenance.reader_record_revision,
            provenance.reader_record_revision_digest,
            provenance.reader_record_contract_id,
            provenance.reader_record_content_digest,
            provenance.reader_record_path,
        )
        if observed_provenance != expected_provenance:
            raise MetastudyContractError(
                f"materialization attempt Reader identity differs from profile provenance for {experiment_id}"
            )
        if (
            provenance.evidence_binding_artifact_id != attempt.evidence_binding_artifact_id
            or provenance.evidence_binding_artifact_digest != attempt.evidence_binding_artifact_digest
        ):
            raise MetastudyContractError(
                f"materialization attempt binding identity differs from profile provenance for {experiment_id}"
            )
    profile_coordinates = {(row.profile.subject_id, _profile_reduction_id(row)) for row in experiment_rows}
    omission_coordinates = {(row.subject_id, row.reduction_id) for row in attempt.candidate_omissions}
    expected_coordinates = {
        (subject_id, f"window-{start:g}-{end:g}h")
        for subject_id in attempt.expected_subject_ids
        for start, end in protocol.candidate_windows_h
    }
    if profile_coordinates & omission_coordinates or profile_coordinates | omission_coordinates != expected_coordinates:
        raise MetastudyContractError(
            f"materialization attempt candidate coordinate closure differs for {experiment_id}"
        )


def _profile_reduction_id(row: ProfileEvidence | ProfileEvidenceProjection) -> str:
    reduction = row.profile.reduction
    if not isinstance(reduction, TimeWindowReduction):
        raise MetastudyContractError("primary materialization attempts accept only time-window profiles")
    return f"window-{reduction.recorded_start_time_h:g}-{reduction.recorded_end_time_h:g}h"


def require_cross_window_identity(
    grouped: Mapping[Window, list[ProfileEvidence | ProfileEvidenceProjection]],
    *,
    readiness: EvidenceReadiness,
    protocol: MetastudyProtocol,
) -> set[tuple[str, str]]:
    """Return coordinates whose Reader source identity is stable across windows."""

    rosters: list[set[tuple[str, str]]] = []
    expected_provenance: dict[tuple[str, str], tuple[object, ...]] = {}
    ready_ids = set(readiness.ready_experiment_ids)
    planned_ids = set(protocol.planned_kinetic_experiment_ids)
    for window in protocol.candidate_windows_h:
        roster: set[tuple[str, str]] = set()
        for row in grouped[window]:
            profile = row.profile
            identity = (profile.provenance.reader_experiment_id, profile.subject_id)
            roster.add(identity)
            if identity[0] not in ready_ids or identity[0] not in planned_ids:
                raise MetastudyContractError("profile experiment identity is not a verified planned kinetic experiment")
            source_identity_payload = (
                profile_source_identity_payload(profile)
                if isinstance(row, ProfileEvidence)
                else profile_source_identity_projection(profile)
            )
            source_identity = tuple(source_identity_payload.items())
            prior = expected_provenance.setdefault(identity, source_identity)
            if prior != source_identity:
                raise MetastudyContractError("cross-window Reader provenance identity changed")
        rosters.append(roster)
    common = set.intersection(*rosters) if rosters else set()
    if not common:
        raise MetastudyContractError("candidate windows have no common experiment-subject coordinates")
    return common


__all__ = ["decision_evidence_payload"]
