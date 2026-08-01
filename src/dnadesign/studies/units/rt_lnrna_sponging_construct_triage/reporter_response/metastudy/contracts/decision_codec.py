"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/reporter_response/metastudy/contracts/decision_codec.py

Strict serialization and parsing for meta-study decisions.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, replace

from ._values import MetastudyContractError, _digest, _unique_text
from .candidate import CandidateEvaluation
from .decision import (
    DECISION_CONTRACT_ID,
    MetastudyDecision,
    _validate_evaluated_decision_order,
    _validate_selected_projection,
    decision_is_evidence_bearing,
)
from .materialization import (
    EvidenceReadiness,
    materialization_attempt_from_payload,
)
from .protocol import DEFAULT_PROTOCOL, PROTOCOL_ID, protocol_digest

_DECISION_FIELDS = {
    "contract_id",
    "protocol_id",
    "condition_ontology_digest",
    "status",
    "selection_use",
    "evidence_grade",
    "selected_reduction",
    "blockers",
    "limitations",
    "policy_digest",
    "evidence_digest",
    "readiness",
    "evaluations",
    "materialization_attempts",
}
_EVALUATION_FIELDS = {
    "reduction",
    "eligible_experiment_count",
    "worst_experiment_control_separation",
    "repeated_anchor_drift",
    "within_acquisition_observation_range",
    "growth_phase_start",
    "growth_phase_end",
    "anchor_ordered_acquisition_count",
    "co_measured_anchor_acquisition_count",
    "loo_same_or_adjacent_fraction",
    "eligible",
    "blockers",
    "limitations",
}


def decision_to_dict(decision: MetastudyDecision) -> dict[str, object]:
    """Serialize and revalidate a decision as strict JSON data."""

    if not isinstance(decision, MetastudyDecision):
        raise MetastudyContractError("decision must be MetastudyDecision")
    payload = asdict(decision)
    payload.pop("_selection_closure", None)
    readiness = payload.get("readiness")
    if isinstance(readiness, dict):
        readiness.pop("_receipt_closure", None)
        readiness.pop("_owner_bridge_closure", None)
    validate_decision_payload(payload)
    return payload


def validate_decision_payload(payload: Mapping[str, object]) -> None:
    """Fail closed on unknown, missing, or internally inconsistent decision fields."""

    if not isinstance(payload, Mapping) or set(payload) != _DECISION_FIELDS:
        raise MetastudyContractError("decision payload fields do not match the exact contract")
    if payload["contract_id"] != DECISION_CONTRACT_ID or payload["protocol_id"] != PROTOCOL_ID:
        raise MetastudyContractError("decision payload identity changed")
    status = payload["status"]
    blockers = payload["blockers"]
    limitations = payload["limitations"]
    if not isinstance(blockers, (list, tuple)) or not isinstance(limitations, (list, tuple)):
        raise MetastudyContractError("decision blockers and limitations must be arrays")
    if status not in {"blocked", "selected"}:
        raise MetastudyContractError("decision status must be selected or blocked")
    blocker_rows = tuple(blockers)
    limitation_rows = tuple(limitations)
    _unique_text(blocker_rows, label="blockers", allow_empty=status == "selected")
    _unique_text(limitation_rows, label="limitations", allow_empty=True)
    if payload["selection_use"] != "descriptive_comparison":
        raise MetastudyContractError("decision selection_use changed")
    if status == "blocked":
        if payload["selected_reduction"] is not None or not blockers or payload["evidence_grade"] != "none":
            raise MetastudyContractError("blocked decision requires no reduction, no grade, and explicit blockers")
    elif status == "selected":
        if (
            not isinstance(payload["selected_reduction"], (list, tuple))
            or blockers
            or payload["evidence_grade"] != "provisional_descriptive"
        ):
            raise MetastudyContractError("selected decision requires one provisional descriptive reduction")
    _validate_protocol_identity(payload)
    parsed_readiness = _readiness_from_payload(payload["readiness"])
    parsed_evaluations = _evaluations_from_payload(payload["evaluations"])
    attempts = payload["materialization_attempts"]
    if not isinstance(attempts, (list, tuple)):
        raise MetastudyContractError("decision materialization_attempts must be an array")
    parsed_attempts = tuple(
        materialization_attempt_from_payload(row, index=index) for index, row in enumerate(attempts)
    )
    reduction_payload = payload["selected_reduction"]
    selected_reduction = tuple(reduction_payload) if isinstance(reduction_payload, (list, tuple)) else None
    evidence_bearing = decision_is_evidence_bearing(
        {
            "evaluations": parsed_evaluations,
            "materialization_attempts": parsed_attempts,
        }
    )
    if evidence_bearing:
        _validate_evaluated_decision_order(
            evaluations=parsed_evaluations,
            attempts=parsed_attempts,
        )
    elif status == "selected":
        raise MetastudyContractError("selected decisions must be evidence-bearing")
    if status == "selected":
        assert selected_reduction is not None
        _validate_selected_projection(
            readiness=parsed_readiness,
            evaluations=parsed_evaluations,
            selected_reduction=selected_reduction,
        )
        return
    MetastudyDecision(
        contract_id=payload["contract_id"],
        protocol_id=payload["protocol_id"],
        condition_ontology_digest=payload["condition_ontology_digest"],
        status=status,
        selection_use=payload["selection_use"],
        evidence_grade=payload["evidence_grade"],
        selected_reduction=selected_reduction,
        blockers=blocker_rows,
        limitations=limitation_rows,
        policy_digest=payload["policy_digest"],
        evidence_digest=payload["evidence_digest"],
        readiness=parsed_readiness,
        evaluations=parsed_evaluations,
        materialization_attempts=parsed_attempts,
    )


def _validate_protocol_identity(payload: Mapping[str, object]) -> None:
    _digest(payload["policy_digest"], label="policy_digest")
    _digest(payload["evidence_digest"], label="evidence_digest")
    condition_ontology_digest = payload["condition_ontology_digest"]
    _digest(condition_ontology_digest, label="condition_ontology_digest")
    expected_protocol = replace(
        DEFAULT_PROTOCOL,
        condition_ontology_digest=condition_ontology_digest,
    )
    if payload["policy_digest"] != protocol_digest(expected_protocol):
        raise MetastudyContractError("decision policy_digest does not match the predeclared protocol")


def _readiness_from_payload(value: object) -> EvidenceReadiness:
    fields = {
        "selected_experiment_count",
        "ready_experiment_count",
        "ready_experiment_ids",
        "blocked_experiment_ids",
        "receipt_digest",
    }
    if (
        not isinstance(value, Mapping)
        or set(value) != fields
        or not isinstance(value["ready_experiment_ids"], (list, tuple))
        or not isinstance(value["blocked_experiment_ids"], (list, tuple))
    ):
        raise MetastudyContractError("decision readiness fields do not match the exact contract")
    return EvidenceReadiness(
        selected_experiment_count=value["selected_experiment_count"],
        ready_experiment_count=value["ready_experiment_count"],
        ready_experiment_ids=tuple(value["ready_experiment_ids"]),
        blocked_experiment_ids=tuple(value["blocked_experiment_ids"]),
        receipt_digest=value["receipt_digest"],
    )


def _evaluations_from_payload(value: object) -> tuple[CandidateEvaluation, ...]:
    if not isinstance(value, (list, tuple)):
        raise MetastudyContractError("decision evaluations must be an array")
    parsed: list[CandidateEvaluation] = []
    for index, row in enumerate(value):
        if not isinstance(row, Mapping) or set(row) != _EVALUATION_FIELDS:
            raise MetastudyContractError(f"evaluations[{index}] fields do not match the exact contract")
        reduction = row["reduction"]
        blockers = row["blockers"]
        limitations = row["limitations"]
        if (
            not isinstance(reduction, (list, tuple))
            or not isinstance(blockers, (list, tuple))
            or not isinstance(limitations, (list, tuple))
        ):
            raise MetastudyContractError(f"evaluations[{index}] array fields are malformed")
        parsed.append(
            CandidateEvaluation(
                reduction=tuple(reduction),
                eligible_experiment_count=row["eligible_experiment_count"],
                worst_experiment_control_separation=row["worst_experiment_control_separation"],
                repeated_anchor_drift=row["repeated_anchor_drift"],
                within_acquisition_observation_range=row["within_acquisition_observation_range"],
                growth_phase_start=row["growth_phase_start"],
                growth_phase_end=row["growth_phase_end"],
                anchor_ordered_acquisition_count=row["anchor_ordered_acquisition_count"],
                co_measured_anchor_acquisition_count=row["co_measured_anchor_acquisition_count"],
                loo_same_or_adjacent_fraction=row["loo_same_or_adjacent_fraction"],
                eligible=row["eligible"],
                blockers=tuple(blockers),
                limitations=tuple(limitations),
            )
        )
    return tuple(parsed)


__all__ = ["decision_to_dict", "validate_decision_payload"]
