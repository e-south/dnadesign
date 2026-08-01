"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/reporter_response/metastudy/contracts/decision.py

Evaluation, objective-readiness, and meta-study decision contracts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Literal

from ._values import MetastudyContractError, _digest, _nonnegative, _unique_text
from .materialization import (
    EvidenceReadiness,
    MaterializationAttemptReceipt,
    materialization_attempt_from_payload,
)
from .protocol import DEFAULT_PROTOCOL, PROTOCOL_ID, Window, protocol_digest

DECISION_CONTRACT_ID = "rt_lnrna_reporter_response_metastudy_decision.v4"
DecisionStatus = Literal["selected", "blocked"]
_SELECTION_CLOSURE_TOKEN = object()


@dataclass(frozen=True, slots=True)
class SensitivityEvaluation:
    """Digest-bound non-selectable sensitivity evidence summary."""

    kind: Literal["dose", "endpoint", "centered_window"]
    value: float
    profile_count: int
    evidence_digest: str
    selectable: Literal[False] = False

    def __post_init__(self) -> None:
        if self.kind not in {"dose", "endpoint", "centered_window"}:
            raise MetastudyContractError("sensitivity kind is undeclared")
        _nonnegative(self.value, label="sensitivity value")
        if isinstance(self.profile_count, bool) or not isinstance(self.profile_count, int) or self.profile_count < 1:
            raise MetastudyContractError("sensitivity profile_count must be positive")
        _digest(self.evidence_digest, label="sensitivity evidence_digest")
        if self.selectable is not False:
            raise MetastudyContractError("sensitivity evaluations are never selectable")


@dataclass(frozen=True, slots=True)
class CandidateEvaluation:
    """Derived metrics for one selectable primary-cohort window."""

    reduction: Window
    eligible_experiment_count: int
    worst_experiment_control_separation: float
    repeated_anchor_drift: float
    within_acquisition_observation_range: float
    growth_phase_start: float
    growth_phase_end: float
    anchor_ordered_acquisition_count: int
    co_measured_anchor_acquisition_count: int
    loo_same_or_adjacent_fraction: float
    eligible: bool
    blockers: tuple[str, ...]
    limitations: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.reduction not in DEFAULT_PROTOCOL.candidate_windows_h:
            raise MetastudyContractError("candidate evaluation reduction is undeclared")
        for name in (
            "eligible_experiment_count",
            "anchor_ordered_acquisition_count",
            "co_measured_anchor_acquisition_count",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise MetastudyContractError(f"{name} must be a non-negative integer")
        for name in (
            "worst_experiment_control_separation",
            "repeated_anchor_drift",
            "within_acquisition_observation_range",
            "growth_phase_start",
            "growth_phase_end",
            "loo_same_or_adjacent_fraction",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise MetastudyContractError(f"{name} must be finite")
        if not 0.0 <= self.loo_same_or_adjacent_fraction <= 1.0:
            raise MetastudyContractError("loo_same_or_adjacent_fraction must be between zero and one")
        _unique_text(self.blockers, label="candidate blockers", allow_empty=self.eligible)
        if self.eligible and self.blockers:
            raise MetastudyContractError("eligible candidate cannot contain blockers")
        if not self.eligible and not self.blockers:
            raise MetastudyContractError("ineligible candidate requires blockers")
        _unique_text(self.limitations, label="candidate limitations", allow_empty=True)


@dataclass(frozen=True, slots=True)
class MetastudyDecision:
    """Typed selected-or-blocked result with nullable selected reduction."""

    contract_id: str
    protocol_id: str
    status: DecisionStatus
    selection_use: Literal["descriptive_comparison"]
    evidence_grade: Literal["provisional_descriptive", "none"]
    selected_reduction: Window | None
    blockers: tuple[str, ...]
    limitations: tuple[str, ...]
    policy_digest: str
    evidence_digest: str
    readiness: EvidenceReadiness
    evaluations: tuple[CandidateEvaluation, ...]
    materialization_attempts: tuple[MaterializationAttemptReceipt, ...] = ()
    _selection_closure: object | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.contract_id != DECISION_CONTRACT_ID or self.protocol_id != PROTOCOL_ID:
            raise MetastudyContractError("decision contract or protocol identity changed")
        if self.status not in {"selected", "blocked"}:
            raise MetastudyContractError("decision status must be selected or blocked")
        if self.selection_use != "descriptive_comparison":
            raise MetastudyContractError("meta-study selection use must remain descriptive_comparison")
        _digest(self.policy_digest, label="policy_digest")
        _digest(self.evidence_digest, label="evidence_digest")
        if self.policy_digest != protocol_digest():
            raise MetastudyContractError("decision policy_digest does not match the predeclared protocol")
        _unique_text(self.blockers, label="blockers", allow_empty=self.status == "selected")
        _unique_text(self.limitations, label="limitations", allow_empty=True)
        if not isinstance(self.materialization_attempts, tuple) or not all(
            isinstance(row, MaterializationAttemptReceipt) for row in self.materialization_attempts
        ):
            raise MetastudyContractError("decision materialization_attempts must be a typed tuple")
        evidence_bearing = decision_is_evidence_bearing(self)
        if evidence_bearing:
            _validate_evaluated_decision_order(
                evaluations=self.evaluations,
                attempts=self.materialization_attempts,
            )
        elif self.status == "selected":
            raise MetastudyContractError("selected decisions must be evidence-bearing")
        if self.status == "selected":
            if self._selection_closure is not _SELECTION_CLOSURE_TOKEN:
                raise MetastudyContractError("selected decisions must be returned by canonical evaluation")
            if self.selected_reduction not in DEFAULT_PROTOCOL.candidate_windows_h or self.blockers:
                raise MetastudyContractError("selected decision requires one declared reduction and no blockers")
            if self.evidence_grade != "provisional_descriptive":
                raise MetastudyContractError("selected decisions are provisional descriptive recommendations")
            _validate_selected_decision(self)
        elif self.selected_reduction is not None or not self.blockers or self.evidence_grade != "none":
            raise MetastudyContractError("blocked decision requires no reduction, no evidence grade, and blockers")

    @classmethod
    def _from_canonical_evaluation(cls, **values: object) -> MetastudyDecision:
        decision = cls.__new__(cls)
        for name, value in values.items():
            object.__setattr__(decision, name, value)
        object.__setattr__(decision, "_selection_closure", _SELECTION_CLOSURE_TOKEN)
        decision.__post_init__()
        return decision


@dataclass(frozen=True, slots=True)
class ObjectiveReadiness:
    """Independent readiness of a descriptive reduction for optimization use."""

    contract_id: Literal["rt_lnrna_reporter_response_objective_readiness.v3"]
    status: Literal["ready", "blocked"]
    objective_id: str | None
    blockers: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.contract_id != "rt_lnrna_reporter_response_objective_readiness.v3":
            raise MetastudyContractError("objective-readiness contract_id changed")
        if (
            not isinstance(self.blockers, tuple)
            or len(self.blockers) != len(set(self.blockers))
            or any(not isinstance(value, str) or not value.strip() or value != value.strip() for value in self.blockers)
        ):
            raise MetastudyContractError("objective-readiness blockers must be unique trimmed strings")
        if self.status == "ready":
            if not isinstance(self.objective_id, str) or not self.objective_id.strip():
                raise MetastudyContractError("ready objective readiness requires an objective_id")
            if self.blockers:
                raise MetastudyContractError("ready objective readiness cannot contain blockers")
        elif self.status == "blocked":
            if self.objective_id is not None or not self.blockers:
                raise MetastudyContractError("blocked objective readiness requires no objective and explicit blockers")
        else:
            raise MetastudyContractError("objective-readiness status is invalid")


DEFAULT_OBJECTIVE_READINESS = ObjectiveReadiness(
    contract_id="rt_lnrna_reporter_response_objective_readiness.v3",
    status="blocked",
    objective_id=None,
    blockers=(
        "constrained_objective_not_defined",
        "biological_replicate_uncertainty_not_estimable",
        "od_linearity_not_validated",
    ),
)


def objective_readiness_from_payload(value: object) -> ObjectiveReadiness:
    """Parse one exact JSON/YAML objective-readiness projection."""

    expected = {"contract_id", "status", "objective_id", "blockers"}
    if not isinstance(value, Mapping) or set(value) != expected:
        raise MetastudyContractError("objective-readiness fields do not match the exact contract")
    blockers = value["blockers"]
    if not isinstance(blockers, (list, tuple)):
        raise MetastudyContractError("objective-readiness blockers must be an array")
    return ObjectiveReadiness(
        contract_id=value["contract_id"],
        status=value["status"],
        objective_id=value["objective_id"],
        blockers=tuple(blockers),
    )


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

    expected = {
        "contract_id",
        "protocol_id",
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
    if not isinstance(payload, Mapping) or set(payload) != expected:
        raise MetastudyContractError("decision payload fields do not match the exact contract")
    if payload["contract_id"] != DECISION_CONTRACT_ID or payload["protocol_id"] != PROTOCOL_ID:
        raise MetastudyContractError("decision payload identity changed")
    status = payload["status"]
    blockers = payload["blockers"]
    limitations = payload["limitations"]
    if not isinstance(blockers, (list, tuple)) or not isinstance(limitations, (list, tuple)):
        raise MetastudyContractError("decision blockers and limitations must be arrays")
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
    else:
        raise MetastudyContractError("decision status must be selected or blocked")
    _digest(payload["policy_digest"], label="policy_digest")
    _digest(payload["evidence_digest"], label="evidence_digest")
    if payload["policy_digest"] != protocol_digest():
        raise MetastudyContractError("decision policy_digest does not match the predeclared protocol")
    readiness = payload["readiness"]
    evaluations = payload["evaluations"]
    attempts = payload["materialization_attempts"]
    if (
        not isinstance(readiness, Mapping)
        or not isinstance(evaluations, (list, tuple))
        or not isinstance(attempts, (list, tuple))
    ):
        raise MetastudyContractError("decision readiness, evaluations, and attempts must be structured")
    readiness_fields = {
        "selected_experiment_count",
        "ready_experiment_count",
        "ready_experiment_ids",
        "blocked_experiment_ids",
        "receipt_digest",
    }
    if (
        set(readiness) != readiness_fields
        or not isinstance(readiness["ready_experiment_ids"], (list, tuple))
        or not isinstance(readiness["blocked_experiment_ids"], (list, tuple))
    ):
        raise MetastudyContractError("decision readiness fields do not match the exact contract")
    parsed_readiness = EvidenceReadiness(
        selected_experiment_count=readiness["selected_experiment_count"],
        ready_experiment_count=readiness["ready_experiment_count"],
        ready_experiment_ids=tuple(readiness["ready_experiment_ids"]),
        blocked_experiment_ids=tuple(readiness["blocked_experiment_ids"]),
        receipt_digest=readiness["receipt_digest"],
    )
    evaluation_fields = {
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
    parsed_evaluations: list[CandidateEvaluation] = []
    for index, row in enumerate(evaluations):
        if not isinstance(row, Mapping) or set(row) != evaluation_fields:
            raise MetastudyContractError(f"evaluations[{index}] fields do not match the exact contract")
        reduction = row["reduction"]
        row_blockers = row["blockers"]
        row_limitations = row["limitations"]
        if (
            not isinstance(reduction, (list, tuple))
            or not isinstance(row_blockers, (list, tuple))
            or not isinstance(row_limitations, (list, tuple))
        ):
            raise MetastudyContractError(f"evaluations[{index}] array fields are malformed")
        parsed_evaluations.append(
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
                blockers=tuple(row_blockers),
                limitations=tuple(row_limitations),
            )
        )
    reduction_payload = payload["selected_reduction"]
    selected_reduction = tuple(reduction_payload) if isinstance(reduction_payload, (list, tuple)) else None
    parsed_attempts = tuple(
        materialization_attempt_from_payload(row, index=index) for index, row in enumerate(attempts)
    )
    evidence_bearing = decision_is_evidence_bearing(
        {
            "evaluations": parsed_evaluations,
            "materialization_attempts": parsed_attempts,
        }
    )
    if evidence_bearing:
        _validate_evaluated_decision_order(
            evaluations=tuple(parsed_evaluations),
            attempts=parsed_attempts,
        )
    elif status == "selected":
        raise MetastudyContractError("selected decisions must be evidence-bearing")
    if status == "selected":
        assert selected_reduction is not None
        _validate_selected_projection(
            readiness=parsed_readiness,
            evaluations=tuple(parsed_evaluations),
            selected_reduction=selected_reduction,
        )
    else:
        MetastudyDecision(
            contract_id=payload["contract_id"],
            protocol_id=payload["protocol_id"],
            status=status,
            selection_use=payload["selection_use"],
            evidence_grade=payload["evidence_grade"],
            selected_reduction=selected_reduction,
            blockers=tuple(blockers),
            limitations=tuple(limitations),
            policy_digest=payload["policy_digest"],
            evidence_digest=payload["evidence_digest"],
            readiness=parsed_readiness,
            evaluations=tuple(parsed_evaluations),
            materialization_attempts=parsed_attempts,
        )


def _validate_selected_decision(decision: MetastudyDecision) -> None:
    _validate_selected_projection(
        readiness=decision.readiness,
        evaluations=decision.evaluations,
        selected_reduction=decision.selected_reduction,
    )


def _validate_selected_projection(
    *,
    readiness: EvidenceReadiness,
    evaluations: tuple[CandidateEvaluation, ...],
    selected_reduction: Window | None,
) -> None:
    """Validate a selected projection without minting canonical-evaluation authority."""

    ready_kinetic = set(readiness.ready_experiment_ids) & set(DEFAULT_PROTOCOL.planned_kinetic_experiment_ids)
    if len(ready_kinetic) < DEFAULT_PROTOCOL.minimum_kinetic_experiments:
        raise MetastudyContractError("selected decision requires at least 7 verified kinetic experiment identities")
    reductions = tuple(row.reduction for row in evaluations)
    if len(reductions) != len(DEFAULT_PROTOCOL.candidate_windows_h) or set(reductions) != set(
        DEFAULT_PROTOCOL.candidate_windows_h
    ):
        raise MetastudyContractError("selected decision requires exactly one evaluation per declared candidate window")
    if any(row.eligible_experiment_count == 0 for row in evaluations):
        raise MetastudyContractError("selected decision cannot contain zero experiment support")
    if selected_reduction not in DEFAULT_PROTOCOL.candidate_windows_h:
        raise MetastudyContractError("selected reduction must be one declared candidate window")
    selected = next(row for row in evaluations if row.reduction == selected_reduction)
    if (
        not selected.eligible
        or selected.eligible_experiment_count < DEFAULT_PROTOCOL.minimum_kinetic_experiments
        or selected.worst_experiment_control_separation <= 0.0
        or selected.growth_phase_start < DEFAULT_PROTOCOL.growth_phase_start_minimum
        or not DEFAULT_PROTOCOL.growth_phase_end_minimum
        <= selected.growth_phase_end
        <= DEFAULT_PROTOCOL.growth_phase_end_maximum
    ):
        raise MetastudyContractError("selected evaluation does not satisfy descriptive support and phase gates")
    eligible = tuple(row for row in evaluations if row.eligible)
    expected = min(
        eligible,
        key=lambda row: (
            -row.worst_experiment_control_separation,
            (
                float("inf")
                if "repeated_reference_drift_not_estimable" in row.limitations
                else row.repeated_anchor_drift
            ),
            row.within_acquisition_observation_range,
            row.reduction[1],
        ),
    )
    if expected.reduction != selected_reduction:
        raise MetastudyContractError("selected reduction does not match the lexicographic evaluation winner")


def decision_is_evidence_bearing(decision: MetastudyDecision | Mapping[str, object]) -> bool:
    """Return whether a decision contains a complete primary-evidence evaluation."""

    if isinstance(decision, MetastudyDecision):
        evaluations = decision.evaluations
        attempts = decision.materialization_attempts
    elif isinstance(decision, Mapping):
        evaluations = decision.get("evaluations")
        attempts = decision.get("materialization_attempts")
        if not isinstance(evaluations, (list, tuple)) or not isinstance(attempts, (list, tuple)):
            raise MetastudyContractError("decision evaluations and attempts must be arrays")
    else:
        raise MetastudyContractError("decision must be typed or structured")
    if bool(evaluations) != bool(attempts):
        raise MetastudyContractError("decision evaluations and attempts must be jointly empty or complete")
    return bool(evaluations)


def _validate_evaluated_decision_order(
    *,
    evaluations: tuple[CandidateEvaluation, ...],
    attempts: tuple[MaterializationAttemptReceipt, ...],
) -> None:
    if tuple(row.reduction for row in evaluations) != DEFAULT_PROTOCOL.candidate_windows_h:
        raise MetastudyContractError("evaluated decisions must use canonical candidate-window order")
    if tuple(row.experiment_id for row in attempts) != DEFAULT_PROTOCOL.planned_kinetic_experiment_ids:
        raise MetastudyContractError("evaluated decisions must use canonical materialization-attempt order")


__all__ = [
    "DECISION_CONTRACT_ID",
    "DEFAULT_OBJECTIVE_READINESS",
    "CandidateEvaluation",
    "MetastudyDecision",
    "ObjectiveReadiness",
    "SensitivityEvaluation",
    "decision_is_evidence_bearing",
    "decision_to_dict",
    "objective_readiness_from_payload",
    "validate_decision_payload",
]
