"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/reporter_response/metastudy/contracts/decision.py

Selected-or-blocked meta-study decision model and invariants.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Literal

from ._values import MetastudyContractError, _digest, _unique_text
from .candidate import (
    CandidateEvaluation,
    candidate_meets_selection_gates,
    select_best_candidate,
)
from .materialization import EvidenceReadiness, MaterializationAttemptReceipt
from .protocol import DEFAULT_PROTOCOL, PROTOCOL_ID, Window, protocol_digest

DECISION_CONTRACT_ID = "rt_lnrna_reporter_response_metastudy_decision.v4"
DecisionStatus = Literal["selected", "blocked"]
_SELECTION_CLOSURE_TOKEN = object()


@dataclass(frozen=True, slots=True)
class MetastudyDecision:
    """Typed selected-or-blocked result with nullable selected reduction."""

    contract_id: str
    protocol_id: str
    condition_ontology_digest: str
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
        _digest(self.condition_ontology_digest, label="condition_ontology_digest")
        if self.status not in {"selected", "blocked"}:
            raise MetastudyContractError("decision status must be selected or blocked")
        if self.selection_use != "descriptive_comparison":
            raise MetastudyContractError("meta-study selection use must remain descriptive_comparison")
        _digest(self.policy_digest, label="policy_digest")
        _digest(self.evidence_digest, label="evidence_digest")
        expected_protocol = replace(
            DEFAULT_PROTOCOL,
            condition_ontology_digest=self.condition_ontology_digest,
        )
        if self.policy_digest != protocol_digest(expected_protocol):
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
            _validate_selected_projection(
                readiness=self.readiness,
                evaluations=self.evaluations,
                selected_reduction=self.selected_reduction,
            )
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
    if not candidate_meets_selection_gates(selected, protocol=DEFAULT_PROTOCOL):
        raise MetastudyContractError("selected evaluation does not satisfy descriptive support and phase gates")
    expected = select_best_candidate(evaluations)
    if expected is None or expected.reduction != selected_reduction:
        raise MetastudyContractError("selected reduction does not match the lexicographic evaluation winner")


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
    "MetastudyDecision",
    "decision_is_evidence_bearing",
]
