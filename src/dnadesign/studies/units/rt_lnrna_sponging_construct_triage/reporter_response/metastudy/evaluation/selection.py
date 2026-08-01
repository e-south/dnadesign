"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/reporter_response/metastudy/evaluation/selection.py

Evidence validation and deterministic candidate-selection orchestration.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import replace

from ...profile.measurement import TimeWindowReduction
from ...temporal import window_temporal_policy_projection
from ..contracts._values import MetastudyContractError
from ..contracts.candidate import (
    CandidateEvaluation,
    candidate_quality_blockers,
    select_best_candidate,
)
from ..contracts.decision import DECISION_CONTRACT_ID, MetastudyDecision
from ..contracts.materialization import (
    EvidenceReadiness,
    MaterializationAttemptReceipt,
)
from ..contracts.profile import ProfileEvidence
from ..contracts.protocol import (
    DEFAULT_PROTOCOL,
    MetastudyProtocol,
    Window,
    protocol_digest,
)
from ..evidence_projection.contracts import ProfileEvidenceProjection
from .candidate import evaluate_candidate
from .comparability import profiles_are_selection_comparable
from .evidence import (
    canonical_evidence_digest,
    require_attempt_ledger,
    require_cross_window_identity,
)
from .readiness import decision_from_readiness


def evaluate_metastudy(
    evidence: Iterable[ProfileEvidence],
    *,
    readiness: EvidenceReadiness,
    attempts: Iterable[MaterializationAttemptReceipt],
    protocol: MetastudyProtocol = DEFAULT_PROTOCOL,
) -> MetastudyDecision:
    """Evaluate the predeclared primary cohort and select without a weighted score."""

    if not isinstance(readiness, EvidenceReadiness) or not readiness.is_receipt_validated:
        raise MetastudyContractError("readiness must come from readiness_from_receipt")
    ready_kinetic_ids = set(readiness.ready_experiment_ids) & set(protocol.planned_kinetic_experiment_ids)
    if len(ready_kinetic_ids) < protocol.minimum_kinetic_experiments:
        return decision_from_readiness(readiness, protocol=protocol)
    if not readiness.is_selection_authorized:
        raise MetastudyContractError("selection requires readiness returned by the owner-bound live bridge runner")
    rows = tuple(evidence)
    attempt_rows = tuple(attempts)
    return _evaluate_canonical_evidence(
        rows,
        readiness=readiness,
        attempts=attempt_rows,
        protocol=protocol,
        evidence_digest=canonical_evidence_digest(rows, readiness, attempt_rows),
        require_source_closure=True,
    )


def reevaluate_evidence_projection(
    evidence: Iterable[ProfileEvidenceProjection],
    *,
    readiness: EvidenceReadiness,
    attempts: Iterable[MaterializationAttemptReceipt],
    evidence_digest: str,
    protocol: MetastudyProtocol = DEFAULT_PROTOCOL,
) -> MetastudyDecision:
    """Repeat decision math from bundled content without claiming source authenticity."""

    if not isinstance(readiness, EvidenceReadiness):
        raise MetastudyContractError("publication readiness must be a typed structural projection")
    rows = tuple(evidence)
    attempt_rows = tuple(attempts)
    return _evaluate_canonical_evidence(
        rows,
        readiness=readiness,
        attempts=attempt_rows,
        protocol=protocol,
        evidence_digest=evidence_digest,
        require_source_closure=False,
    )


def _evaluate_canonical_evidence(
    rows: tuple[ProfileEvidence | ProfileEvidenceProjection, ...],
    *,
    readiness: EvidenceReadiness,
    attempts: tuple[MaterializationAttemptReceipt, ...],
    protocol: MetastudyProtocol,
    evidence_digest: str,
    require_source_closure: bool,
) -> MetastudyDecision:
    """Shared deterministic evaluation after live or publication-specific parsing."""

    anchor_subject_order = protocol.anchor_subject_order
    require_attempt_ledger(attempts, rows=rows, protocol=protocol)
    if not rows:
        raise MetastudyContractError("profile evidence must not be empty")
    grouped: dict[Window, list[ProfileEvidence | ProfileEvidenceProjection]] = defaultdict(list)
    for row in rows:
        expected_type = ProfileEvidence if require_source_closure else ProfileEvidenceProjection
        if not isinstance(row, expected_type):
            raise MetastudyContractError("evidence rows do not match the selected evaluation boundary")
        if require_source_closure and (
            row.audit.method_id != "canonical_profile_observation_audit_v1" or not row.audit.is_derivation_closed
        ):
            raise MetastudyContractError("selection requires derivation-closed canonical profile audits")
        if not require_source_closure and row.audit.method_id != "canonical_profile_observation_audit_v1":
            raise MetastudyContractError("publication evaluation requires canonical profile-observation audits")
        if row.audit.condition_ontology_digest != protocol.condition_ontology_digest:
            raise MetastudyContractError("profile evidence condition ontology does not match the protocol")
        if row.profile.observation_policy.digest != protocol.observation_policy_digest:
            raise MetastudyContractError("profile observation policy does not match the protocol")
        reduction = row.profile.reduction
        if not isinstance(reduction, TimeWindowReduction):
            raise MetastudyContractError("primary selection accepts only TimeWindowReduction profiles")
        window = (reduction.recorded_start_time_h, reduction.recorded_end_time_h)
        if window not in protocol.candidate_windows_h:
            raise MetastudyContractError(f"profile uses undeclared candidate window {window!r}")
        if (
            reduction.summary_statistic != protocol.time_summary_statistic
            or reduction.ratio_reduction_order != protocol.ratio_reduction_order
        ):
            raise MetastudyContractError("profile reduction semantics do not match the protocol")
        expected_temporal_policy = window_temporal_policy_projection(
            start_h=reduction.recorded_start_time_h,
            end_h=reduction.recorded_end_time_h,
            expected_cadence_h=reduction.expected_cadence_h,
        )
        if reduction.temporal_policy != expected_temporal_policy:
            raise MetastudyContractError("profile temporal policy projection does not match the canonical operator")
        if protocol.primary_dose_uM not in row.profile.dose_grid_uM:
            raise MetastudyContractError("primary selection profiles must contain the 500 uM cohort")
        if any(
            measurement.within_acquisition_reduction_statistic != protocol.within_acquisition_observation_reduction
            for measurement in row.profile.measurements
        ):
            raise MetastudyContractError("profiles must use median within-acquisition observation reduction")
        grouped[window].append(row)
    if set(grouped) != set(protocol.candidate_windows_h):
        raise MetastudyContractError("evidence must cover every predeclared candidate window exactly")
    common_coordinates = require_cross_window_identity(grouped, readiness=readiness, protocol=protocol)
    grouped = {
        window: [
            row
            for row in candidate_rows
            if (row.profile.provenance.reader_experiment_id, row.profile.subject_id) in common_coordinates
        ]
        for window, candidate_rows in grouped.items()
    }
    for candidate_rows in grouped.values():
        if not profiles_are_selection_comparable(candidate_rows):
            raise MetastudyContractError("candidate profiles fail exact comparability")

    evaluations = tuple(
        evaluate_candidate(tuple(grouped[window]), window=window, anchors=anchor_subject_order, protocol=protocol)
        for window in protocol.candidate_windows_h
    )
    quality_blockers = candidate_quality_blockers(evaluations)
    if quality_blockers:
        return _blocked(
            readiness,
            protocol,
            attempts,
            evaluations,
            evidence_digest=evidence_digest,
            blockers=quality_blockers,
        )
    selected = select_best_candidate(evaluations)
    if selected is None:
        return _blocked(
            readiness,
            protocol,
            attempts,
            evaluations,
            evidence_digest=evidence_digest,
            blockers=tuple(
                f"window_{row.reduction[0]:g}_{row.reduction[1]:g}:" + ",".join(row.blockers) for row in evaluations
            ),
        )
    experiment_ids = sorted(
        {row.profile.provenance.reader_experiment_id for candidate_rows in grouped.values() for row in candidate_rows}
    )
    stable = 0
    failed_fold = False
    selected_index = protocol.candidate_windows_h.index(selected.reduction)
    for omitted in experiment_ids:
        loo_evaluations = tuple(
            evaluate_candidate(
                tuple(row for row in grouped[window] if row.profile.provenance.reader_experiment_id != omitted),
                window=window,
                anchors=anchor_subject_order,
                protocol=protocol,
                minimum_experiments=protocol.minimum_kinetic_experiments - 1,
            )
            for window in protocol.candidate_windows_h
        )
        loo_selected = select_best_candidate(loo_evaluations)
        if loo_selected is None:
            failed_fold = True
            continue
        loo_index = protocol.candidate_windows_h.index(loo_selected.reduction)
        stable += int(abs(loo_index - selected_index) <= 1)
    loo_fraction = stable / len(experiment_ids) if experiment_ids else 0.0
    selected_limitations = list(selected.limitations)
    if failed_fold:
        selected_limitations.append("loo_fold_without_eligible_candidate")
    if loo_fraction < protocol.loo_same_or_adjacent_target_fraction:
        selected_limitations.append("loo_choice_same_or_adjacent_below_75_percent")
    evaluations = tuple(
        replace(
            row,
            loo_same_or_adjacent_fraction=loo_fraction,
            limitations=tuple(dict.fromkeys(selected_limitations)),
        )
        if row.reduction == selected.reduction
        else row
        for row in evaluations
    )
    selected = next(row for row in evaluations if row.reduction == selected.reduction)
    decision_limitations = [
        "retrospective_calibration_cohort",
        "growth_phase_rule_requires_external_replication",
        "acquisition_projection_is_descriptive_only",
        "study_side_blank_correction_not_claimed",
        *selected.limitations,
    ]
    if any(attempt.candidate_omissions for attempt in attempts):
        decision_limitations.append("subject_window_omissions_present")
    return MetastudyDecision._from_canonical_evaluation(
        contract_id=DECISION_CONTRACT_ID,
        protocol_id=protocol.protocol_id,
        condition_ontology_digest=protocol.condition_ontology_digest,
        status="selected",
        selection_use="descriptive_comparison",
        evidence_grade="provisional_descriptive",
        selected_reduction=selected.reduction,
        blockers=(),
        limitations=tuple(dict.fromkeys(decision_limitations)),
        policy_digest=protocol_digest(protocol),
        evidence_digest=evidence_digest,
        readiness=readiness,
        evaluations=evaluations,
        materialization_attempts=attempts,
    )


def _blocked(
    readiness: EvidenceReadiness,
    protocol: MetastudyProtocol,
    attempts: tuple[MaterializationAttemptReceipt, ...],
    evaluations: tuple[CandidateEvaluation, ...],
    *,
    evidence_digest: str,
    blockers: tuple[str, ...],
) -> MetastudyDecision:
    attempt_blockers = tuple(
        f"materialization:{attempt.experiment_id}:source:{blocker.code}"
        for attempt in attempts
        if attempt.status == "blocked"
        for blocker in attempt.blockers
    )
    return MetastudyDecision(
        contract_id=DECISION_CONTRACT_ID,
        protocol_id=protocol.protocol_id,
        condition_ontology_digest=protocol.condition_ontology_digest,
        status="blocked",
        selection_use="descriptive_comparison",
        evidence_grade="none",
        selected_reduction=None,
        blockers=tuple(dict.fromkeys((*attempt_blockers, *blockers))),
        limitations=(),
        policy_digest=protocol_digest(protocol),
        evidence_digest=evidence_digest,
        readiness=readiness,
        evaluations=evaluations,
        materialization_attempts=attempts,
    )


__all__ = [
    "evaluate_metastudy",
    "reevaluate_evidence_projection",
]
