"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/reporter_response/metastudy/evaluation/selection.py

Candidate evaluation and lexicographic window selection.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import replace

from ...measurement_profile import ReporterMeasurementProfile
from ...profile import ReporterResponseProfile, TimeWindowReduction
from ...temporal import window_temporal_policy_projection
from ..contracts._values import MetastudyContractError
from ..contracts.decision import (
    DECISION_CONTRACT_ID,
    CandidateEvaluation,
    MetastudyDecision,
)
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
from ..evidence_projection import ProfileContentProjection, ProfileEvidenceProjection
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
        _evaluate_candidate(tuple(grouped[window]), window=window, anchors=anchor_subject_order, protocol=protocol)
        for window in protocol.candidate_windows_h
    )
    quality_blockers = tuple(
        dict.fromkeys(
            blocker
            for evaluation in evaluations
            for blocker in evaluation.blockers
            if blocker
            in {
                "required_observation_count_zero",
                "observation_overflow_detected",
                "observation_clipping_detected",
            }
        )
    )
    if quality_blockers:
        return _blocked(
            readiness,
            protocol,
            attempts,
            evaluations,
            evidence_digest=evidence_digest,
            blockers=quality_blockers,
        )
    eligible = tuple(row for row in evaluations if row.eligible)
    if not eligible:
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
    selected = min(eligible, key=_selection_key)
    experiment_ids = sorted(
        {row.profile.provenance.reader_experiment_id for candidate_rows in grouped.values() for row in candidate_rows}
    )
    stable = 0
    failed_fold = False
    selected_index = protocol.candidate_windows_h.index(selected.reduction)
    for omitted in experiment_ids:
        loo_evaluations = tuple(
            _evaluate_candidate(
                tuple(row for row in grouped[window] if row.profile.provenance.reader_experiment_id != omitted),
                window=window,
                anchors=anchor_subject_order,
                protocol=protocol,
                minimum_experiments=protocol.minimum_kinetic_experiments - 1,
            )
            for window in protocol.candidate_windows_h
        )
        loo_eligible = tuple(row for row in loo_evaluations if row.eligible)
        if not loo_eligible:
            failed_fold = True
            continue
        loo_selected = min(loo_eligible, key=_selection_key)
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


def _evaluate_candidate(
    rows: tuple[ProfileEvidence | ProfileEvidenceProjection, ...],
    *,
    window: Window,
    anchors: tuple[str, ...],
    protocol: MetastudyProtocol,
    minimum_experiments: int | None = None,
) -> CandidateEvaluation:
    by_identity: dict[tuple[str, str], ProfileEvidence | ProfileEvidenceProjection] = {}
    experiment_subjects: dict[str, set[str]] = defaultdict(set)
    blockers: list[str] = []
    limitations: list[str] = []
    growth_phase_start_by_experiment: dict[str, list[float]] = defaultdict(list)
    growth_phase_end_by_experiment: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        experiment_id = row.profile.provenance.reader_experiment_id
        identity = (experiment_id, row.profile.subject_id)
        if identity in by_identity:
            raise MetastudyContractError(f"duplicate candidate profile identity {identity!r}")
        by_identity[identity] = row
        experiment_subjects[experiment_id].add(row.profile.subject_id)
        audit = row.audit
        quality = row.audit
        if audit.within_acquisition_observation_range > audit.reference_within_acquisition_observation_range:
            limitations.append("within_acquisition_observation_range_exceeds_endpoint_reference")
        if quality.required_observation_count == 0:
            blockers.append("required_observation_count_zero")
        if quality.overflow_observation_count:
            blockers.append("observation_overflow_detected")
        if quality.clipped_observation_count:
            blockers.append("observation_clipping_detected")
        growth_phase_start_by_experiment[experiment_id].extend(
            value.normalized_start_slope for value in audit.growth_phase_strata
        )
        growth_phase_end_by_experiment[experiment_id].extend(
            value.normalized_end_slope for value in audit.growth_phase_strata
        )
    anchor_set = set(anchors)
    planned_anchor_experiments = set(protocol.planned_anchor_experiment_ids)
    eligible_experiments = tuple(sorted(experiment_subjects))
    required_experiments = minimum_experiments or protocol.minimum_kinetic_experiments
    if len(eligible_experiments) < required_experiments:
        blockers.append("minimum_7_of_8_kinetic_experiments_not_met")
    growth_phase_start_values = tuple(
        statistics.median(growth_phase_start_by_experiment[experiment_id])
        for experiment_id in sorted(growth_phase_start_by_experiment)
        if growth_phase_start_by_experiment[experiment_id]
    )
    growth_phase_end_values = tuple(
        statistics.median(growth_phase_end_by_experiment[experiment_id])
        for experiment_id in sorted(growth_phase_end_by_experiment)
        if growth_phase_end_by_experiment[experiment_id]
    )
    growth_phase_start = statistics.median(growth_phase_start_values) if growth_phase_start_values else 0.0
    growth_phase_end = statistics.median(growth_phase_end_values) if growth_phase_end_values else 0.0
    if not growth_phase_start_values or not growth_phase_end_values:
        blockers.append("growth_phase_not_estimable")
    else:
        if growth_phase_start < protocol.growth_phase_start_minimum:
            blockers.append("growth_phase_start_below_active_threshold")
        if growth_phase_end < protocol.growth_phase_end_minimum:
            blockers.append("growth_phase_end_near_plateau")
        if growth_phase_end > protocol.growth_phase_end_maximum:
            blockers.append("growth_phase_end_before_deceleration")

    separation_by_experiment: dict[str, list[float]] = defaultdict(list)
    anchor_values_by_acquisition: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        profile = row.profile
        experiment_id = profile.provenance.reader_experiment_id
        if experiment_id not in eligible_experiments:
            continue
        if isinstance(profile, ReporterMeasurementProfile) or getattr(profile, "reference_normalization", None):
            limitations.append("reference_normalization_unavailable")
            continue
        if not isinstance(profile, (ReporterResponseProfile, ProfileContentProjection)):
            raise MetastudyContractError("candidate profile variant is undeclared")
        if profile.pairing_policy is None:
            raise MetastudyContractError("normalized profile projection requires a pairing policy")
        observations = {measurement.observation_id: measurement for measurement in profile.measurements}
        response_by_observation = {response.dose_observation_id: response for response in profile.dose_responses}
        for assignment in profile.pairing_policy.assignments:
            dose = observations[assignment.dose_observation_id]
            if dose.dose_uM != protocol.primary_dose_uM:
                continue
            baseline = statistics.median(
                observations[value].rfp_over_od600 for value in assignment.baseline_observation_ids
            )
            positive = statistics.median(
                observations[value].rfp_over_od600 for value in assignment.positive_control_observation_ids
            )
            separation_by_experiment[experiment_id].append(positive - baseline)
            if experiment_id in planned_anchor_experiments and profile.subject_id in anchor_set:
                acquisition_key = (experiment_id, dose.acquisition_id)
                anchor_values_by_acquisition[acquisition_key][profile.subject_id].append(
                    response_by_observation[dose.observation_id].normalized_reporter_response
                )
    if any(not values or min(values) <= 0.0 for values in separation_by_experiment.values()):
        blockers.append("positive_control_separation_failed")
    worst_separation = min((min(values) for values in separation_by_experiment.values()), default=None)

    co_measured = 0
    ordered = 0
    plate_anchor_medians: dict[str, list[float]] = defaultdict(list)
    for subjects in anchor_values_by_acquisition.values():
        if set(subjects) != anchor_set:
            continue
        co_measured += 1
        medians = [statistics.median(subjects[subject]) for subject in anchors]
        ordered += int(all(left < right for left, right in zip(medians, medians[1:], strict=False)))
        for subject, value in zip(anchors, medians, strict=True):
            plate_anchor_medians[subject].append(value)
    if co_measured < protocol.planned_anchor_acquisitions:
        limitations.append("reference_panel_incomplete")
    if (
        co_measured < protocol.reference_panel_target_ordered_acquisitions
        or ordered < protocol.reference_panel_target_ordered_acquisitions
    ):
        limitations.append("reference_panel_support_below_target")
    drift_values = [max(values) - min(values) for values in plate_anchor_medians.values() if len(values) >= 2]
    if not drift_values:
        limitations.append("repeated_reference_drift_not_estimable")
    drift = max(drift_values, default=0.0)
    ranges = [row.audit.within_acquisition_observation_range for row in rows]
    within_acquisition_range = statistics.median(ranges) if ranges else 0.0
    unique_blockers = tuple(dict.fromkeys(blockers))
    return CandidateEvaluation(
        reduction=window,
        eligible_experiment_count=len(eligible_experiments),
        worst_experiment_control_separation=worst_separation,
        repeated_anchor_drift=drift,
        within_acquisition_observation_range=within_acquisition_range,
        growth_phase_start=growth_phase_start,
        growth_phase_end=growth_phase_end,
        anchor_ordered_acquisition_count=ordered,
        co_measured_anchor_acquisition_count=co_measured,
        loo_same_or_adjacent_fraction=0.0,
        eligible=not unique_blockers,
        blockers=unique_blockers,
        limitations=tuple(dict.fromkeys(limitations)),
    )


def _selection_key(row: CandidateEvaluation) -> tuple[float, float, float, float, float]:
    has_reference = row.worst_experiment_control_separation is not None
    return (
        0.0 if has_reference else 1.0,
        -row.worst_experiment_control_separation if has_reference else 0.0,
        (float("inf") if "repeated_reference_drift_not_estimable" in row.limitations else row.repeated_anchor_drift),
        row.within_acquisition_observation_range,
        row.reduction[1],
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
