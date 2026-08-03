"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/reporter_response/metastudy/evaluation/candidate.py

Derived metrics for one meta-study candidate window.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import statistics
from collections import defaultdict

from ...measurement_profile import ReporterMeasurementProfile
from ...profile.normalized import ReporterResponseProfile
from ..contracts._values import MetastudyContractError
from ..contracts.candidate import CandidateEvaluation
from ..contracts.profile import ProfileEvidence
from ..contracts.protocol import MetastudyProtocol, Window
from ..evidence_projection.contracts import ProfileContentProjection, ProfileEvidenceProjection


def evaluate_candidate(
    rows: tuple[ProfileEvidence | ProfileEvidenceProjection, ...],
    *,
    window: Window,
    anchors: tuple[str, ...],
    protocol: MetastudyProtocol,
    minimum_experiments: int | None = None,
) -> CandidateEvaluation:
    """Evaluate one declared window against the protocol gates."""

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
    normalization_complete = True
    anchor_values_by_acquisition: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        profile = row.profile
        experiment_id = profile.provenance.reader_experiment_id
        if experiment_id not in eligible_experiments:
            continue
        if isinstance(profile, ReporterMeasurementProfile) or getattr(profile, "reference_normalization", None):
            normalization_complete = False
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
    worst_separation = (
        min((min(values) for values in separation_by_experiment.values()), default=None)
        if normalization_complete
        else None
    )

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


__all__ = ["evaluate_candidate"]
