"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/tests/reporter_response/metastudy/selection/test_selection.py

Tests primary reduction selection and descriptive sensitivity isolation.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reporter_response import (
    TimeWindowReduction,
)
from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reporter_response.metastudy import (
    GrowthPhaseStratum,
    MetastudyContractError,
    ProfileEvidence,
    decision_to_dict,
    evaluate_sensitivity,
    validate_decision_payload,
)
from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reporter_response.metastudy.audits import (
    _build_derivation_closed_profile_audit as build_profile_audit_artifact,
)

from .._builders import (
    ANCHOR_IDS,
    LOW_ANCHOR,
    _evidence,
    _profile,
    _ready,
    _selected_evaluation,
    evaluate_metastudy,
)
from ..evidence._builders import _sensitivity_evidence


def test_lexicographic_selection_uses_primary_cohort_and_is_loo_stable() -> None:
    decision = evaluate_metastudy(
        _evidence(),
        readiness=_ready(),
    )

    assert decision.status == "selected"
    assert decision.selected_reduction == (6.0, 10.0)
    assert decision.blockers == ()
    selected = _selected_evaluation(decision)
    assert selected.growth_phase_start == pytest.approx(1.0)
    assert selected.growth_phase_end == pytest.approx(0.5)
    assert selected.worst_experiment_control_separation == pytest.approx(39.0)
    assert selected.anchor_ordered_acquisition_count == 5
    assert selected.loo_same_or_adjacent_fraction == pytest.approx(1.0)


def test_selection_uses_control_separation_only_when_reference_normalization_exists() -> None:
    decision = evaluate_metastudy(
        _evidence(reference_normalized=False),
        readiness=_ready(),
    )

    assert decision.status == "selected"
    assert decision.selected_reduction == (6.0, 10.0)
    assert all(row.worst_experiment_control_separation is None for row in decision.evaluations)
    assert all("reference_normalization_unavailable" in row.limitations for row in decision.evaluations)
    assert all("positive_control_separation_failed" not in row.blockers for row in decision.evaluations)


def test_partial_normalization_coverage_disables_separation_for_the_whole_window() -> None:
    normalized = _evidence()
    raw = {
        (row.profile.provenance.reader_experiment_id, row.profile.subject_id, row.profile.reduction): row
        for row in _evidence(reference_normalized=False)
    }
    first = normalized[0]
    key = (
        first.profile.provenance.reader_experiment_id,
        first.profile.subject_id,
        first.profile.reduction,
    )
    evidence = (raw[key], *normalized[1:])

    decision = evaluate_metastudy(evidence, readiness=_ready())

    mixed = next(row for row in decision.evaluations if row.reduction == (4.0, 8.0))
    assert mixed.worst_experiment_control_separation is None
    assert "reference_normalization_unavailable" in mixed.limitations
    assert decision.selected_reduction == (6.0, 10.0)


def test_growth_phase_gate_reduces_within_acquisition_before_across_experiments() -> None:
    evidence: list[ProfileEvidence] = []
    for row in _evidence():
        reduction = row.profile.reduction
        if not isinstance(reduction, TimeWindowReduction) or (
            reduction.recorded_start_time_h,
            reduction.recorded_end_time_h,
        ) != (6.0, 10.0):
            evidence.append(row)
            continue
        experiment_id = row.profile.provenance.reader_experiment_id
        if experiment_id in ANCHOR_IDS:
            start_slope = 0.1 if row.profile.subject_id == LOW_ANCHOR else 1.0
        else:
            start_slope = 0.4
        evidence.append(
            replace(
                row,
                audit=build_profile_audit_artifact(
                    row.profile,
                    method_id=row.audit.method_id,
                    within_acquisition_observation_range=row.audit.within_acquisition_observation_range,
                    reference_within_acquisition_observation_range=(
                        row.audit.reference_within_acquisition_observation_range
                    ),
                    required_observation_count=row.audit.required_observation_count,
                    overflow_observation_count=row.audit.overflow_observation_count,
                    clipped_observation_count=row.audit.clipped_observation_count,
                    growth_phase_strata=(GrowthPhaseStratum("synthetic", start_slope, 0.5),),
                ),
            )
        )

    decision = evaluate_metastudy(tuple(evidence), readiness=_ready())
    selected = next(row for row in decision.evaluations if row.reduction == (6.0, 10.0))

    assert selected.growth_phase_start == pytest.approx(0.55)
    assert selected.eligible
    assert decision.selected_reduction == (6.0, 10.0)


def test_four_of_five_anchor_acquisitions_pass_full_and_leave_one_out_support_gates() -> None:
    missing_anchor_experiment = ANCHOR_IDS[-1]
    evidence = tuple(
        row for row in _evidence() if row.profile.provenance.reader_experiment_id != missing_anchor_experiment
    )

    decision = evaluate_metastudy(evidence, readiness=_ready())

    assert decision.status == "selected"
    selected = _selected_evaluation(decision)
    assert selected.co_measured_anchor_acquisition_count == 4
    assert selected.anchor_ordered_acquisition_count == 4
    assert selected.loo_same_or_adjacent_fraction == pytest.approx(1.0)


def test_three_of_five_anchor_acquisitions_preserve_descriptive_selection_with_limitation() -> None:
    missing_anchor_experiments = set(ANCHOR_IDS[-2:])
    evidence = tuple(
        row
        for row in _evidence()
        if not (
            row.profile.provenance.reader_experiment_id in missing_anchor_experiments
            and row.profile.subject_id == LOW_ANCHOR
        )
    )

    decision = evaluate_metastudy(evidence, readiness=_ready())

    assert decision.status == "selected"
    selected = _selected_evaluation(decision)
    assert selected.anchor_ordered_acquisition_count == 3
    assert "reference_panel_support_below_target" in selected.limitations


def test_serialized_selected_decision_accepts_exactly_four_anchor_acquisitions() -> None:
    selected = evaluate_metastudy(_evidence(), readiness=_ready())
    payload = decision_to_dict(selected)
    selected_evaluation = next(
        evaluation
        for evaluation in payload["evaluations"]
        if tuple(evaluation["reduction"]) == tuple(payload["selected_reduction"])
    )
    selected_evaluation["co_measured_anchor_acquisition_count"] = 4
    selected_evaluation["anchor_ordered_acquisition_count"] = 4

    validate_decision_payload(payload)


def test_serialized_selected_decision_accepts_three_anchor_acquisitions_as_limited() -> None:
    selected = evaluate_metastudy(_evidence(), readiness=_ready())
    payload = decision_to_dict(selected)
    selected_evaluation = next(
        evaluation
        for evaluation in payload["evaluations"]
        if tuple(evaluation["reduction"]) == tuple(payload["selected_reduction"])
    )
    selected_evaluation["co_measured_anchor_acquisition_count"] = 3
    selected_evaluation["anchor_ordered_acquisition_count"] = 3

    selected_evaluation["limitations"] = tuple(
        sorted((*selected_evaluation["limitations"], "reference_panel_support_below_target"))
    )
    validate_decision_payload(payload)


def test_optional_sensitivity_doses_do_not_change_primary_selection() -> None:
    primary_only = evaluate_metastudy(_evidence(), readiness=_ready())
    with_sensitivity = evaluate_metastudy(_evidence(doses=(5.0, 50.0, 500.0)), readiness=_ready())

    assert primary_only.selected_reduction == with_sensitivity.selected_reduction
    assert primary_only.evaluations == with_sensitivity.evaluations


def test_primary_selection_rejects_profiles_without_500_um() -> None:
    evidence = list(_evidence())
    profile = _profile(
        experiment_index=1,
        subject_id=LOW_ANCHOR,
        window=(4.0, 8.0),
        separation=40.0,
        response=0.5,
        doses=(50.0,),
    )
    prior = evidence[0].audit
    evidence[0] = ProfileEvidence(
        profile=profile,
        audit=build_profile_audit_artifact(
            profile,
            within_acquisition_observation_range=prior.within_acquisition_observation_range,
            reference_within_acquisition_observation_range=prior.reference_within_acquisition_observation_range,
            required_observation_count=prior.required_observation_count,
            overflow_observation_count=prior.overflow_observation_count,
            clipped_observation_count=prior.clipped_observation_count,
        ),
    )

    with pytest.raises(MetastudyContractError, match="must contain the 500 uM cohort"):
        evaluate_metastudy(evidence, readiness=_ready())


def test_anchor_acquisitions_are_keyed_by_experiment_and_plate() -> None:
    decision = evaluate_metastudy(_evidence(), readiness=_ready())

    assert _selected_evaluation(decision).co_measured_anchor_acquisition_count == 5


def test_one_anchor_ordering_failure_preserves_the_loo_one_missing_allowance() -> None:
    decision = evaluate_metastudy(
        _evidence(reversed_experiments=(5,)),
        readiness=_ready(),
    )

    assert decision.status == "selected"
    selected = _selected_evaluation(decision)
    assert selected.anchor_ordered_acquisition_count == 4
    assert selected.loo_same_or_adjacent_fraction == pytest.approx(1.0)


def test_missing_repeated_anchors_return_finite_sentinel_and_limitation() -> None:
    evidence = tuple(row for row in _evidence() if row.profile.subject_id == LOW_ANCHOR)

    decision = evaluate_metastudy(evidence, readiness=_ready())

    assert decision.status == "blocked"
    assert all(row.repeated_anchor_drift == 0.0 for row in decision.evaluations)
    assert all("repeated_reference_drift_not_estimable" in row.limitations for row in decision.evaluations)
    assert all("repeated_reference_drift_not_estimable" not in row.blockers for row in decision.evaluations)


def test_sensitivity_evaluation_cannot_change_primary_selection() -> None:
    primary = _evidence()
    selected = evaluate_metastudy(primary, readiness=_ready())
    before = decision_to_dict(selected)

    evaluate_sensitivity(_sensitivity_evidence())

    assert decision_to_dict(selected) == before
    assert selected.selected_reduction == (6.0, 10.0)
    assert "sensitivity_evaluations" not in before
