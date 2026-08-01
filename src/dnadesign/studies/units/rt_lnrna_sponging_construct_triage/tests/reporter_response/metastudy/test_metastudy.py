"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/tests/reporter_response/metastudy/test_metastudy.py

Contract, selection, publication, and architecture tests for the meta-study.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import ast
import copy
import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest
import yaml

from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reader_evidence import (
    ReaderEvidenceBinding,
    ReaderEvidenceBindingSet,
)
from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reporter_response import (
    ConditionMeasurement,
    ControlAssignment,
    DoseUncertainty,
    EndpointReduction,
    NotEstimableMetricUncertainty,
    PairingPolicy,
    ReporterResponseObservationPolicy,
    ReporterResponseProfile,
    TimeWindowReduction,
    UncertaintyPolicy,
    build_reporter_response_profile,
)
from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reporter_response.metastudy import (
    ACQUISITION_PROJECTION_CONTRACT_ID,
    DEFAULT_OBJECTIVE_READINESS,
    DEFAULT_PROTOCOL,
    EvidenceReadiness,
    GrowthPhaseStratum,
    MaterializationAttemptReceipt,
    MaterializationBlocker,
    MaterializationOmission,
    MetastudyContractError,
    MetastudyDecision,
    ProfileEvidence,
    ReaderRecordIdentity,
    acquisition_projection_payload,
    build_acquisition_projection,
    decision_evidence_payload,
    decision_from_readiness,
    decision_to_dict,
    evaluate_sensitivity,
    publish_metastudy,
    readiness_from_live_bridge,
    readiness_from_receipt,
    sensitivity_coverage,
    validate_decision_payload,
    verify_publication,
)
from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reporter_response.metastudy import (
    build_profile_audit_artifact as build_synthetic_profile_audit_artifact,
)
from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reporter_response.metastudy import (
    evaluate_metastudy as evaluate_metastudy_with_attempts,
)
from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reporter_response.metastudy import (
    sensitivity_coverage as sensitivity_coverage_contracts,
)
from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reporter_response.metastudy.audits import (
    _build_derivation_closed_profile_audit as build_profile_audit_artifact,
)
from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reporter_response.metastudy.audits import (
    profile_digest,
    profile_source_identity_payload,
)
from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reporter_response.metastudy.contracts import (
    canonical_digest,
)
from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reporter_response.metastudy.evidence_projection import (
    parse_profile_evidence_projection,
    profile_source_identity_projection,
)
from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reporter_response.metastudy.sensitivity import (
    sensitivity_evaluations_to_payload,
)

SENSITIVITY_COVERAGE_CONTRACT_ID = sensitivity_coverage.SENSITIVITY_COVERAGE_CONTRACT_ID
SensitivityCoverageEntry = sensitivity_coverage.SensitivityCoverageEntry
SensitivityCoverageLedger = sensitivity_coverage.SensitivityCoverageLedger
SensitivitySubjectCoordinate = sensitivity_coverage.SensitivitySubjectCoordinate
declared_sensitivity_reduction_ids = sensitivity_coverage.declared_sensitivity_reduction_ids


def _digest(character: str) -> str:
    return "sha256:" + character * 64


LOW_ANCHOR = "rt_lnrna_pair__eco1_wt_rt__retron43_lnrna__tetO"
HIGH_ANCHOR = "rt_lnrna_pair__eco1_wt_rt__retron26_lnrna__tetO"
KINETIC_IDS = DEFAULT_PROTOCOL.planned_kinetic_experiment_ids
ANCHOR_IDS = DEFAULT_PROTOCOL.planned_anchor_experiment_ids


def _attempts(evidence: tuple[ProfileEvidence, ...]) -> tuple[MaterializationAttemptReceipt, ...]:
    grouped: dict[str, list[ProfileEvidence]] = {experiment_id: [] for experiment_id in KINETIC_IDS}
    for row in evidence:
        grouped[row.profile.provenance.reader_experiment_id].append(row)
    attempts = []
    for experiment_id, rows in grouped.items():
        provenance = rows[0].profile.provenance if rows else None
        identity = ReaderRecordIdentity(
            reader_experiment_id=experiment_id,
            reader_protocol_id=provenance.reader_protocol_id if provenance else "plate_reader/single_reporter_screen",
            reader_record_id="sample_measurements/df",
            reader_record_kind=provenance.reader_record_kind if provenance else "dataframe_artifact",
            reader_record_schema_version=6,
            reader_record_revision=provenance.reader_record_revision if provenance else 1,
            reader_record_revision_digest=provenance.reader_record_revision_digest if provenance else _digest("e"),
            reader_record_contract_id="plate_reader.annotated.v1",
            reader_record_content_digest=provenance.reader_record_content_digest if provenance else _digest("f"),
            reader_record_path=(
                provenance.reader_record_path if provenance else "artifacts/sample_measurements/df.parquet"
            ),
        )
        attempts.append(
            MaterializationAttemptReceipt(
                contract_id="rt_lnrna_reporter_response_materialization_attempt.v4",
                experiment_id=experiment_id,
                reader_record_identity=identity,
                evidence_binding_artifact_id=(provenance.evidence_binding_artifact_id if provenance else None),
                evidence_binding_artifact_digest=(provenance.evidence_binding_artifact_digest if provenance else None),
                expected_subject_ids=tuple(sorted({row.profile.subject_id for row in rows})),
                status="complete" if rows else "blocked",
                candidate_profile_count=len(rows),
                candidate_profile_digests=tuple(sorted(profile_digest(row.profile) for row in rows)),
                candidate_omissions=(),
                blockers=() if rows else (MaterializationBlocker(code="synthetic_test_evidence_missing"),),
            )
        )
    return tuple(attempts)


def evaluate_metastudy(evidence, *, readiness, protocol=DEFAULT_PROTOCOL):
    rows = tuple(evidence)
    return evaluate_metastudy_with_attempts(
        rows,
        readiness=readiness,
        attempts=_attempts(rows),
        protocol=protocol,
    )


def _selected_evaluation(decision: MetastudyDecision):
    assert decision.selected_reduction is not None
    return next(row for row in decision.evaluations if row.reduction == decision.selected_reduction)


def _profile(
    *,
    experiment_index: int,
    subject_id: str,
    window: tuple[float, float],
    separation: float,
    response: float,
    observation_policy_id: str = "rt_lnrna_reporter_response_observation_policy.v3",
    revision_digest: str = _digest("b"),
    doses: tuple[float, ...] = (500.0,),
):
    measurements: list[ConditionMeasurement] = []
    assignments: list[ControlAssignment] = []
    uncertainties: list[DoseUncertainty] = []
    for dose in doses:
        token = f"{experiment_index}-{subject_id}-{dose:g}"
        baseline_id = f"baseline-{token}"
        positive_id = f"positive-{token}"
        dose_id = f"dose-{token}"
        ratio = 100.0 + separation * (response if dose == 500.0 else response / 4.0)
        common = {
            "biological_replicate_id": None,
            "acquisition_id": KINETIC_IDS[experiment_index - 1],
            "within_acquisition_observation_count": 3,
            "within_acquisition_reduction_statistic": "median",
        }
        for observation_id, role, observed_ratio, observed_dose in (
            (baseline_id, "baseline", 100.0, None),
            (positive_id, "positive_control", 100.0 + separation, None),
            (dose_id, "dose", ratio, dose),
        ):
            measurements.append(
                ConditionMeasurement(
                    observation_id=observation_id,
                    condition_id=f"condition-{observation_id}",
                    source_condition_value=f"condition-{observation_id}",
                    role=role,
                    dose_uM=observed_dose,
                    rfp=observed_ratio,
                    od600=1.0,
                    rfp_over_od600=observed_ratio,
                    **common,
                )
            )
        assignments.append(
            ControlAssignment(
                dose_observation_id=dose_id,
                baseline_observation_ids=(baseline_id,),
                positive_control_observation_ids=(positive_id,),
            )
        )
        uncertainties.append(
            DoseUncertainty(
                dose_uM=dose,
                biological_replicate_count=0,
                normalized_reporter_response=NotEstimableMetricUncertainty(
                    estimate=(ratio - 100.0) / separation,
                    reason="biological_replicate_identity_unknown",
                ),
                relative_od=NotEstimableMetricUncertainty(
                    estimate=1.0,
                    reason="biological_replicate_identity_unknown",
                ),
            )
        )
    evidence_bindings = _binding_set(
        experiment_index=experiment_index,
        subject_id=subject_id,
        revision_digest=revision_digest,
    )
    evidence_binding = next(row for row in evidence_bindings.rows if row.subject_id == subject_id)
    return build_reporter_response_profile(
        profile_id=f"profile-{experiment_index}-{subject_id}-{window[0]:g}-{window[1]:g}",
        subject_id=subject_id,
        raw_design_id=evidence_binding.raw_design_id,
        raw_assay_subject_id=evidence_binding.raw_assay_subject_id,
        evidence_bindings=evidence_bindings,
        observation_policy=ReporterResponseObservationPolicy(
            policy_id=observation_policy_id,
            pairing_kind="pooled_controls_by_design",
            within_acquisition_reduction_statistic="median",
            biological_replicate_uncertainty_policy=UncertaintyPolicy(
                minimum_biological_replicates=2,
                biological_replicate_reduction_statistic="median",
            ),
        ),
        reduction=TimeWindowReduction(
            recorded_start_time_h=window[0],
            recorded_end_time_h=window[1],
            summary_statistic="median",
            ratio_reduction_order="ratio_then_reduce",
        ),
        dose_grid_uM=doses,
        measurements=measurements,
        pairing_policy=PairingPolicy(kind="pooled_controls_by_design", assignments=tuple(assignments)),
        dose_uncertainties=uncertainties,
        ineligibility_reasons=("preference_objective_not_defined",),
    )


def _binding_set(
    *,
    experiment_index: int,
    subject_id: str,
    revision_digest: str,
) -> ReaderEvidenceBindingSet:
    experiment_id = KINETIC_IDS[experiment_index - 1]
    subjects = (LOW_ANCHOR, HIGH_ANCHOR) if experiment_id in ANCHOR_IDS else (HIGH_ANCHOR,)
    rows = tuple(
        ReaderEvidenceBinding(
            reader_experiment_id=experiment_id,
            reader_protocol_id="plate_reader/single_reporter_screen",
            reader_replicate_kind="biological",
            reader_replicate_identity_field=None,
            reader_record_id="sample_measurements/df",
            reader_record_kind="dataframe_artifact",
            reader_record_schema_version=6,
            reader_record_revision=4,
            reader_record_revision_digest=revision_digest,
            reader_record_contract_id="plate_reader.annotated.v1",
            reader_record_content_digest=_digest("c"),
            reader_record_path="outputs/records/sample_measurements/df__r4.parquet",
            raw_design_id=f"design-{row_subject_id}",
            raw_assay_subject_id=f"assay-{row_subject_id}",
            subject_id=row_subject_id,
            observation_identity_field="position",
            observation_identity_values=(f"position-{experiment_index}-{row_subject_id}",),
            binding_state="bound",
            binding_reason="exact_subject_alias_match",
        )
        for row_subject_id in subjects
    )
    return ReaderEvidenceBindingSet._from_source_closed_record(
        schema_id="rt_lnrna_reader_evidence_bindings_v4",
        subject_binding_set_id="subject-bindings-v1",
        rows=rows,
    )


def _evidence(
    *,
    doses: tuple[float, ...] = (500.0,),
    reversed_experiments: tuple[int, ...] = (),
) -> tuple[ProfileEvidence, ...]:
    rows: list[ProfileEvidence] = []
    phase_by_window = {
        (4.0, 8.0): (0.65, 1.0),
        (6.0, 10.0): (1.0, 0.5),
        (8.0, 12.0): (0.9, 0.05),
        (10.0, 14.0): (0.2, 0.03),
        (12.0, 16.0): (0.03, 0.02),
    }
    for candidate_index, window in enumerate(DEFAULT_PROTOCOL.candidate_windows_h):
        for experiment_index in range(1, 9):
            experiment_id = KINETIC_IDS[experiment_index - 1]
            subject_responses = [(HIGH_ANCHOR, 1.0)]
            if experiment_id in ANCHOR_IDS:
                subject_responses.insert(0, (LOW_ANCHOR, 0.5))
            for subject_id, default_response in subject_responses:
                response = (
                    0.25 if subject_id == HIGH_ANCHOR and experiment_index in reversed_experiments else default_response
                )
                profile = _profile(
                    experiment_index=experiment_index,
                    subject_id=subject_id,
                    window=window,
                    separation=40.0 - candidate_index,
                    response=response + experiment_index / 1000.0,
                    doses=doses,
                )
                rows.append(
                    ProfileEvidence(
                        profile=profile,
                        audit=build_profile_audit_artifact(
                            profile,
                            method_id="synthetic_profile_audit_v1",
                            within_acquisition_observation_range=0.10 + candidate_index / 100.0,
                            reference_within_acquisition_observation_range=0.20,
                            required_observation_count=1,
                            overflow_observation_count=0,
                            clipped_observation_count=0,
                            growth_phase_strata=(
                                GrowthPhaseStratum(
                                    "synthetic",
                                    phase_by_window[window][0],
                                    phase_by_window[window][1],
                                ),
                            ),
                        ),
                    )
                )
    return tuple(rows)


def test_acquisition_projection_is_descriptive_and_exposes_leave_one_acquisition_out() -> None:
    from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reporter_response.metastudy import (
        build_acquisition_projection,
    )

    projection = build_acquisition_projection(_evidence(), selected_reduction=(6.0, 10.0))

    assert projection.contract_id == ACQUISITION_PROJECTION_CONTRACT_ID
    assert projection.selected_reduction == (6.0, 10.0)
    assert {row.reduction_id for row in projection.coordinates} == {"window-6-10h"}
    high = next(
        row
        for row in projection.coordinates
        if row.subject_id == HIGH_ANCHOR and row.reduction_id == "window-6-10h" and row.dose_uM == 500.0
    )
    assert high.acquisition_ids == KINETIC_IDS
    assert all(row.declared_biological_replicate_ids == () for row in high.contributions)
    assert high.normalized_reporter_response.method == "median_across_acquisitions"
    assert high.normalized_reporter_response.acquisition_count == 8
    assert len(high.normalized_reporter_response.leave_one_acquisition_out_estimates) == 8
    assert not hasattr(high.normalized_reporter_response, "interval_lower")


def test_acquisition_projection_keeps_single_acquisition_descriptive() -> None:
    from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reporter_response.metastudy import (
        build_acquisition_projection,
    )

    source = next(
        row
        for row in _evidence()
        if row.profile.subject_id == HIGH_ANCHOR
        and row.profile.provenance.reader_experiment_id == KINETIC_IDS[0]
        and isinstance(row.profile.reduction, TimeWindowReduction)
        and row.profile.reduction.recorded_start_time_h == 6.0
    )
    projection = build_acquisition_projection((source,), selected_reduction=(6.0, 10.0))

    assert len(projection.coordinates) == 1
    coordinate = projection.coordinates[0]
    assert coordinate.acquisition_ids == (KINETIC_IDS[0],)
    assert coordinate.normalized_reporter_response.acquisition_count == 1
    assert coordinate.normalized_reporter_response.leave_one_acquisition_out_estimates == ()


def test_acquisition_projection_rejects_duplicate_acquisition_coordinate() -> None:
    from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reporter_response.metastudy import (
        build_acquisition_projection,
    )

    source = _evidence()[0]
    with pytest.raises(MetastudyContractError, match="duplicate acquisition"):
        build_acquisition_projection((source, source), selected_reduction=(4.0, 8.0))


def _sensitivity_evidence() -> tuple[ProfileEvidence, ...]:
    rows: list[ProfileEvidence] = []
    for source in _evidence(doses=(5.0, 50.0, 500.0)):
        reduction = source.profile.reduction
        if not isinstance(reduction, TimeWindowReduction) or reduction.recorded_start_time_h != 4.0:
            continue
        profile = replace(
            source.profile,
            profile_id=f"{source.profile.profile_id}-endpoint-8",
            reduction=EndpointReduction(recorded_time_h=8.0),
        )
        rows.append(
            ProfileEvidence(
                profile=profile,
                audit=build_profile_audit_artifact(
                    profile,
                    method_id=source.audit.method_id,
                    within_acquisition_observation_range=source.audit.within_acquisition_observation_range,
                    reference_within_acquisition_observation_range=(
                        source.audit.reference_within_acquisition_observation_range
                    ),
                    required_observation_count=source.audit.required_observation_count,
                    overflow_observation_count=source.audit.overflow_observation_count,
                    clipped_observation_count=source.audit.clipped_observation_count,
                ),
            )
        )
    return tuple(rows)


def _complete_sensitivity_evidence(
    primary: tuple[ProfileEvidence, ...],
) -> tuple[ProfileEvidence, ...]:
    sources: dict[tuple[str, str], ProfileEvidence] = {}
    for row in primary:
        key = (row.profile.provenance.reader_experiment_id, row.profile.subject_id)
        sources.setdefault(key, row)
    reductions = (
        *(EndpointReduction(recorded_time_h=value) for value in DEFAULT_PROTOCOL.endpoint_sensitivity_h),
        *(
            TimeWindowReduction(
                recorded_start_time_h=(start + end) / 2.0 - width / 2.0,
                recorded_end_time_h=(start + end) / 2.0 + width / 2.0,
                summary_statistic="median",
                ratio_reduction_order="ratio_then_reduce",
            )
            for start, end in DEFAULT_PROTOCOL.candidate_windows_h
            for width in DEFAULT_PROTOCOL.centered_window_sensitivity_widths_h
        ),
    )
    rows: list[ProfileEvidence] = []
    for source in sources.values():
        for reduction in reductions:
            reduction_id = (
                f"endpoint-{reduction.recorded_time_h:g}h"
                if isinstance(reduction, EndpointReduction)
                else f"window-{reduction.recorded_start_time_h:g}-{reduction.recorded_end_time_h:g}h"
            )
            profile = replace(
                source.profile,
                profile_id=(
                    f"{source.profile.provenance.reader_experiment_id}:{source.profile.subject_id}:{reduction_id}"
                ),
                reduction=reduction,
            )
            rows.append(
                ProfileEvidence(
                    profile=profile,
                    audit=build_profile_audit_artifact(
                        profile,
                        method_id=source.audit.method_id,
                        within_acquisition_observation_range=source.audit.within_acquisition_observation_range,
                        reference_within_acquisition_observation_range=(
                            source.audit.reference_within_acquisition_observation_range
                        ),
                        required_observation_count=source.audit.required_observation_count,
                        overflow_observation_count=source.audit.overflow_observation_count,
                        clipped_observation_count=source.audit.clipped_observation_count,
                    ),
                )
            )
    return tuple(rows)


def _sensitivity_coverages(
    evidence: tuple[ProfileEvidence, ...],
    attempts: tuple[MaterializationAttemptReceipt, ...],
) -> tuple[SensitivityCoverageLedger, ...]:
    attempt_by_experiment = {row.experiment_id: row for row in attempts if row.status in {"complete", "partial"}}
    by_experiment: dict[str, list[ProfileEvidence]] = {}
    for row in evidence:
        by_experiment.setdefault(row.profile.provenance.reader_experiment_id, []).append(row)
    coverages = []
    for experiment_id, rows in sorted(by_experiment.items()):
        provenance = rows[0].profile.provenance
        binding_ids = {
            (
                row.profile.provenance.evidence_binding_artifact_id,
                row.profile.provenance.evidence_binding_artifact_digest,
            )
            for row in rows
        }
        assert len(binding_ids) == 1
        binding_id, binding_digest = binding_ids.pop()
        subjects = tuple(
            sorted(
                {
                    SensitivitySubjectCoordinate(
                        row.profile.provenance.raw_design_id,
                        row.profile.provenance.raw_assay_subject_id,
                        row.profile.subject_id,
                    )
                    for row in rows
                },
                key=lambda row: (row.subject_id, row.raw_design_id or "", row.raw_assay_subject_id or ""),
            )
        )
        by_coordinate = {
            (row.profile.subject_id, row.profile.reduction.kind, row.profile.profile_id.rsplit(":", 1)[-1]): row
            for row in rows
        }
        entries = []
        for subject in subjects:
            for reduction_id in declared_sensitivity_reduction_ids():
                kind = "endpoint" if reduction_id.startswith("endpoint-") else "time_window"
                profile = by_coordinate[(subject.subject_id, kind, reduction_id)]
                entries.append(
                    SensitivityCoverageEntry(
                        subject=subject,
                        reduction_id=reduction_id,
                        outcome="profile",
                        profile_digest=profile.audit.profile_digest,
                        omission=None,
                    )
                )
        coverages.append(
            SensitivityCoverageLedger(
                contract_id=SENSITIVITY_COVERAGE_CONTRACT_ID,
                experiment_id=experiment_id,
                materialization_attempt_digest=attempt_by_experiment[experiment_id].attempt_digest,
                reader_record_identity=ReaderRecordIdentity(
                    reader_experiment_id=experiment_id,
                    reader_protocol_id=provenance.reader_protocol_id,
                    reader_record_id=provenance.reader_record_id,
                    reader_record_kind=provenance.reader_record_kind,
                    reader_record_schema_version=provenance.reader_record_schema_version,
                    reader_record_revision=provenance.reader_record_revision,
                    reader_record_revision_digest=provenance.reader_record_revision_digest,
                    reader_record_contract_id=provenance.reader_record_contract_id,
                    reader_record_content_digest=provenance.reader_record_content_digest,
                    reader_record_path=provenance.reader_record_path,
                ),
                evidence_binding_artifact_id=binding_id,
                evidence_binding_artifact_digest=binding_digest,
                expected_subjects=subjects,
                expected_reduction_ids=declared_sensitivity_reduction_ids(),
                entries=tuple(entries),
            )
        )
    return tuple(coverages)


def _publish_selected(decision: MetastudyDecision, destination: Path, *, evidence=None) -> Path:
    primary = _evidence() if evidence is None else evidence
    return _publish_evaluated(decision, destination, evidence=primary)


def _publish_evaluated(
    decision: MetastudyDecision,
    destination: Path,
    *,
    evidence: tuple[ProfileEvidence, ...],
) -> Path:
    sensitivity = _complete_sensitivity_evidence(_evidence())
    return publish_metastudy(
        decision,
        destination,
        primary_evidence=evidence,
        sensitivity_evidence=sensitivity,
        sensitivity_evaluations=evaluate_sensitivity(sensitivity),
        sensitivity_coverages=_sensitivity_coverages(sensitivity, decision.materialization_attempts),
    )


def _quality_blocked_evidence() -> tuple[ProfileEvidence, ...]:
    evidence = list(_evidence())
    audit = evidence[0].audit
    evidence[0] = replace(
        evidence[0],
        audit=build_profile_audit_artifact(
            evidence[0].profile,
            method_id=audit.method_id,
            within_acquisition_observation_range=audit.within_acquisition_observation_range,
            reference_within_acquisition_observation_range=audit.reference_within_acquisition_observation_range,
            required_observation_count=0,
            overflow_observation_count=audit.overflow_observation_count,
            clipped_observation_count=audit.clipped_observation_count,
        ),
    )
    return tuple(evidence)


def _ready() -> EvidenceReadiness:
    return EvidenceReadiness._from_owner_bridge_receipt(
        selected_experiment_count=8,
        ready_experiment_count=8,
        ready_experiment_ids=KINETIC_IDS,
        blocked_experiment_ids=(),
        receipt_digest=_digest("e"),
    )


def test_predeclared_protocol_is_exact_and_has_no_weighted_score() -> None:
    assert DEFAULT_PROTOCOL.protocol_id == "rt_lnrna_reporter_response_metastudy.v3"
    assert DEFAULT_PROTOCOL.primary_dose_uM == 500.0
    assert DEFAULT_PROTOCOL.sensitivity_doses_uM == (5.0, 50.0)
    assert DEFAULT_PROTOCOL.candidate_windows_h == (
        (4.0, 8.0),
        (6.0, 10.0),
        (8.0, 12.0),
        (10.0, 14.0),
        (12.0, 16.0),
    )
    assert DEFAULT_PROTOCOL.endpoint_sensitivity_h == (8.0, 10.0, 12.0, 14.0, 16.0)
    assert DEFAULT_PROTOCOL.centered_window_sensitivity_widths_h == (2.0, 6.0)
    assert DEFAULT_PROTOCOL.growth_phase_slope_window_h == 1.0
    assert DEFAULT_PROTOCOL.growth_phase_scale_quantile == 0.9
    assert DEFAULT_PROTOCOL.growth_phase_minimum_slope_points == 4
    assert DEFAULT_PROTOCOL.growth_phase_start_minimum == 0.5
    assert DEFAULT_PROTOCOL.growth_phase_end_minimum == 0.1
    assert DEFAULT_PROTOCOL.growth_phase_end_maximum == 0.6
    assert DEFAULT_PROTOCOL.minimum_kinetic_experiments == 7
    assert DEFAULT_PROTOCOL.planned_kinetic_experiments == 8
    assert DEFAULT_PROTOCOL.anchor_subject_order == (LOW_ANCHOR, HIGH_ANCHOR)
    assert DEFAULT_PROTOCOL.planned_anchor_experiment_ids == (
        KINETIC_IDS[0],
        KINETIC_IDS[1],
        KINETIC_IDS[3],
        KINETIC_IDS[4],
        KINETIC_IDS[5],
    )
    assert DEFAULT_PROTOCOL.reference_panel_target_ordered_acquisitions == 4
    assert DEFAULT_PROTOCOL.planned_anchor_acquisitions == 5
    assert DEFAULT_PROTOCOL.loo_same_or_adjacent_target_fraction == 0.75
    assert DEFAULT_PROTOCOL.selection_order[0] == "require_active_to_decelerating_growth_phase"
    assert "weight" not in repr(DEFAULT_PROTOCOL).lower()


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


@pytest.mark.parametrize(
    ("audit_changes", "blocker"),
    [
        ({"required_observation_count": 0}, "required_observation_count_zero"),
        ({"overflow_observation_count": 1}, "observation_overflow_detected"),
        ({"clipped_observation_count": 1}, "observation_clipping_detected"),
    ],
)
def test_observation_quality_audit_blocks_zero_overflow_and_clipping(audit_changes, blocker: str) -> None:
    evidence = list(_evidence())
    audit = evidence[0].audit
    evidence[0] = replace(
        evidence[0],
        audit=build_profile_audit_artifact(
            evidence[0].profile,
            method_id=audit.method_id,
            within_acquisition_observation_range=audit.within_acquisition_observation_range,
            reference_within_acquisition_observation_range=audit.reference_within_acquisition_observation_range,
            required_observation_count=audit_changes.get(
                "required_observation_count", audit.required_observation_count
            ),
            overflow_observation_count=audit_changes.get(
                "overflow_observation_count", audit.overflow_observation_count
            ),
            clipped_observation_count=audit_changes.get("clipped_observation_count", audit.clipped_observation_count),
        ),
    )

    decision = evaluate_metastudy(evidence, readiness=_ready())

    assert decision.status == "blocked"
    assert blocker in decision.evaluations[0].blockers


def test_profile_audit_rejects_mutation_and_cross_profile_rebinding() -> None:
    evidence = _evidence()
    with pytest.raises(MetastudyContractError, match="artifact digest mismatch"):
        replace(
            evidence[0],
            audit=replace(evidence[0].audit, within_acquisition_observation_range=9.0),
        )
    with pytest.raises(MetastudyContractError, match="source identity digest mismatch"):
        replace(evidence[1], audit=evidence[0].audit)


def test_public_audit_builder_cannot_claim_canonical_derivation() -> None:
    profile = _evidence()[0].profile

    with pytest.raises(ValueError, match="canonical audits are derived only"):
        build_synthetic_profile_audit_artifact(
            profile,
            method_id="canonical_profile_observation_audit_v1",
            within_acquisition_observation_range=0.1,
            reference_within_acquisition_observation_range=0.2,
            required_observation_count=1,
            overflow_observation_count=0,
            clipped_observation_count=0,
        )

    synthetic = build_synthetic_profile_audit_artifact(
        profile,
        method_id="synthetic_profile_audit_v1",
        within_acquisition_observation_range=0.1,
        reference_within_acquisition_observation_range=0.2,
        required_observation_count=1,
        overflow_observation_count=0,
        clipped_observation_count=0,
    )
    evidence = list(_evidence())
    evidence[0] = ProfileEvidence(profile=profile, audit=synthetic)
    with pytest.raises(MetastudyContractError, match="derivation-closed canonical"):
        evaluate_metastudy(evidence, readiness=_ready())


def test_full_profile_digest_prevents_rebinding_after_profile_mutation() -> None:
    evidence = _evidence()[0]
    changed_profile = replace(evidence.profile, profile_id="forged-profile-id")

    with pytest.raises(MetastudyContractError, match="full profile digest mismatch"):
        ProfileEvidence(profile=changed_profile, audit=evidence.audit)


def test_synthetic_readiness_cannot_enter_evaluation() -> None:
    readiness = EvidenceReadiness(
        selected_experiment_count=8,
        ready_experiment_count=8,
        ready_experiment_ids=KINETIC_IDS,
        blocked_experiment_ids=(),
        receipt_digest=_digest("f"),
    )

    with pytest.raises(MetastudyContractError, match="readiness_from_receipt"):
        evaluate_metastudy(_evidence(), readiness=readiness)


def test_cross_window_roster_drift_fails_closed() -> None:
    evidence = list(_evidence())
    evidence.pop()

    with pytest.raises(MetastudyContractError, match="candidate coordinate closure differs"):
        evaluate_metastudy(evidence, readiness=_ready())


def test_cross_window_reader_provenance_drift_fails_closed() -> None:
    evidence = list(_evidence())
    changed_profile = _profile(
        experiment_index=8,
        subject_id=HIGH_ANCHOR,
        window=DEFAULT_PROTOCOL.candidate_windows_h[-1],
        separation=36.0,
        response=1.008,
        revision_digest=_digest("f"),
    )
    prior_audit = evidence[-1].audit
    evidence[-1] = ProfileEvidence(
        profile=changed_profile,
        audit=build_profile_audit_artifact(
            changed_profile,
            method_id=prior_audit.method_id,
            within_acquisition_observation_range=prior_audit.within_acquisition_observation_range,
            reference_within_acquisition_observation_range=(prior_audit.reference_within_acquisition_observation_range),
            required_observation_count=prior_audit.required_observation_count,
            overflow_observation_count=prior_audit.overflow_observation_count,
            clipped_observation_count=prior_audit.clipped_observation_count,
        ),
    )

    with pytest.raises(MetastudyContractError, match="Reader identity differs from profile provenance"):
        evaluate_metastudy(evidence, readiness=_ready())


@pytest.mark.parametrize(
    ("field_name", "changed_value"),
    [
        ("reader_protocol_id", "plate_reader/another_protocol"),
        ("reader_record_kind", "another_kind"),
        ("reader_record_path", "artifacts/another.parquet"),
    ],
)
def test_primary_profile_requires_complete_attempt_reader_identity(
    field_name: str,
    changed_value: str,
) -> None:
    evidence = _evidence()
    attempts = list(_attempts(evidence))
    identity = attempts[0].reader_record_identity
    assert identity is not None
    attempts[0] = replace(
        attempts[0],
        reader_record_identity=replace(identity, **{field_name: changed_value}),
    )

    with pytest.raises(MetastudyContractError, match="Reader identity differs from profile provenance"):
        evaluate_metastudy_with_attempts(evidence, readiness=_ready(), attempts=attempts)


@pytest.mark.parametrize(
    ("field_name", "changed_value"),
    [
        ("reader_protocol_id", "plate_reader/another_protocol"),
        ("reader_record_kind", "another_kind"),
        ("reader_record_path", "artifacts/another.parquet"),
    ],
)
def test_sensitivity_profile_requires_complete_coverage_reader_identity(
    field_name: str,
    changed_value: str,
) -> None:
    primary = _evidence()
    attempts = _attempts(primary)
    sensitivity = _complete_sensitivity_evidence(primary)
    coverage = _sensitivity_coverages(sensitivity, attempts)[0]
    changed_coverage = replace(
        coverage,
        reader_record_identity=replace(
            coverage.reader_record_identity,
            **{field_name: changed_value},
        ),
    )
    experiment_evidence = tuple(
        row for row in sensitivity if row.profile.provenance.reader_experiment_id == coverage.experiment_id
    )

    with pytest.raises(MetastudyContractError, match="sensitivity profile provenance differs"):
        sensitivity_coverage_contracts.validate_sensitivity_coverage(
            changed_coverage,
            evidence=experiment_evidence,
        )


def test_sensitivity_evidence_is_typed_and_never_selectable() -> None:
    primary = _profile(
        experiment_index=1,
        subject_id=LOW_ANCHOR,
        window=(4.0, 8.0),
        separation=40.0,
        response=0.5,
        doses=(5.0, 50.0, 500.0),
    )
    endpoint = replace(primary, reduction=EndpointReduction(recorded_time_h=8.0))
    centered = replace(
        primary,
        profile_id="centered-sensitivity-profile",
        reduction=TimeWindowReduction(
            recorded_start_time_h=8.0,
            recorded_end_time_h=10.0,
            summary_statistic="median",
            ratio_reduction_order="ratio_then_reduce",
        ),
    )
    template = _evidence(doses=(5.0, 50.0, 500.0))[0]
    audit = template.audit

    def bind(profile):
        return ProfileEvidence(
            profile=profile,
            audit=build_profile_audit_artifact(
                profile,
                method_id=audit.method_id,
                within_acquisition_observation_range=audit.within_acquisition_observation_range,
                reference_within_acquisition_observation_range=audit.reference_within_acquisition_observation_range,
                required_observation_count=audit.required_observation_count,
                overflow_observation_count=audit.overflow_observation_count,
                clipped_observation_count=audit.clipped_observation_count,
            ),
        )

    results = evaluate_sensitivity(
        (
            bind(endpoint),
            bind(centered),
        )
    )

    assert {row.kind for row in results} == {"dose", "endpoint", "centered_window"}
    assert all(row.selectable is False for row in results)


def test_mismatched_profile_comparability_fails_closed() -> None:
    evidence = list(_evidence())
    changed_profile = _profile(
        experiment_index=1,
        subject_id=LOW_ANCHOR,
        window=(4.0, 8.0),
        separation=40.0,
        response=0.5,
        observation_policy_id="rt_lnrna_observation_policy_v2",
        doses=(500.0,),
    )
    prior_audit = evidence[0].audit
    evidence[0] = ProfileEvidence(
        profile=changed_profile,
        audit=build_profile_audit_artifact(
            changed_profile,
            method_id=prior_audit.method_id,
            within_acquisition_observation_range=prior_audit.within_acquisition_observation_range,
            reference_within_acquisition_observation_range=(prior_audit.reference_within_acquisition_observation_range),
            required_observation_count=prior_audit.required_observation_count,
            overflow_observation_count=prior_audit.overflow_observation_count,
            clipped_observation_count=prior_audit.clipped_observation_count,
        ),
    )

    with pytest.raises(MetastudyContractError, match="observation policy|comparability"):
        evaluate_metastudy(
            evidence,
            readiness=_ready(),
        )


def test_endpoint_profiles_cannot_enter_primary_selection() -> None:
    evidence = list(_evidence())
    endpoint = replace(evidence[0].profile, reduction=replace(evidence[0].profile.reduction))
    object.__setattr__(endpoint, "reduction", EndpointReduction(recorded_time_h=8.0))
    prior = evidence[0].audit
    evidence[0] = ProfileEvidence(
        profile=endpoint,
        audit=build_profile_audit_artifact(
            endpoint,
            within_acquisition_observation_range=prior.within_acquisition_observation_range,
            reference_within_acquisition_observation_range=prior.reference_within_acquisition_observation_range,
            required_observation_count=prior.required_observation_count,
            overflow_observation_count=prior.overflow_observation_count,
            clipped_observation_count=prior.clipped_observation_count,
        ),
    )

    with pytest.raises(MetastudyContractError, match="only time-window profiles"):
        evaluate_metastudy(
            evidence,
            readiness=_ready(),
        )


def test_zero_of_eight_reader_readiness_produces_typed_blocked_decision() -> None:
    readiness = EvidenceReadiness._from_validated_receipt(
        selected_experiment_count=8,
        ready_experiment_count=0,
        ready_experiment_ids=(),
        blocked_experiment_ids=KINETIC_IDS,
        receipt_digest=_digest("9"),
    )
    decision = decision_from_readiness(readiness)

    assert decision.status == "blocked"
    assert decision.selected_reduction is None
    assert "reader_evidence_ready_0_of_8" in decision.blockers
    assert decision.policy_digest.startswith("sha256:")
    assert decision.evidence_digest.startswith("sha256:")


def test_arbitrary_ready_experiments_cannot_clear_the_7_of_8_kinetic_gate() -> None:
    arbitrary = tuple(f"arbitrary-experiment-{index}" for index in range(1, 8))
    readiness = EvidenceReadiness._from_validated_receipt(
        selected_experiment_count=7,
        ready_experiment_count=7,
        ready_experiment_ids=arbitrary,
        blocked_experiment_ids=(),
        receipt_digest=_digest("7"),
    )

    decision = evaluate_metastudy((), readiness=readiness)

    assert decision.status == "blocked"
    assert "minimum_7_of_8_kinetic_experiments_not_met" in decision.blockers


def test_read_only_readiness_receipt_adapter_preserves_zero_of_eight() -> None:
    receipt = _readiness_receipt()

    readiness = readiness_from_receipt(receipt)

    assert readiness.selected_experiment_count == 8
    assert readiness.ready_experiment_count == 0
    assert readiness.ready_experiment_ids == ()
    assert len(readiness.blocked_experiment_ids) == 8


def test_readiness_receipt_digest_omits_only_environment_specific_reader_command() -> None:
    first = _readiness_receipt()
    second = json.loads(json.dumps(first))
    second["reader_command"] = ["/different/workstation/.venv/bin/reader"]

    assert readiness_from_receipt(first).receipt_digest == readiness_from_receipt(second).receipt_digest


def test_structurally_valid_synthetic_receipt_cannot_authorize_selection() -> None:
    readiness = readiness_from_receipt(_readiness_receipt(ready_ids=KINETIC_IDS))

    with pytest.raises(MetastudyContractError, match="owner-bound live bridge runner"):
        evaluate_metastudy(_evidence(), readiness=readiness)


def test_live_bridge_runner_is_the_selection_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    skill = tmp_path / ".agents/skills/retron-assay-study-bridge"
    registry = skill / "references/reader-experiment-routes.json"
    checker = skill / "scripts/check_reader_experiment_readiness.py"
    registry.parent.mkdir(parents=True)
    checker.parent.mkdir(parents=True)
    registry.write_text("{}\n", encoding="utf-8")
    checker.write_text("# fixture\n", encoding="utf-8")
    receipt = _readiness_receipt(ready_ids=KINETIC_IDS)

    monkeypatch.setattr(
        "subprocess.run",
        lambda *_args, **_kwargs: type(
            "Completed",
            (),
            {"stdout": json.dumps(receipt), "stderr": "", "returncode": 0},
        )(),
    )

    readiness = readiness_from_live_bridge(phd_root=tmp_path)

    assert readiness.is_selection_authorized
    assert evaluate_metastudy(_evidence(), readiness=readiness).status == "selected"


def _readiness_receipt(*, ready_ids: tuple[str, ...] = ()) -> dict[str, object]:
    selected_ids = KINETIC_IDS
    related_ids = DEFAULT_PROTOCOL.excluded_snapshot_experiment_ids
    blocked_ids = tuple(value for value in selected_ids if value not in ready_ids)
    return {
        "available_protocols": ["plate_reader/single_reporter_screen"],
        "contract_errors": [],
        "experiments": [
            {
                "experiment_id": experiment_id,
                "memberships": [
                    {
                        "membership": "selected" if experiment_id in selected_ids else "related",
                        "ready": experiment_id in ready_ids,
                        "required_reader_state": "records_ready",
                        "route_id": "rt_lnrna_reporter_response_metastudy",
                    }
                ],
            }
            for experiment_id in (*selected_ids, *related_ids)
        ],
        "ok": len(ready_ids) == len(selected_ids),
        "reader_command": ["reader"],
        "route_id": "rt_lnrna_reporter_response_metastudy",
        "summary": {
            "contract_error_count": 0,
            "experiment_count": 9,
            "membership_count": 9,
            "related_membership_count": 1,
            "selected_membership_count": 8,
            "selected_ready_count": len(ready_ids),
            "selected_blocker_count": len(blocked_ids),
        },
        "selected_blockers": [
            {"experiment_id": experiment_id, "route_id": "rt_lnrna_reporter_response_metastudy"}
            for experiment_id in blocked_ids
        ],
    }


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda payload: payload.update({"unexpected": True}), "top-level fields"),
        (lambda payload: payload["summary"].update({"unexpected": 1}), "summary fields"),
        (lambda payload: payload.update({"route_id": "wrong-route"}), "route_id"),
    ],
)
def test_readiness_receipt_rejects_contract_drift(mutation, match: str) -> None:
    receipt = _readiness_receipt()
    mutation(receipt)

    with pytest.raises(MetastudyContractError, match=match):
        readiness_from_receipt(receipt)


def test_readiness_receipt_rejects_errors_ok_drift_and_wrong_selected_identity() -> None:
    receipt = _readiness_receipt()
    receipt["contract_errors"] = [{"code": "invalid"}]
    receipt["summary"]["contract_error_count"] = 1
    with pytest.raises(MetastudyContractError, match="contains contract_errors"):
        readiness_from_receipt(receipt)

    receipt = _readiness_receipt()
    receipt["ok"] = True
    with pytest.raises(MetastudyContractError, match="ok does not match"):
        readiness_from_receipt(receipt)

    receipt = _readiness_receipt()
    receipt["experiments"][0]["experiment_id"] = "arbitrary-substitute"
    receipt["selected_blockers"][0]["experiment_id"] = "arbitrary-substitute"
    with pytest.raises(MetastudyContractError, match="predeclared route cohort"):
        readiness_from_receipt(receipt)


def test_publication_is_create_only_deterministic_and_verified(tmp_path: Path) -> None:
    decision = decision_from_readiness(
        EvidenceReadiness._from_validated_receipt(
            selected_experiment_count=8,
            ready_experiment_count=0,
            ready_experiment_ids=(),
            blocked_experiment_ids=KINETIC_IDS,
            receipt_digest=_digest("9"),
        )
    )
    destination = tmp_path / "decision-v1"

    publish_metastudy(decision, destination)
    first = {path.name: path.read_bytes() for path in destination.iterdir()}
    verify_publication(destination)

    with pytest.raises(FileExistsError):
        publish_metastudy(decision, destination)
    assert {path.name: path.read_bytes() for path in destination.iterdir()} == first


def test_readiness_only_publication_rejects_primary_evidence(tmp_path: Path) -> None:
    decision = decision_from_readiness(
        EvidenceReadiness._from_validated_receipt(
            selected_experiment_count=8,
            ready_experiment_count=0,
            ready_experiment_ids=(),
            blocked_experiment_ids=KINETIC_IDS,
            receipt_digest=_digest("9"),
        )
    )

    with pytest.raises(MetastudyContractError, match="readiness-only publication"):
        publish_metastudy(decision, tmp_path / "invalid", primary_evidence=_evidence())


def test_evidence_bearing_blocked_publication_round_trips_offline(tmp_path: Path) -> None:
    primary = _quality_blocked_evidence()
    decision = evaluate_metastudy(primary, readiness=_ready())
    destination = _publish_evaluated(decision, tmp_path / "evaluated-blocked", evidence=primary)

    assert decision.status == "blocked"
    assert decision.evaluations
    assert {path.name for path in destination.iterdir()} == {
        "manifest.json",
        "report.md",
        "evidence.json",
        "sensitivity.json",
    }
    verify_publication(destination)


def test_evidence_bearing_blocked_publication_requires_primary_evidence(tmp_path: Path) -> None:
    primary = _quality_blocked_evidence()
    decision = evaluate_metastudy(primary, readiness=_ready())

    with pytest.raises(MetastudyContractError, match="evidence-bearing publication"):
        publish_metastudy(decision, tmp_path / "missing-evidence")


def test_publication_installs_one_complete_staged_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reporter_response.metastudy import publication

    decision = decision_from_readiness(
        EvidenceReadiness._from_validated_receipt(
            selected_experiment_count=8,
            ready_experiment_count=0,
            ready_experiment_ids=(),
            blocked_experiment_ids=KINETIC_IDS,
            receipt_digest=_digest("9"),
        )
    )
    destination = tmp_path / "atomic-publication"
    original_install = publication._rename_directory_create_only
    observed_install: list[tuple[set[str], bool]] = []

    def inspect_install(stage: Path, target: Path) -> None:
        verify_publication(stage)
        observed_install.append(({entry.name for entry in stage.iterdir()}, target.exists()))
        original_install(stage, target)

    monkeypatch.setattr(publication, "_rename_directory_create_only", inspect_install)

    publish_metastudy(decision, destination)

    assert observed_install == [({"manifest.json", "report.md", "sensitivity.json"}, False)]
    verify_publication(destination)


def test_publication_target_race_preserves_competitor_and_cleans_staging(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reporter_response.metastudy import publication

    decision = decision_from_readiness(
        EvidenceReadiness._from_validated_receipt(
            selected_experiment_count=8,
            ready_experiment_count=0,
            ready_experiment_ids=(),
            blocked_experiment_ids=KINETIC_IDS,
            receipt_digest=_digest("9"),
        )
    )
    destination = tmp_path / "raced-publication"
    original_install = publication._rename_directory_create_only
    competitor_inode: list[int] = []

    def race_install(stage: Path, target: Path) -> None:
        target.mkdir()
        competitor_inode.append(target.stat().st_ino)
        original_install(stage, target)

    monkeypatch.setattr(publication, "_rename_directory_create_only", race_install)

    with pytest.raises(FileExistsError, match="create-only"):
        publish_metastudy(decision, destination)

    assert destination.is_dir()
    assert destination.stat().st_ino == competitor_inode[0]
    assert list(destination.iterdir()) == []
    assert list(tmp_path.glob(".raced-publication.*")) == []


def test_publication_rejects_broken_destination_symlink_without_following_it(tmp_path: Path) -> None:
    decision = decision_from_readiness(
        EvidenceReadiness._from_validated_receipt(
            selected_experiment_count=8,
            ready_experiment_count=0,
            ready_experiment_ids=(),
            blocked_experiment_ids=KINETIC_IDS,
            receipt_digest=_digest("9"),
        )
    )
    outside = tmp_path / "outside" / "redirected-publication"
    destination = tmp_path / "symlink-publication"
    destination.symlink_to(outside, target_is_directory=True)

    with pytest.raises(FileExistsError, match="create-only"):
        publish_metastudy(decision, destination)

    assert destination.is_symlink()
    assert not outside.exists()
    assert list(tmp_path.glob(".symlink-publication.*")) == []


def test_publication_install_failure_cleans_private_staging_without_publishing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reporter_response.metastudy import publication

    decision = decision_from_readiness(
        EvidenceReadiness._from_validated_receipt(
            selected_experiment_count=8,
            ready_experiment_count=0,
            ready_experiment_ids=(),
            blocked_experiment_ids=KINETIC_IDS,
            receipt_digest=_digest("9"),
        )
    )
    destination = tmp_path / "failed-publication"

    def fail_install(_stage: Path, _target: Path) -> None:
        raise OSError("simulated atomic rename failure")

    monkeypatch.setattr(publication, "_rename_directory_create_only", fail_install)

    with pytest.raises(OSError, match="simulated atomic rename failure"):
        publish_metastudy(decision, destination)

    assert not destination.exists()
    assert list(tmp_path.glob(".failed-publication.*")) == []


def test_termination_before_atomic_install_exposes_no_publication(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reporter_response.metastudy import publication

    decision = decision_from_readiness(
        EvidenceReadiness._from_validated_receipt(
            selected_experiment_count=8,
            ready_experiment_count=0,
            ready_experiment_ids=(),
            blocked_experiment_ids=KINETIC_IDS,
            receipt_digest=_digest("9"),
        )
    )
    destination = tmp_path / "interrupted-publication"

    def interrupt_install(stage: Path, target: Path) -> None:
        verify_publication(stage)
        assert not target.exists()
        raise SystemExit("simulated process termination")

    monkeypatch.setattr(publication, "_rename_directory_create_only", interrupt_install)

    with pytest.raises(SystemExit, match="simulated process termination"):
        publish_metastudy(decision, destination)

    assert not destination.exists()
    assert list(tmp_path.glob(".interrupted-publication.*")) == []


def test_selected_publication_is_create_only_and_evidence_bearing(tmp_path: Path) -> None:
    selected = evaluate_metastudy(_evidence(), readiness=_ready())
    destination = tmp_path / "selected"

    _publish_selected(selected, destination)
    assert {path.name for path in destination.iterdir()} == {
        "manifest.json",
        "report.md",
        "evidence.json",
        "acquisition.json",
        "sensitivity.json",
    }
    verify_publication(destination)
    with pytest.raises(FileExistsError):
        _publish_selected(selected, destination)


def test_selected_publication_rejects_missing_or_tampered_evidence(tmp_path: Path) -> None:
    selected = evaluate_metastudy(_evidence(), readiness=_ready())
    with pytest.raises(
        MetastudyContractError,
        match="evidence-bearing publication requires canonical profile evidence",
    ):
        publish_metastudy(selected, tmp_path / "missing")

    destination = _publish_selected(selected, tmp_path / "tampered")
    payload = json.loads((destination / "evidence.json").read_text(encoding="utf-8"))
    payload["profiles"][0]["profile"]["profile_id"] = "tampered"
    (destination / "evidence.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(MetastudyContractError, match="evidence file digest mismatch"):
        verify_publication(destination)


def test_publication_rejects_rehashed_acquisition_projection_not_derived_from_profiles(tmp_path: Path) -> None:
    import hashlib

    selected = evaluate_metastudy(_evidence(), readiness=_ready())
    destination = _publish_selected(selected, tmp_path / "acquisition-tamper")
    projection_path = destination / "acquisition.json"
    manifest_path = destination / "manifest.json"
    payload = json.loads(projection_path.read_text(encoding="utf-8"))
    payload["coordinates"][0]["contributions"][0]["normalized_reporter_response"] += 1.0
    payload_without_digest = {key: value for key, value in payload.items() if key != "projection_digest"}
    payload["projection_digest"] = canonical_digest(payload_without_digest)
    projection_bytes = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    projection_path.write_bytes(projection_bytes)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["acquisition_file_digest"] = "sha256:" + hashlib.sha256(projection_bytes).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(MetastudyContractError, match="differs from bundled profiles"):
        verify_publication(destination)


def test_publication_rejects_tampered_or_reordered_sensitivity_projection(tmp_path: Path) -> None:
    import hashlib

    selected = evaluate_metastudy(_evidence(), readiness=_ready())
    destination = _publish_selected(selected, tmp_path / "sensitivity-tamper")
    sensitivity_path = destination / "sensitivity.json"
    manifest_path = destination / "manifest.json"
    payload = json.loads(sensitivity_path.read_text(encoding="utf-8"))
    payload["evaluations"][0]["evidence_digest"] = _digest("f")
    sensitivity_bytes = (json.dumps(payload, sort_keys=True) + "\n").encode()
    sensitivity_path.write_bytes(sensitivity_bytes)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["sensitivity_file_digest"] = "sha256:" + hashlib.sha256(sensitivity_bytes).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(MetastudyContractError, match="summaries differ"):
        verify_publication(destination)

    destination = _publish_selected(selected, tmp_path / "sensitivity-reorder")
    sensitivity_path = destination / "sensitivity.json"
    manifest_path = destination / "manifest.json"
    payload = json.loads(sensitivity_path.read_text(encoding="utf-8"))
    payload["profiles"].reverse()
    sensitivity_bytes = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    sensitivity_path.write_bytes(sensitivity_bytes)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["sensitivity_file_digest"] = "sha256:" + hashlib.sha256(sensitivity_bytes).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(MetastudyContractError, match="not canonical"):
        verify_publication(destination)


def test_publication_rejects_self_consistent_sensitivity_chain_with_wrong_reader_revision(
    tmp_path: Path,
) -> None:
    import hashlib

    selected = evaluate_metastudy(_evidence(), readiness=_ready())
    destination = _publish_selected(selected, tmp_path / "revision-drift")
    sensitivity_path = destination / "sensitivity.json"
    evidence_path = destination / "evidence.json"
    report_path = destination / "report.md"
    manifest_path = destination / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    decision = manifest["decision"]
    attempt = decision["materialization_attempts"][0]
    experiment_id = attempt["experiment_id"]
    new_revision = attempt["reader_record_identity"]["reader_record_revision"] + 1_000
    attempt["reader_record_identity"]["reader_record_revision"] = new_revision
    attempt["attempt_digest"] = canonical_digest(
        {key: value for key, value in attempt.items() if key != "attempt_digest"}
    )

    payload = json.loads(sensitivity_path.read_text(encoding="utf-8"))
    changed_profile_digests: dict[str, str] = {}
    for profile_row in payload["profiles"]:
        profile = profile_row["profile"]
        if profile["provenance"]["reader_experiment_id"] != experiment_id:
            continue
        old_profile_digest = profile_row["audit"]["profile_digest"]
        profile["provenance"]["reader_record_revision"] = new_revision
        source_identity = {
            **profile["provenance"],
            "observation_policy_identity": profile["observation_policy"]["digest"],
        }
        audit = profile_row["audit"]
        audit["profile_source_digest"] = canonical_digest(source_identity)
        audit["profile_digest"] = canonical_digest(profile)
        audit_without_digest = {key: value for key, value in audit.items() if key != "artifact_digest"}
        audit["artifact_digest"] = canonical_digest(audit_without_digest)
        changed_profile_digests[old_profile_digest] = audit["profile_digest"]
    assert len(changed_profile_digests) == 30

    coverage = next(row for row in payload["coverages"] if row["experiment_id"] == experiment_id)
    coverage["reader_record_identity"]["reader_record_revision"] = new_revision
    coverage["materialization_attempt_digest"] = attempt["attempt_digest"]
    for entry in coverage["entries"]:
        if entry["profile_digest"] in changed_profile_digests:
            entry["profile_digest"] = changed_profile_digests[entry["profile_digest"]]
    coverage["coverage_digest"] = canonical_digest(
        {key: value for key, value in coverage.items() if key != "coverage_digest"}
    )
    projections = tuple(
        parse_profile_evidence_projection(row, index=index) for index, row in enumerate(payload["profiles"])
    )
    payload["evaluations"] = sensitivity_evaluations_to_payload(evaluate_sensitivity(projections))

    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    primary_profiles = copy.deepcopy(evidence["profiles"])
    evidence["materialization_attempts"] = copy.deepcopy(decision["materialization_attempts"])
    old_evidence_digest = decision["evidence_digest"]
    decision["evidence_digest"] = canonical_digest(evidence)
    assert evidence["profiles"] == primary_profiles

    sensitivity_bytes = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    evidence_bytes = (json.dumps(evidence, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    sensitivity_path.write_bytes(sensitivity_bytes)
    evidence_path.write_bytes(evidence_bytes)
    report = report_path.read_text(encoding="utf-8")
    assert old_evidence_digest in report
    report = report.replace(old_evidence_digest, decision["evidence_digest"])
    report_path.write_text(report, encoding="utf-8")
    manifest["sensitivity_file_digest"] = "sha256:" + hashlib.sha256(sensitivity_bytes).hexdigest()
    manifest["evidence_file_digest"] = "sha256:" + hashlib.sha256(evidence_bytes).hexdigest()
    manifest["report_digest"] = "sha256:" + hashlib.sha256(report.encode()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(MetastudyContractError, match="Reader identity differs from profile provenance"):
        verify_publication(destination)


def test_publication_requires_sensitivity_or_omission_for_each_ready_attempt(tmp_path: Path) -> None:
    selected = evaluate_metastudy(_evidence(), readiness=_ready())
    sensitivity = tuple(
        row for row in _sensitivity_evidence() if row.profile.provenance.reader_experiment_id != KINETIC_IDS[-1]
    )

    with pytest.raises(MetastudyContractError, match="exact ready-attempt set"):
        publish_metastudy(
            selected,
            tmp_path / "missing-one-experiment",
            primary_evidence=_evidence(),
            sensitivity_evidence=sensitivity,
            sensitivity_evaluations=evaluate_sensitivity(sensitivity),
        )


def test_publication_rejects_endpoint_8_only_as_incomplete_sensitivity_coverage(tmp_path: Path) -> None:
    selected = evaluate_metastudy(_evidence(), readiness=_ready())
    endpoint_8_only = _sensitivity_evidence()

    with pytest.raises(MetastudyContractError, match="sensitivity coverage"):
        publish_metastudy(
            selected,
            tmp_path / "endpoint-8-only",
            primary_evidence=_evidence(),
            sensitivity_evidence=endpoint_8_only,
            sensitivity_evaluations=evaluate_sensitivity(endpoint_8_only),
        )


def test_publication_allows_empty_sensitivity_only_with_typed_ready_attempt_omissions(
    tmp_path: Path,
) -> None:
    primary = _evidence()
    complete = _complete_sensitivity_evidence(primary)
    attempts = _attempts(primary)
    omitted_coverages = tuple(
        SensitivityCoverageLedger(
            contract_id=SENSITIVITY_COVERAGE_CONTRACT_ID,
            experiment_id=coverage.experiment_id,
            materialization_attempt_digest=coverage.materialization_attempt_digest,
            reader_record_identity=coverage.reader_record_identity,
            evidence_binding_artifact_id=coverage.evidence_binding_artifact_id,
            evidence_binding_artifact_digest=coverage.evidence_binding_artifact_digest,
            expected_subjects=coverage.expected_subjects,
            expected_reduction_ids=coverage.expected_reduction_ids,
            entries=tuple(
                SensitivityCoverageEntry(
                    subject=entry.subject,
                    reduction_id=entry.reduction_id,
                    outcome="omission",
                    profile_digest=None,
                    omission=MaterializationOmission(
                        code="synthetic_sensitivity_unavailable",
                        subject_id=entry.subject.subject_id,
                        reduction_id=entry.reduction_id,
                    ),
                )
                for entry in coverage.entries
            ),
        )
        for coverage in _sensitivity_coverages(complete, attempts)
    )
    selected = evaluate_metastudy_with_attempts(
        primary,
        readiness=_ready(),
        attempts=attempts,
    )

    destination = publish_metastudy(
        selected,
        tmp_path / "omitted-sensitivity",
        primary_evidence=primary,
        sensitivity_coverages=omitted_coverages,
    )

    payload = json.loads((destination / "sensitivity.json").read_text(encoding="utf-8"))
    assert payload["evaluations"] == []
    assert payload["profiles"] == []
    verify_publication(destination)


def test_sensitivity_evaluation_cannot_change_primary_selection() -> None:
    primary = _evidence()
    selected = evaluate_metastudy(primary, readiness=_ready())
    before = decision_to_dict(selected)

    evaluate_sensitivity(_sensitivity_evidence())

    assert decision_to_dict(selected) == before
    assert selected.selected_reduction == (6.0, 10.0)
    assert "sensitivity_evaluations" not in before


def test_selected_decision_cannot_be_reconstructed_from_copied_fields() -> None:
    selected = evaluate_metastudy(_evidence(), readiness=_ready())

    with pytest.raises(MetastudyContractError, match="canonical evaluation"):
        MetastudyDecision(
            contract_id=selected.contract_id,
            protocol_id=selected.protocol_id,
            status=selected.status,
            selection_use=selected.selection_use,
            evidence_grade=selected.evidence_grade,
            selected_reduction=selected.selected_reduction,
            blockers=selected.blockers,
            limitations=selected.limitations,
            policy_digest=selected.policy_digest,
            evidence_digest=selected.evidence_digest,
            readiness=selected.readiness,
            evaluations=selected.evaluations,
            materialization_attempts=selected.materialization_attempts,
        )


def test_verify_publication_rejects_report_rewrite_even_with_matching_digest(tmp_path: Path) -> None:
    import hashlib

    decision = decision_from_readiness(
        EvidenceReadiness._from_validated_receipt(
            selected_experiment_count=8,
            ready_experiment_count=0,
            ready_experiment_ids=(),
            blocked_experiment_ids=KINETIC_IDS,
            receipt_digest=_digest("9"),
        )
    )
    destination = publish_metastudy(decision, tmp_path / "tampered-report")
    rewritten = (destination / "report.md").read_text(encoding="utf-8") + "\nforged\n"
    (destination / "report.md").write_text(rewritten, encoding="utf-8")
    manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
    manifest["report_digest"] = "sha256:" + hashlib.sha256(rewritten.encode("utf-8")).hexdigest()
    (destination / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(MetastudyContractError, match="canonical rendered decision"):
        verify_publication(destination)


def test_verify_publication_rejects_forged_selection_with_recomputed_digests(tmp_path: Path) -> None:
    import hashlib

    evidence = _evidence()
    selected = evaluate_metastudy(evidence, readiness=_ready())
    assert selected.selected_reduction == (6.0, 10.0)
    destination = _publish_selected(selected, tmp_path / "forged-selection", evidence=evidence)

    manifest_path = destination / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    decision = manifest["decision"]
    decision["selected_reduction"] = [12.0, 16.0]
    for evaluation in decision["evaluations"]:
        if evaluation["reduction"] == [12.0, 16.0]:
            evaluation["worst_experiment_control_separation"] = 1_000.0
            evaluation["repeated_anchor_drift"] = 0.0
            evaluation["within_acquisition_observation_range"] = 0.0
            evaluation["growth_phase_start"] = 1.0
            evaluation["growth_phase_end"] = 0.5
            evaluation["eligible_experiment_count"] = 8
            evaluation["anchor_ordered_acquisition_count"] = 5
            evaluation["co_measured_anchor_acquisition_count"] = 5
            evaluation["loo_same_or_adjacent_fraction"] = 1.0
            evaluation["eligible"] = True
            evaluation["blockers"] = []
    report = (destination / "report.md").read_text(encoding="utf-8").replace("`6-10 h`", "`12-16 h`")
    (destination / "report.md").write_text(report, encoding="utf-8")
    manifest["report_digest"] = "sha256:" + hashlib.sha256(report.encode("utf-8")).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(MetastudyContractError, match="canonical evidence evaluation"):
        verify_publication(destination)


def test_selected_source_state_is_compact_and_rejects_phase_ineligible_forgery() -> None:
    from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reporter_response.metastudy.operator import (
        state as operator_state,
    )

    evidence = _evidence()
    selected = evaluate_metastudy(evidence, readiness=_ready())
    decision = json.loads(json.dumps(decision_to_dict(selected)))
    readiness = {
        "schema_id": "rt_lnrna_reporter_response_readiness_snapshot.v1",
        "source_identity": {
            "route_id": "rt_lnrna_reporter_response_metastudy",
            "route_registry_path": ".agents/skills/retron-assay-study-bridge/references/reader-experiment-routes.json",
            "route_registry_digest": _digest("a"),
            "normalized_full_receipt_digest": decision["readiness"]["receipt_digest"],
            "normalization": "omit environment-specific reader_command before canonical JSON hashing",
        },
        "last_verified": "2026-07-30",
        "selected_experiment_count": 8,
        "related_experiment_count": 1,
        "related_experiment_ids": ["20251105_retron_Eco1_RT_variants"],
        "ready_experiment_count": 8,
        "ready_experiment_ids": list(KINETIC_IDS),
        "blocked_experiment_ids": [],
    }
    body = {
        "readiness": readiness,
        "decision": decision,
        "objective_readiness": asdict(DEFAULT_OBJECTIVE_READINESS),
        "sensitivity_evaluations": [],
        "sensitivity_coverage_receipts": [
            sensitivity_coverage_contracts.sensitivity_coverage_receipt_payload(row)
            for row in _sensitivity_coverages(
                _complete_sensitivity_evidence(evidence),
                selected.materialization_attempts,
            )
        ],
        "acquisition_projection": acquisition_projection_payload(
            build_acquisition_projection(
                evidence,
                selected_reduction=selected.selected_reduction,
            )
        ),
    }
    payload = {
        "schema_id": "rt_lnrna_reporter_response_metastudy_state.v6",
        "generation_digest": operator_state.canonical_digest(body),
        **body,
    }
    operator_state.validate_state_payload(payload)

    with_embedded_evidence = {
        **payload,
        "evidence": json.loads(json.dumps(decision_evidence_payload(evidence, decision=selected))),
    }
    with_embedded_evidence["generation_digest"] = operator_state.canonical_digest(
        {
            **body,
            "evidence": with_embedded_evidence["evidence"],
        }
    )
    with pytest.raises(MetastudyContractError, match="fields do not match"):
        operator_state.validate_state_payload(with_embedded_evidence)

    decision["selected_reduction"] = [12.0, 16.0]
    for evaluation in decision["evaluations"]:
        if evaluation["reduction"] == [12.0, 16.0]:
            evaluation.update(
                worst_experiment_control_separation=1_000.0,
                repeated_anchor_drift=0.0,
                within_acquisition_observation_range=0.0,
                eligible_experiment_count=8,
                anchor_ordered_acquisition_count=5,
                co_measured_anchor_acquisition_count=5,
                loo_same_or_adjacent_fraction=1.0,
                eligible=True,
                blockers=[],
            )
    payload["generation_digest"] = operator_state.canonical_digest(body)

    with pytest.raises(MetastudyContractError, match="descriptive support and phase gates"):
        operator_state.validate_state_payload(payload)


@pytest.mark.parametrize(
    ("readiness_path", "replacement"),
    (
        (("schema_id",), "wrong-readiness-schema"),
        (("source_identity", "route_id"), "wrong-route"),
        (("source_identity", "route_registry_path"), "wrong-registry.json"),
        (("source_identity", "route_registry_digest"), "sha256:" + "g" * 64),
        (("source_identity", "normalized_full_receipt_digest"), "sha256:" + "g" * 64),
        (("source_identity", "normalization"), "hash the whole receipt"),
        (("last_verified",), "2026-02-30"),
        (("selected_experiment_count",), 7),
        (("related_experiment_count",), 2),
        (("related_experiment_ids",), ["wrong-related-experiment"]),
    ),
)
def test_source_state_rejects_noncanonical_readiness_snapshot_with_recomputed_generation_digest(
    readiness_path: tuple[str, ...],
    replacement: object,
) -> None:
    from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reporter_response.metastudy.operator import (
        state as operator_state,
    )

    state_path = next(
        parent
        / "docs/studies/rt_lnrna_sponging_construct_triage/contexts/reporter-response-metastudy/metastudy-state.yaml"
        for parent in Path(__file__).resolve().parents
        if (parent / "docs/studies/rt_lnrna_sponging_construct_triage").is_dir()
    )
    payload = yaml.safe_load(state_path.read_text(encoding="utf-8"))
    readiness = copy.deepcopy(payload["readiness"])
    target = readiness
    for key in readiness_path[:-1]:
        target = target[key]
    target[readiness_path[-1]] = replacement
    payload["readiness"] = readiness
    if readiness_path == ("source_identity", "normalized_full_receipt_digest"):
        decision = copy.deepcopy(payload["decision"])
        decision["readiness"]["receipt_digest"] = replacement
        payload["decision"] = decision
    payload["generation_digest"] = operator_state.canonical_digest(
        {
            key: payload[key]
            for key in (
                "readiness",
                "decision",
                "objective_readiness",
                "sensitivity_evaluations",
                "sensitivity_coverage_receipts",
                "acquisition_projection",
            )
            if key in payload
        }
    )

    with pytest.raises(MetastudyContractError, match="readiness"):
        operator_state.validate_state_payload(payload)


def test_publication_projection_parser_mints_no_live_source_or_audit_closure() -> None:
    evidence = _evidence()
    selected = evaluate_metastudy(evidence, readiness=_ready())
    payload = decision_evidence_payload(evidence, decision=selected)
    row = json.loads(json.dumps(payload["profiles"][0]))
    projection = parse_profile_evidence_projection(row, index=0)

    assert not isinstance(projection.profile, ReporterResponseProfile)
    assert not hasattr(projection.profile.provenance, "is_source_closed")
    assert projection.audit.is_derivation_closed is False


def test_live_and_offline_source_identity_bind_raw_reader_aliases_symmetrically() -> None:
    evidence = _evidence()
    selected = evaluate_metastudy(evidence, readiness=_ready())
    payload = decision_evidence_payload(evidence, decision=selected)
    profile = evidence[0].profile
    row = next(item for item in payload["profiles"] if item["profile"]["profile_id"] == profile.profile_id)
    projection = parse_profile_evidence_projection(json.loads(json.dumps(row)), index=0).profile

    live_identity = profile_source_identity_payload(profile)
    offline_identity = profile_source_identity_projection(projection)
    assert offline_identity == live_identity
    assert live_identity["raw_design_id"] == profile.provenance.raw_design_id
    assert live_identity["raw_assay_subject_id"] == profile.provenance.raw_assay_subject_id
    assert live_identity["reader_protocol_id"] == profile.provenance.reader_protocol_id
    assert live_identity["reader_record_kind"] == profile.provenance.reader_record_kind
    assert live_identity["reader_record_path"] == profile.provenance.reader_record_path

    changed_offline = replace(
        projection,
        provenance=replace(projection.provenance, raw_design_id="changed-reader-alias"),
    )
    assert canonical_digest(profile_source_identity_projection(changed_offline)) != canonical_digest(offline_identity)


@pytest.mark.parametrize("reader_record_path", ["../outside.parquet", "/outside.parquet"])
def test_publication_projection_rejects_unconfined_reader_record_path(reader_record_path: str) -> None:
    evidence = _evidence()
    selected = evaluate_metastudy(evidence, readiness=_ready())
    row = decision_evidence_payload(evidence, decision=selected)["profiles"][0]
    row["profile"]["provenance"]["reader_record_path"] = reader_record_path

    with pytest.raises(ValueError, match="reader_record_path must be outputs-relative"):
        parse_profile_evidence_projection(row, index=0)


def test_publication_projection_rejects_forged_null_raw_identity_after_digest_recomputation() -> None:
    evidence = _evidence()
    selected = evaluate_metastudy(evidence, readiness=_ready())
    row = decision_evidence_payload(evidence, decision=selected)["profiles"][0]
    profile = row["profile"]
    provenance = profile["provenance"]
    provenance["raw_design_id"] = None
    provenance["raw_assay_subject_id"] = None
    audit = row["audit"]
    audit["profile_digest"] = canonical_digest(profile)
    audit["profile_source_digest"] = canonical_digest(
        {
            "raw_design_id": None,
            "raw_assay_subject_id": None,
            "reader_experiment_id": provenance["reader_experiment_id"],
            "reader_record_id": provenance["reader_record_id"],
            "reader_record_revision": provenance["reader_record_revision"],
            "reader_record_revision_digest": provenance["reader_record_revision_digest"],
            "reader_record_content_digest": provenance["reader_record_content_digest"],
            "reader_record_schema_version": provenance["reader_record_schema_version"],
            "reader_record_contract_id": provenance["reader_record_contract_id"],
            "evidence_binding_artifact_id": provenance["evidence_binding_artifact_id"],
            "evidence_binding_artifact_digest": provenance["evidence_binding_artifact_digest"],
            "observation_policy_identity": profile["observation_policy"]["digest"],
        }
    )
    audit_without_digest = {key: value for key, value in audit.items() if key != "artifact_digest"}
    audit["artifact_digest"] = canonical_digest(audit_without_digest)

    with pytest.raises(ValueError, match="at least one raw Reader identity"):
        parse_profile_evidence_projection(row, index=0)


@pytest.mark.parametrize("field_name", ["raw_design_id", "raw_assay_subject_id"])
def test_publication_projection_rejects_empty_raw_identity_coordinate(field_name: str) -> None:
    evidence = _evidence()
    selected = evaluate_metastudy(evidence, readiness=_ready())
    row = decision_evidence_payload(evidence, decision=selected)["profiles"][0]
    row["profile"]["provenance"][field_name] = ""

    with pytest.raises(ValueError, match=rf"{field_name} must be non-empty text"):
        parse_profile_evidence_projection(row, index=0)


def test_mutated_payload_is_rejected_before_publication(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reporter_response.metastudy import publication

    decision = decision_from_readiness(
        EvidenceReadiness._from_validated_receipt(
            selected_experiment_count=8,
            ready_experiment_count=0,
            ready_experiment_ids=(),
            blocked_experiment_ids=KINETIC_IDS,
            receipt_digest=_digest("9"),
        )
    )
    payload = decision_to_dict(decision)
    payload["selected_reduction"] = {"recorded_start_time_h": 4.0, "recorded_end_time_h": 8.0}
    with pytest.raises(MetastudyContractError, match="blocked decision"):
        validate_decision_payload(payload)

    monkeypatch.setattr(publication, "decision_to_dict", lambda _decision: payload)
    destination = tmp_path / "must-not-exist"
    with pytest.raises(MetastudyContractError, match="blocked decision"):
        publish_metastudy(decision, destination)
    assert not destination.exists()


def test_selected_decision_serialization_binds_attempts_but_is_not_a_publication() -> None:
    selected = evaluate_metastudy(_evidence(), readiness=_ready())
    payload = decision_to_dict(selected)

    validate_decision_payload(payload)
    assert len(payload["materialization_attempts"]) == 8

    fabricated = dict(payload)
    fabricated["materialization_attempts"] = payload["materialization_attempts"][:-1]
    with pytest.raises(MetastudyContractError, match="canonical materialization-attempt order"):
        validate_decision_payload(fabricated)


def test_materialization_attempt_rejects_noncanonical_or_duplicate_profile_digests() -> None:
    attempt = _attempts(_evidence())[0]

    with pytest.raises(MetastudyContractError, match="canonical digest order"):
        replace(attempt, candidate_profile_digests=tuple(reversed(attempt.candidate_profile_digests)))

    duplicate = (attempt.candidate_profile_digests[0],) * attempt.candidate_profile_count
    with pytest.raises(MetastudyContractError, match="must be unique"):
        replace(attempt, candidate_profile_digests=duplicate)


def test_omission_only_blocked_attempt_requires_complete_coordinate_closure() -> None:
    attempt = _attempts(_evidence())[0]
    incomplete = (
        MaterializationOmission(
            code="condition_or_channel_observations_incomplete",
            subject_id=attempt.expected_subject_ids[0],
            reduction_id="window-4-8h",
        ),
    )
    blocked_fields = {
        "status": "blocked",
        "candidate_profile_count": 0,
        "candidate_profile_digests": (),
        "candidate_omissions": incomplete,
    }

    with pytest.raises(MetastudyContractError, match="complete expected coordinate closure"):
        replace(attempt, blockers=(), **blocked_fields)

    complete_omissions = tuple(
        sorted(
            (
                MaterializationOmission(
                    code="condition_or_channel_observations_incomplete",
                    subject_id=subject_id,
                    reduction_id=f"window-{start:g}-{end:g}h",
                )
                for subject_id in attempt.expected_subject_ids
                for start, end in DEFAULT_PROTOCOL.candidate_windows_h
            ),
            key=lambda row: (row.subject_id, row.reduction_id, row.code),
        )
    )
    closed = replace(
        attempt,
        blockers=(),
        **{**blocked_fields, "candidate_omissions": complete_omissions},
    )
    assert closed.status == "blocked"

    fatal = replace(
        attempt,
        blockers=(MaterializationBlocker("reader_artifact_unreadable"),),
        **blocked_fields,
    )
    assert fatal.status == "blocked"


@pytest.mark.parametrize(
    "malformed_reduction",
    ([], [6.0], [5.0, 9.0], ["6", "10"], [6.0, 10.0, 12.0]),
)
def test_selected_decision_rejects_malformed_reduction_with_contract_error(
    malformed_reduction: list[object],
) -> None:
    payload = decision_to_dict(evaluate_metastudy(_evidence(), readiness=_ready()))
    payload["selected_reduction"] = malformed_reduction

    with pytest.raises(MetastudyContractError, match="declared candidate window"):
        validate_decision_payload(payload)


def test_evaluation_rejects_noncanonical_materialization_attempt_order() -> None:
    evidence = _evidence()

    with pytest.raises(MetastudyContractError, match="canonical selected-experiment order"):
        evaluate_metastudy_with_attempts(
            evidence,
            readiness=_ready(),
            attempts=tuple(reversed(_attempts(evidence))),
        )


def test_decision_payload_rejects_noncanonical_attempt_and_evaluation_order() -> None:
    selected = evaluate_metastudy(_evidence(), readiness=_ready())
    payload = decision_to_dict(selected)
    payload["materialization_attempts"] = tuple(reversed(payload["materialization_attempts"]))
    with pytest.raises(MetastudyContractError, match="canonical materialization-attempt order"):
        validate_decision_payload(payload)

    payload = decision_to_dict(selected)
    payload["evaluations"] = tuple(reversed(payload["evaluations"]))
    with pytest.raises(MetastudyContractError, match="canonical candidate-window order"):
        validate_decision_payload(payload)


def test_seven_of_eight_decision_serializes_the_unavailable_reader_attempt() -> None:
    blocked_id = KINETIC_IDS[-1]
    evidence = tuple(row for row in _evidence() if row.profile.provenance.reader_experiment_id != blocked_id)
    attempts = list(_attempts(evidence))
    attempts[-1] = replace(
        attempts[-1],
        reader_record_identity=None,
        blockers=(MaterializationBlocker("reader_records_not_ready"),),
    )
    readiness = EvidenceReadiness._from_owner_bridge_receipt(
        selected_experiment_count=8,
        ready_experiment_count=7,
        ready_experiment_ids=KINETIC_IDS[:-1],
        blocked_experiment_ids=(blocked_id,),
        receipt_digest=_digest("7"),
    )

    decision = evaluate_metastudy_with_attempts(
        evidence,
        readiness=readiness,
        attempts=attempts,
    )
    payload = decision_to_dict(decision)

    validate_decision_payload(payload)
    unavailable = next(row for row in payload["materialization_attempts"] if row["experiment_id"] == blocked_id)
    assert decision.status == "selected"
    assert unavailable["reader_record_identity"] is None
    assert unavailable["blockers"] == ({"code": "reader_records_not_ready"},)


def test_decision_rejects_attempt_reader_identity_drift_from_primary_profiles() -> None:
    evidence = _evidence()
    attempts = _attempts(evidence)
    changed_identity = replace(
        attempts[0].reader_record_identity,
        reader_record_content_digest=_digest("9"),
    )
    changed_attempt = replace(attempts[0], reader_record_identity=changed_identity)
    with pytest.raises(MetastudyContractError, match="Reader identity differs from profile provenance"):
        evaluate_metastudy_with_attempts(
            evidence,
            readiness=_ready(),
            attempts=(changed_attempt, *attempts[1:]),
        )


def test_metastudy_has_no_reader_opal_or_historical_spop_import_dependency() -> None:
    package = Path(__file__).resolve().parents[3] / "reporter_response" / "metastudy"
    forbidden_import_roots = {"reader", "reader_workbench", "opal"}
    paths = tuple(package.rglob("*.py"))
    assert paths
    for path in paths:
        text = path.read_text(encoding="utf-8")
        assert "spop" not in text.lower()
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert not ({alias.name.split(".")[0] for alias in node.names} & forbidden_import_roots)
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert node.module.split(".")[0] not in forbidden_import_roots


def test_checked_in_protocol_and_live_descriptive_selection_match_runtime_contracts() -> None:
    from dataclasses import asdict

    from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reporter_response.metastudy.contracts import (
        canonical_digest,
        protocol_digest,
    )

    study_root = next(
        parent / "docs/studies/rt_lnrna_sponging_construct_triage"
        for parent in Path(__file__).resolve().parents
        if (parent / "docs/studies/rt_lnrna_sponging_construct_triage").is_dir()
    )
    docs = study_root / "contexts/reporter-response-metastudy"
    protocol_payload = yaml.safe_load((docs / "protocol.yaml").read_text(encoding="utf-8"))
    expected_protocol = json.loads(json.dumps(asdict(DEFAULT_PROTOCOL)))
    assert protocol_payload == expected_protocol
    assert canonical_digest(protocol_payload) == protocol_digest()

    from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reporter_response.metastudy.operator import (
        state as operator_state,
    )

    state = yaml.safe_load((docs / "metastudy-state.yaml").read_text(encoding="utf-8"))
    operator_state.validate_state_payload(state)
    snapshot = state["readiness"]
    decision_payload = state["decision"]
    validate_decision_payload(decision_payload)
    assert decision_payload["readiness"] == {
        "selected_experiment_count": snapshot["selected_experiment_count"],
        "ready_experiment_count": snapshot["ready_experiment_count"],
        "ready_experiment_ids": snapshot["ready_experiment_ids"],
        "blocked_experiment_ids": snapshot["blocked_experiment_ids"],
        "receipt_digest": snapshot["source_identity"]["normalized_full_receipt_digest"],
    }
    assert decision_payload["status"] == "selected"
    assert decision_payload["selection_use"] == "descriptive_comparison"
    assert decision_payload["evidence_grade"] == "provisional_descriptive"
    assert decision_payload["selected_reduction"] == [6.0, 10.0]
    assert decision_payload["blockers"] == []
    assert len(decision_payload["evaluations"]) == len(DEFAULT_PROTOCOL.candidate_windows_h)
    assert {row["eligible_experiment_count"] for row in decision_payload["evaluations"]} == {8}
    eligible = [row for row in decision_payload["evaluations"] if row["eligible"]]
    assert [row["reduction"] for row in eligible] == [[6.0, 10.0]]
    assert state["objective_readiness"] == {
        "contract_id": "rt_lnrna_reporter_response_objective_readiness.v3",
        "status": "blocked",
        "objective_id": None,
        "blockers": [
            "constrained_objective_not_defined",
            "biological_replicate_uncertainty_not_estimable",
            "od_linearity_not_validated",
        ],
    }
    projection = state["acquisition_projection"]
    assert projection["selected_reduction"] == [6.0, 10.0]
    assert len(projection["coordinates"]) == 32
    assert {row["reduction_id"] for row in projection["coordinates"]} == {"window-6-10h"}

    route_text = (study_root / "routes/README.md").read_text(encoding="utf-8")
    assert "6-10 h reduction selected as `provisional_descriptive`" in route_text
    assert "objective readiness remains blocked" in route_text
    assert "0/8 selected kinetic Reader experiments" not in route_text
