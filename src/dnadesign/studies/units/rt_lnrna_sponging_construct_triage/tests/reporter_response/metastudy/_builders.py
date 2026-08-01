"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/tests/reporter_response/metastudy/_builders.py

Builds shared, source-closed reporter-response metastudy test evidence.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import replace

from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reader_evidence import (
    ReaderEvidenceBinding,
    ReaderEvidenceBindingSet,
)
from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reporter_response import (
    ConditionMeasurement,
    ControlAssignment,
    DoseUncertainty,
    NotEstimableMetricUncertainty,
    PairingPolicy,
    ReferenceNormalizationUnavailable,
    ReporterResponseObservationPolicy,
    TimeWindowReduction,
    UncertaintyPolicy,
    build_reporter_measurement_profile,
    build_reporter_response_profile,
)
from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reporter_response.metastudy import (
    DEFAULT_PROTOCOL,
    EvidenceReadiness,
    GrowthPhaseStratum,
    MaterializationAttemptReceipt,
    MaterializationBlocker,
    MetastudyDecision,
    ProfileEvidence,
    ReaderRecordIdentity,
)
from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reporter_response.metastudy import (
    evaluate_metastudy as evaluate_metastudy_with_attempts,
)
from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reporter_response.metastudy.audits import (
    _build_derivation_closed_profile_audit as build_profile_audit_artifact,
)
from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reporter_response.metastudy.audits import (
    profile_digest,
)

LOW_ANCHOR = "rt_lnrna_pair__eco1_wt_rt__retron43_lnrna__tetO"
HIGH_ANCHOR = "rt_lnrna_pair__eco1_wt_rt__retron26_lnrna__tetO"
KINETIC_IDS = DEFAULT_PROTOCOL.planned_kinetic_experiment_ids
ANCHOR_IDS = DEFAULT_PROTOCOL.planned_anchor_experiment_ids


def _digest(character: str) -> str:
    return "sha256:" + character * 64


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
    reference_normalized: bool = True,
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
    common_profile = dict(
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
        measurements=(
            measurements if reference_normalized else [row for row in measurements if row.role != "positive_control"]
        ),
        ineligibility_reasons=("preference_objective_not_defined",),
    )
    if not reference_normalized:
        return build_reporter_measurement_profile(
            **common_profile,
            reference_normalization=ReferenceNormalizationUnavailable(
                reason="positive_control_not_declared",
                positive_control_condition_id=None,
            ),
        )
    return build_reporter_response_profile(
        **common_profile,
        pairing_policy=PairingPolicy(kind="pooled_controls_by_design", assignments=tuple(assignments)),
        dose_uncertainties=uncertainties,
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
    reference_normalized: bool = True,
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
                    reference_normalized=reference_normalized,
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
