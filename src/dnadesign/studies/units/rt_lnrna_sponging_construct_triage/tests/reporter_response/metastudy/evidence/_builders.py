"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/tests/reporter_response/metastudy/evidence/_builders.py

Builds typed sensitivity evidence shared by evidence, selection, and publication tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import replace

from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reporter_response import (
    EndpointReduction,
    TimeWindowReduction,
)
from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reporter_response.metastudy import (
    DEFAULT_PROTOCOL,
    MaterializationAttemptReceipt,
    ProfileEvidence,
    ReaderRecordIdentity,
    sensitivity_coverage,
)
from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reporter_response.metastudy.audits import (
    _build_derivation_closed_profile_audit as build_profile_audit_artifact,
)

from .._builders import _evidence

SENSITIVITY_COVERAGE_CONTRACT_ID = sensitivity_coverage.SENSITIVITY_COVERAGE_CONTRACT_ID
SensitivityCoverageEntry = sensitivity_coverage.SensitivityCoverageEntry
SensitivityCoverageLedger = sensitivity_coverage.SensitivityCoverageLedger
SensitivitySubjectCoordinate = sensitivity_coverage.SensitivitySubjectCoordinate
declared_sensitivity_reduction_ids = sensitivity_coverage.declared_sensitivity_reduction_ids


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
                    reader_record_config_digest=provenance.reader_record_config_digest,
                    reader_record_producer_config_digest=provenance.reader_record_producer_config_digest,
                    reader_record_producer=provenance.reader_record_producer,
                    reader_record_inputs=provenance.reader_record_inputs,
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
