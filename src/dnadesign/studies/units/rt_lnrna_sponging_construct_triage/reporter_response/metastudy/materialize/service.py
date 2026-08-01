"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/reporter_response/metastudy/materialize/service.py

Source-closed reporter-response materialization orchestration and receipts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from io import BytesIO
from typing import Literal

import pandas as pd

from ....reader_evidence import ReaderDataframeRecordRef, ReaderEvidenceBindingSet
from ...policy import ReporterResponseObservationPolicy
from ...profile.measurement import EndpointReduction, TimeWindowReduction
from ..audits import profile_digest
from ..condition_ontology import ReporterResponseConditionOntology
from ..contracts.materialization import (
    MaterializationAttemptReceipt,
    MaterializationBlocker,
    MaterializationOmission,
    ReaderRecordIdentity,
)
from ..contracts.protocol import DEFAULT_PROTOCOL, MetastudyProtocol
from ..sensitivity_coverage import SensitivitySubjectCoordinate, build_sensitivity_coverage
from .identities import _has_ambiguous_partial_identity, _observed_reader_identities, _reader_identity_mask
from .models import MaterializationReadiness
from .reductions import _materialize_reductions

_BASE_COLUMNS = {"type", "position", "time", "channel", "value", "treatment", "design_id"}
_QUALITY_COLUMNS = {"value_policy_clipped", "value_instrument_overflow", "value_bound_kind"}


def materialize_record_evidence(
    *,
    record: ReaderDataframeRecordRef,
    bindings: ReaderEvidenceBindingSet,
    ontology: ReporterResponseConditionOntology,
    observation_policy: ReporterResponseObservationPolicy,
    protocol: MetastudyProtocol = DEFAULT_PROTOCOL,
) -> MaterializationReadiness:
    """Derive profiles and audits from one verified long-form Reader record.

    Evidence insufficiency returns a deterministic blocked result. Programmer
    errors in typed policy construction continue to fail at their constructors.
    """

    blocker = _preflight(record, bindings, ontology, observation_policy, protocol)
    if blocker is not None:
        return _blocked(record, blocker)
    try:
        artifact_bytes = record.path.read_bytes()
    except OSError:
        return _blocked(record, "reader_artifact_unreadable")
    if "sha256:" + hashlib.sha256(artifact_bytes).hexdigest() != record.content_digest:
        return _blocked(record, "reader_artifact_content_digest_changed")
    try:
        frame = pd.read_parquet(BytesIO(artifact_bytes))
    except Exception:
        return _blocked(record, "reader_artifact_not_readable_parquet")
    missing = sorted((_BASE_COLUMNS | _QUALITY_COLUMNS) - set(frame.columns))
    if missing:
        code = "required_quality_columns_missing" if set(missing) <= _QUALITY_COLUMNS else "required_columns_missing"
        return _blocked(record, code)
    if frame[list(_QUALITY_COLUMNS)].isna().any().any():
        return _blocked(record, "quality_provenance_contains_null")
    for column in ("value_policy_clipped", "value_instrument_overflow"):
        if not frame[column].map(lambda value: isinstance(value, bool)).all():
            return _blocked(record, "quality_flag_not_boolean")
    samples = frame.loc[frame["type"].eq(ontology.sample_type_value)].copy()
    if samples.empty:
        return _blocked(record, "sample_rows_missing")
    declared_labels = set(ontology.by_treatment_label)
    observed_labels = set(samples["treatment"].dropna().astype(str))
    if samples["treatment"].isna().any() or not observed_labels <= declared_labels:
        return _blocked(record, "sample_condition_not_declared")
    channels = {ontology.reporter_channel, ontology.normalizer_channel, ontology.ratio_channel}
    if not channels <= set(samples["channel"].dropna().astype(str)):
        return _blocked(record, "required_channels_missing")
    binding_by_identity = {
        (row.raw_design_id, row.raw_assay_subject_id): row
        for row in bindings.rows
        if row.raw_design_id is not None and row.binding_state == "bound" and row.subject_id is not None
    }
    if samples["design_id"].isna().any():
        return _blocked(record, "sample_subject_not_bound")
    observed_identities = _observed_reader_identities(samples)
    if _has_ambiguous_partial_identity(observed_identities, bindings=bindings):
        return _blocked(record, "sample_subject_identity_ambiguous")
    if not observed_identities <= set(binding_by_identity):
        return _blocked(record, "sample_subject_not_bound")
    projected_subject_ids = tuple(binding_by_identity[identity].subject_id for identity in observed_identities)
    if len(set(projected_subject_ids)) != len(projected_subject_ids):
        return _blocked(record, "multiple_reader_identities_for_subject")
    observation_identity_field = "position"
    if observation_identity_field not in samples.columns or samples[observation_identity_field].isna().any():
        return _blocked(record, "observation_identity_missing")
    for identity in observed_identities:
        observed_values = set(
            samples.loc[_reader_identity_mask(samples, identity), observation_identity_field].astype(str)
        )
        if observed_values != set(binding_by_identity[identity].observation_identity_values):
            return _blocked(record, "binding_observation_identity_mismatch")

    candidate_reductions = tuple(
        TimeWindowReduction(
            recorded_start_time_h=start,
            recorded_end_time_h=end,
            summary_statistic=protocol.time_summary_statistic,
            ratio_reduction_order=protocol.ratio_reduction_order,
        )
        for start, end in protocol.candidate_windows_h
    )
    endpoint_reductions = tuple(EndpointReduction(recorded_time_h=value) for value in protocol.endpoint_sensitivity_h)
    centered_reductions = tuple(
        TimeWindowReduction(
            recorded_start_time_h=(start + end) / 2.0 - width / 2.0,
            recorded_end_time_h=(start + end) / 2.0 + width / 2.0,
            summary_statistic=protocol.time_summary_statistic,
            ratio_reduction_order=protocol.ratio_reduction_order,
        )
        for start, end in protocol.candidate_windows_h
        for width in protocol.centered_window_sensitivity_widths_h
    )
    reference = EndpointReduction(recorded_time_h=10.0)
    candidate, candidate_omissions = _materialize_reductions(
        samples,
        reductions=candidate_reductions,
        reference=reference,
        record=record,
        bindings=bindings,
        binding_by_identity=binding_by_identity,
        ontology=ontology,
        policy=observation_policy,
        protocol=protocol,
        include_sensitivity_doses=False,
    )
    if not candidate:
        return _blocked(
            record,
            omissions=candidate_omissions,
            bindings=bindings,
            expected_subject_ids=tuple(sorted(projected_subject_ids)),
        )
    endpoints, endpoint_blockers = _materialize_reductions(
        samples,
        reductions=endpoint_reductions,
        reference=reference,
        record=record,
        bindings=bindings,
        binding_by_identity=binding_by_identity,
        ontology=ontology,
        policy=observation_policy,
        protocol=protocol,
        include_sensitivity_doses=True,
    )
    centered, centered_blockers = _materialize_reductions(
        samples,
        reductions=centered_reductions,
        reference=reference,
        record=record,
        bindings=bindings,
        binding_by_identity=binding_by_identity,
        ontology=ontology,
        policy=observation_policy,
        protocol=protocol,
        include_sensitivity_doses=True,
    )
    identity = _record_identity(record)
    status: Literal["complete", "partial"] = "partial" if candidate_omissions else "complete"
    attempt = MaterializationAttemptReceipt(
        contract_id="rt_lnrna_reporter_response_materialization_attempt.v4",
        experiment_id=record.experiment_id,
        reader_record_identity=identity,
        evidence_binding_artifact_id=bindings.artifact_id,
        evidence_binding_artifact_digest=bindings.artifact_digest,
        expected_subject_ids=tuple(sorted(projected_subject_ids)),
        status=status,
        candidate_profile_count=len(candidate),
        candidate_profile_digests=tuple(sorted(profile_digest(row.profile) for row in candidate)),
        candidate_omissions=tuple(
            sorted(candidate_omissions, key=lambda row: (row.subject_id, row.reduction_id, row.code))
        ),
        blockers=(),
    )
    sensitivity_coverage = build_sensitivity_coverage(
        attempt=attempt,
        bindings=bindings,
        expected_subjects=tuple(
            SensitivitySubjectCoordinate(
                design_id,
                assay_subject_id,
                binding_by_identity[(design_id, assay_subject_id)].subject_id,
            )
            for design_id, assay_subject_id in sorted(
                observed_identities,
                key=lambda value: (value[0], value[1] or ""),
            )
        ),
        evidence=endpoints + centered,
        omissions=endpoint_blockers + centered_blockers,
    )
    return MaterializationReadiness(
        status=status,
        attempt=attempt,
        candidate_evidence=candidate,
        endpoint_evidence=endpoints,
        centered_window_evidence=centered,
        sensitivity_coverage=sensitivity_coverage,
    )


def _preflight(record, bindings, ontology, policy, protocol) -> str | None:
    if not isinstance(record, ReaderDataframeRecordRef) or not record.is_source_closed:
        return "reader_record_not_source_closed"
    if not isinstance(bindings, ReaderEvidenceBindingSet) or not bindings.is_source_closed:
        return "reader_evidence_bindings_not_source_closed"
    if not isinstance(ontology, ReporterResponseConditionOntology):
        return "condition_ontology_invalid"
    if not isinstance(policy, ReporterResponseObservationPolicy):
        return "observation_policy_invalid"
    if not isinstance(protocol, MetastudyProtocol):
        return "metastudy_protocol_invalid"
    if ontology.digest != protocol.condition_ontology_digest:
        return "condition_ontology_digest_mismatch"
    if policy.digest != protocol.observation_policy_digest:
        return "observation_policy_digest_mismatch"
    if record.experiment_id not in protocol.planned_kinetic_experiment_ids:
        return "experiment_not_in_planned_kinetic_cohort"
    if record.record_id != "sample_measurements/df" or record.contract_id != "plate_reader.annotated.v1":
        return "sample_measurements_record_contract_mismatch"
    if record.replicate_kind not in {"unknown", "biological"}:
        return "replicate_kind_not_supported"
    identity = (
        record.experiment_id,
        record.protocol_id,
        record.record_id,
        record.record_kind,
        record.record_schema_version,
        record.revision,
        record.revision_digest,
        record.contract_id,
        record.content_digest,
        record.reader_path,
        record.replicate_kind,
        record.replicate_identity_field,
        "position",
    )
    for row in bindings.rows:
        observed = (
            row.reader_experiment_id,
            row.reader_protocol_id,
            row.reader_record_id,
            row.reader_record_kind,
            row.reader_record_schema_version,
            row.reader_record_revision,
            row.reader_record_revision_digest,
            row.reader_record_contract_id,
            row.reader_record_content_digest,
            row.reader_record_path,
            row.reader_replicate_kind,
            row.reader_replicate_identity_field,
            row.observation_identity_field,
        )
        if observed != identity:
            return "reader_record_binding_identity_mismatch"
    return None


def _record_identity(record: ReaderDataframeRecordRef) -> ReaderRecordIdentity:
    return ReaderRecordIdentity(
        reader_experiment_id=record.experiment_id,
        reader_protocol_id=record.protocol_id,
        reader_record_id=record.record_id,
        reader_record_kind=record.record_kind,
        reader_record_schema_version=record.record_schema_version,
        reader_record_revision=record.revision,
        reader_record_revision_digest=record.revision_digest,
        reader_record_contract_id=record.contract_id,
        reader_record_content_digest=record.content_digest,
        reader_record_path=record.reader_path,
    )


def _blocked(
    record: ReaderDataframeRecordRef,
    blocker: str | MaterializationBlocker | None = None,
    *,
    omissions: Iterable[MaterializationOmission] = (),
    bindings: ReaderEvidenceBindingSet | None = None,
    expected_subject_ids: tuple[str, ...] = (),
) -> MaterializationReadiness:
    if isinstance(blocker, str):
        blockers = (MaterializationBlocker(blocker),)
    elif isinstance(blocker, MaterializationBlocker):
        blockers = (blocker,)
    elif blocker is None:
        blockers = ()
    else:
        raise ValueError("blocked materialization requires one typed fatal blocker")
    typed_omissions = tuple(sorted(omissions, key=lambda row: (row.subject_id, row.reduction_id, row.code)))
    if not (blockers or typed_omissions):
        raise ValueError("blocked materialization requires a blocker or coordinate omissions")
    attempt = MaterializationAttemptReceipt(
        contract_id="rt_lnrna_reporter_response_materialization_attempt.v4",
        experiment_id=record.experiment_id,
        reader_record_identity=_record_identity(record),
        evidence_binding_artifact_id=bindings.artifact_id if bindings is not None else None,
        evidence_binding_artifact_digest=bindings.artifact_digest if bindings is not None else None,
        expected_subject_ids=expected_subject_ids,
        status="blocked",
        candidate_profile_count=0,
        candidate_profile_digests=(),
        candidate_omissions=typed_omissions,
        blockers=blockers,
    )
    return MaterializationReadiness(status="blocked", attempt=attempt)


__all__ = ["materialize_record_evidence"]
