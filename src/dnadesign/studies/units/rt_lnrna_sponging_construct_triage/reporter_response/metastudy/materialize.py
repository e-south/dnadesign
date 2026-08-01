"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/reporter_response/metastudy/materialize.py

Materialize canonical meta-study profiles from source-closed Reader evidence.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import math
import statistics
from collections.abc import Iterable
from dataclasses import dataclass
from io import BytesIO
from typing import Literal

import numpy as np
import pandas as pd

from ...reader_evidence import ReaderDataframeRecordRef, ReaderEvidenceBinding, ReaderEvidenceBindingSet
from .. import (
    ConditionMeasurement,
    ControlAssignment,
    DoseUncertainty,
    EndpointReduction,
    NotEstimableMetricUncertainty,
    PairingPolicy,
    ReporterResponseContractError,
    ReporterResponseObservationPolicy,
    TimeWindowReduction,
    build_reporter_response_profile,
)
from ..profile import Reduction
from ..temporal import TemporalSelectedRow, reduce_temporal_input_trace
from .audits import _build_derivation_closed_profile_audit, profile_digest
from .condition_ontology import ReporterResponseConditionOntology
from .contracts.materialization import (
    MaterializationAttemptReceipt,
    MaterializationBlocker,
    MaterializationOmission,
    ReaderRecordIdentity,
)
from .contracts.profile import GrowthPhaseStratum, ProfileEvidence
from .contracts.protocol import DEFAULT_PROTOCOL, MetastudyProtocol
from .sensitivity_coverage import (
    SensitivityCoverageLedger,
    SensitivitySubjectCoordinate,
    build_sensitivity_coverage,
)

_BASE_COLUMNS = {"type", "position", "time", "channel", "value", "treatment", "design_id"}
_QUALITY_COLUMNS = {"value_policy_clipped", "value_instrument_overflow", "value_bound_kind"}


@dataclass(frozen=True, slots=True)
class MaterializationReadiness:
    """Complete, partial, or blocked materialization with explicit issue scope."""

    status: Literal["complete", "partial", "blocked"]
    attempt: MaterializationAttemptReceipt
    candidate_evidence: tuple[ProfileEvidence, ...] = ()
    endpoint_evidence: tuple[ProfileEvidence, ...] = ()
    centered_window_evidence: tuple[ProfileEvidence, ...] = ()
    sensitivity_coverage: SensitivityCoverageLedger | None = None

    def __post_init__(self) -> None:
        evidence = self.candidate_evidence + self.endpoint_evidence + self.centered_window_evidence
        if self.status != self.attempt.status:
            raise ValueError("materialization status must equal its attempt receipt")
        if self.status == "blocked" and (
            not (self.blockers or self.omissions) or evidence or self.sensitivity_coverage is not None
        ):
            raise ValueError("blocked materialization requires issues and no evidence")
        if self.status in {"complete", "partial"} and (
            self.blockers
            or not self.candidate_evidence
            or self.sensitivity_coverage is None
            or self.attempt.attempt_digest != self.sensitivity_coverage.materialization_attempt_digest
        ):
            raise ValueError("usable materialization requires candidate evidence and exact sensitivity coverage")
        if self.status == "complete" and self.omissions:
            raise ValueError("complete materialization cannot contain omissions")
        if self.status == "partial" and not self.omissions:
            raise ValueError("partial materialization requires coordinate omissions")

    @property
    def blockers(self) -> tuple[str, ...]:
        return tuple(row.code for row in self.attempt.blockers)

    @property
    def omissions(self) -> tuple[MaterializationOmission, ...]:
        return self.attempt.candidate_omissions


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
            sorted(
                candidate_omissions,
                key=lambda row: (row.subject_id, row.reduction_id, row.code),
            )
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


def _materialize_reductions(
    frame: pd.DataFrame,
    *,
    reductions: Iterable[Reduction],
    reference: EndpointReduction,
    record: ReaderDataframeRecordRef,
    bindings: ReaderEvidenceBindingSet,
    binding_by_identity: dict[tuple[str, str | None], ReaderEvidenceBinding],
    ontology: ReporterResponseConditionOntology,
    policy: ReporterResponseObservationPolicy,
    protocol: MetastudyProtocol,
    include_sensitivity_doses: bool,
) -> tuple[tuple[ProfileEvidence, ...], tuple[MaterializationOmission, ...]]:
    evidence: list[ProfileEvidence] = []
    omissions: list[MaterializationOmission] = []
    for reduction in reductions:
        for identity in sorted(_observed_reader_identities(frame), key=lambda value: (value[0], value[1] or "")):
            design_id, assay_subject_id = identity
            binding = binding_by_identity[identity]
            subject_id = binding.subject_id
            design_frame = frame.loc[_reader_identity_mask(frame, identity)]
            built = _build_profile(
                design_frame,
                reduction=reduction,
                reference=reference,
                record=record,
                bindings=bindings,
                design_id=design_id,
                assay_subject_id=assay_subject_id,
                subject_id=subject_id,
                ontology=ontology,
                policy=policy,
                protocol=protocol,
                include_sensitivity_doses=include_sensitivity_doses,
            )
            if isinstance(built, str):
                omissions.append(
                    MaterializationOmission(
                        code=built,
                        subject_id=subject_id,
                        reduction_id=_reduction_id(reduction),
                    )
                )
            else:
                evidence.append(built)
    return tuple(evidence), tuple(omissions)


def _observed_reader_identities(frame: pd.DataFrame) -> set[tuple[str, str | None]]:
    assay_subjects = (
        frame["assay_subject_id"]
        if "assay_subject_id" in frame.columns
        else pd.Series((None,) * len(frame), index=frame.index, dtype=object)
    )
    return {
        (str(design_id), None if pd.isna(assay_subject_id) else str(assay_subject_id))
        for design_id, assay_subject_id in zip(frame["design_id"], assay_subjects, strict=True)
    }


def _has_ambiguous_partial_identity(
    observed_identities: set[tuple[str, str | None]],
    *,
    bindings: ReaderEvidenceBindingSet,
) -> bool:
    binding_design_ids = tuple(row.raw_design_id for row in bindings.rows)
    return any(
        assay_subject_id is None and binding_design_ids.count(design_id) > 1
        for design_id, assay_subject_id in observed_identities
    )


def _reader_identity_mask(frame: pd.DataFrame, identity: tuple[str, str | None]) -> pd.Series:
    design_id, assay_subject_id = identity
    mask = frame["design_id"].astype(str).eq(design_id)
    if "assay_subject_id" not in frame.columns:
        return mask if assay_subject_id is None else mask & False
    if assay_subject_id is None:
        return mask & frame["assay_subject_id"].isna()
    return mask & frame["assay_subject_id"].astype(str).eq(assay_subject_id)


def _build_profile(
    frame,
    *,
    reduction,
    reference,
    record,
    bindings,
    design_id,
    assay_subject_id,
    subject_id,
    ontology,
    policy,
    protocol,
    include_sensitivity_doses,
):
    growth_phase_strata = _growth_phase_strata(
        frame,
        reduction=reduction,
        ontology=ontology,
        protocol=protocol,
    )
    if isinstance(growth_phase_strata, str):
        return growth_phase_strata
    selected = _select_reduction(frame, reduction)
    reference_rows = _select_reduction(frame, reference)
    primary = ontology.condition_for_dose(protocol.primary_dose_uM)
    baseline = next(row for row in ontology.conditions if row.role == "baseline")
    positive = next(row for row in ontology.conditions if row.role == "positive_control")
    observed_labels = set(selected["treatment"].astype(str))
    dose_definitions = [primary]
    if include_sensitivity_doses:
        definitions_by_dose = {
            float(definition.dose_uM): definition for definition in ontology.conditions if definition.role == "dose"
        }
        dose_definitions.extend(
            definition
            for dose in protocol.sensitivity_doses_uM
            if (definition := definitions_by_dose.get(dose)) is not None
            and definition.treatment_label in observed_labels
        )
    dose_definitions.sort(key=lambda row: float(row.dose_uM))
    included = {baseline.treatment_label, positive.treatment_label}
    included.update(definition.treatment_label for definition in dose_definitions)
    selected = selected.loc[selected["treatment"].isin(included)]
    reference_rows = reference_rows.loc[reference_rows["treatment"].isin(included)]
    if selected.empty or reference_rows.empty:
        return "reduction_observations_missing"
    if (
        reduction.temporal_policy is not None
        and reduction.temporal_policy.support.censored_values == "reject"
        and (_contains_censored_values(selected) or _contains_censored_values(reference_rows))
    ):
        return "censored_observations_rejected"
    measurements: list[ConditionMeasurement] = []
    assignments: list[ControlAssignment] = []
    dose_values: dict[float, list[tuple[float, float]]] = {
        float(definition.dose_uM): [] for definition in dose_definitions
    }
    within_acquisition_ranges: list[float] = []
    reference_spans: list[float] = []
    replicate_field = record.replicate_identity_field
    if record.replicate_kind == "unknown" and replicate_field is not None:
        return "unknown_replicate_identity_field_declared"
    if replicate_field is not None:
        if replicate_field not in selected.columns or replicate_field not in reference_rows.columns:
            return "declared_biological_replicate_identity_missing"
        if selected[replicate_field].isna().any() or reference_rows[replicate_field].isna().any():
            return "declared_biological_replicate_identity_incomplete"

    measurements_by_condition: dict[str, list[ConditionMeasurement]] = {}
    for definition in (baseline, positive, *dose_definitions):
        label = definition.treatment_label
        selected_condition = selected.loc[selected["treatment"].eq(label)]
        reference_condition_rows = reference_rows.loc[reference_rows["treatment"].eq(label)]
        if replicate_field is None:
            biological_replicate_ids: tuple[str | None, ...] = (None,)
        else:
            selected_ids = {str(value) for value in selected_condition[replicate_field]}
            reference_ids = {str(value) for value in reference_condition_rows[replicate_field]}
            if selected_ids != reference_ids:
                return "declared_biological_replicate_identity_inconsistent"
            biological_replicate_ids = tuple(sorted(selected_ids))
        condition_measurements = measurements_by_condition.setdefault(definition.condition_id, [])
        for biological_replicate_id in biological_replicate_ids:
            condition_rows = selected_condition
            reference_condition = reference_condition_rows
            if replicate_field is not None:
                condition_rows = selected_condition.loc[
                    selected_condition[replicate_field].astype(str).eq(biological_replicate_id)
                ]
                reference_condition = reference_condition_rows.loc[
                    reference_condition_rows[replicate_field].astype(str).eq(biological_replicate_id)
                ]
            summary = _condition_summary(condition_rows, ontology, reduction=reduction, protocol=protocol)
            reference_summary = _condition_summary(
                reference_condition,
                ontology,
                reduction=reference,
                protocol=protocol,
            )
            if summary is None or reference_summary is None:
                return "condition_or_channel_observations_incomplete"
            rfp, od, ratio, observation_count, observation_range = summary
            within_acquisition_ranges.append(observation_range)
            reference_spans.append(reference_summary[4])
            replicate_token = biological_replicate_id or "unknown-replicate"
            observation_id = (
                f"{record.experiment_id}:{subject_id}:{definition.condition_id}:"
                f"{replicate_token}:{_reduction_id(reduction)}"
            )
            measurement = ConditionMeasurement(
                observation_id=observation_id,
                condition_id=definition.condition_id,
                source_condition_value=label,
                role=definition.role,
                dose_uM=definition.dose_uM,
                biological_replicate_id=biological_replicate_id,
                acquisition_id=record.experiment_id,
                within_acquisition_observation_count=observation_count,
                within_acquisition_reduction_statistic=policy.within_acquisition_reduction_statistic,
                rfp=rfp,
                od600=od,
                rfp_over_od600=ratio,
            )
            measurements.append(measurement)
            condition_measurements.append(measurement)

    baselines = measurements_by_condition[baseline.condition_id]
    positives = measurements_by_condition[positive.condition_id]
    if replicate_field is not None and policy.pairing_kind == "paired_by_design":
        return "explicit_paired_control_assignment_missing"
    baseline_ratio = statistics.median(row.rfp_over_od600 for row in baselines)
    positive_ratio = statistics.median(row.rfp_over_od600 for row in positives)
    baseline_od = statistics.median(row.od600 for row in baselines)
    separation = positive_ratio - baseline_ratio
    if separation <= 0.0 or baseline_od <= 0.0:
        return "positive_control_separation_failed"
    for definition in dose_definitions:
        dose = float(definition.dose_uM)
        for dose_measurement in measurements_by_condition[definition.condition_id]:
            dose_values[dose].append(
                (
                    (dose_measurement.rfp_over_od600 - baseline_ratio) / separation,
                    dose_measurement.od600 / baseline_od,
                )
            )
            assignments.append(
                ControlAssignment(
                    dose_observation_id=dose_measurement.observation_id,
                    baseline_observation_ids=tuple(row.observation_id for row in baselines),
                    positive_control_observation_ids=tuple(row.observation_id for row in positives),
                )
            )
    statistic = policy.biological_replicate_uncertainty_policy.biological_replicate_reduction_statistic
    minimum = policy.biological_replicate_uncertainty_policy
    uncertainties = tuple(
        _descriptive_uncertainty(
            dose=dose,
            values=values,
            statistic=statistic,
            minimum_replicates=minimum.minimum_biological_replicates,
            identity_complete=replicate_field is not None,
        )
        for dose, values in sorted(dose_values.items())
    )
    profile = build_reporter_response_profile(
        profile_id=f"{record.experiment_id}:{subject_id}:{_reduction_id(reduction)}",
        subject_id=subject_id,
        raw_design_id=design_id,
        raw_assay_subject_id=assay_subject_id,
        evidence_bindings=bindings,
        observation_policy=policy,
        reduction=reduction,
        dose_grid_uM=tuple(sorted(dose_values)),
        measurements=measurements,
        pairing_policy=PairingPolicy(kind=policy.pairing_kind, assignments=tuple(assignments)),
        dose_uncertainties=uncertainties,
        ineligibility_reasons=("preference_objective_not_defined",),
    )
    required = len(selected)
    clipped = int(
        (selected["value_policy_clipped"].astype(bool) | selected["value_bound_kind"].astype(str).ne("exact")).sum()
    )
    overflow = int(selected["value_instrument_overflow"].astype(bool).sum())
    audit = _build_derivation_closed_profile_audit(
        profile,
        condition_ontology_digest=ontology.digest,
        within_acquisition_observation_range=statistics.median(within_acquisition_ranges),
        reference_within_acquisition_observation_range=statistics.median(reference_spans),
        required_observation_count=required,
        overflow_observation_count=overflow,
        clipped_observation_count=clipped,
        growth_phase_strata=growth_phase_strata,
    )
    return ProfileEvidence(profile=profile, audit=audit)


def _growth_phase_strata(
    frame: pd.DataFrame,
    *,
    reduction: Reduction,
    ontology: ReporterResponseConditionOntology,
    protocol: MetastudyProtocol,
) -> tuple[GrowthPhaseStratum, ...] | str:
    """Derive study-owned growth-phase position from observed normalizer traces."""

    if isinstance(reduction, EndpointReduction):
        return ()
    normalizer = frame.loc[frame["channel"].eq(ontology.normalizer_channel)].copy()
    if normalizer.empty:
        return "phase_not_estimable_normalizer_missing"
    normalizer["time"] = pd.to_numeric(normalizer["time"], errors="coerce")
    normalizer["value"] = pd.to_numeric(normalizer["value"], errors="coerce")
    if normalizer["time"].isna().any() or not np.isfinite(normalizer["time"].to_numpy(dtype=float)).all():
        return "phase_not_estimable_nonfinite_od"
    definitions = ontology.by_treatment_label
    rows: list[GrowthPhaseStratum] = []
    for treatment, treatment_rows in normalizer.groupby("treatment", sort=True, dropna=False):
        label = str(treatment)
        definition = definitions.get(label)
        if definition is None:
            return "phase_not_estimable_condition_not_declared"
        trace = (
            treatment_rows.groupby("time", as_index=False, sort=True)["value"]
            .median()
            .sort_values("time", kind="stable")
        )
        times = trace["time"].to_numpy(dtype=float)
        values = trace["value"].to_numpy(dtype=float)
        # The scale spans the trace, so rejected quality observations must not
        # influence it. Keep the raw trace for candidate-boundary checks below.
        scale_rows = treatment_rows.loc[
            ~treatment_rows["value_policy_clipped"].astype(bool)
            & ~treatment_rows["value_instrument_overflow"].astype(bool)
            & treatment_rows["value_bound_kind"].astype(str).eq("exact")
            & np.isfinite(treatment_rows["value"].to_numpy(dtype=float))
            & treatment_rows["value"].gt(0.0)
        ]
        scale_trace = (
            scale_rows.groupby("time", as_index=False, sort=True)["value"].median().sort_values("time", kind="stable")
        )
        if scale_trace.empty:
            return "phase_not_estimable_positive_slope_scale"
        scale_times = scale_trace["time"].to_numpy(dtype=float)
        scale_values = scale_trace["value"].to_numpy(dtype=float)
        first_start = math.ceil(float(scale_times.min()) - 1e-9)
        last_start = math.floor(float(scale_times.max()) - protocol.growth_phase_slope_window_h + 1e-9)
        slopes = tuple(
            value
            for start in range(first_start, last_start + 1)
            if (
                value := _log_normalizer_slope(
                    scale_times,
                    scale_values,
                    start_h=float(start),
                    protocol=protocol,
                )
            )
            is not None
            and value > 0.0
        )
        if not slopes:
            return "phase_not_estimable_positive_slope_scale"
        scale = float(
            np.quantile(
                np.asarray(slopes, dtype=float),
                protocol.growth_phase_scale_quantile,
                method="linear",
            )
        )
        start_slope = _log_normalizer_slope(
            times,
            values,
            start_h=reduction.recorded_start_time_h,
            protocol=protocol,
        )
        end_slope = _log_normalizer_slope(
            times,
            values,
            start_h=reduction.recorded_end_time_h - protocol.growth_phase_slope_window_h,
            protocol=protocol,
        )
        if start_slope is None or end_slope is None or scale <= 0.0:
            return "phase_not_estimable_temporal_support"
        rows.append(
            GrowthPhaseStratum(
                condition_id=definition.condition_id,
                normalized_start_slope=start_slope / scale,
                normalized_end_slope=end_slope / scale,
            )
        )
    return tuple(sorted(rows, key=lambda row: row.condition_id))


def _log_normalizer_slope(
    times: np.ndarray,
    values: np.ndarray,
    *,
    start_h: float,
    protocol: MetastudyProtocol,
) -> float | None:
    end_h = start_h + protocol.growth_phase_slope_window_h
    mask = (times >= start_h - 1e-9) & (times <= end_h + 1e-9)
    if int(mask.sum()) < protocol.growth_phase_minimum_slope_points:
        return None
    selected_times = times[mask]
    if len(set(selected_times.tolist())) != len(selected_times):
        return None
    selected_values = values[mask]
    if not np.isfinite(selected_values).all() or np.any(selected_values <= 0.0):
        return None
    slope = float(np.polyfit(selected_times, np.log(selected_values), 1)[0])
    return slope if math.isfinite(slope) else None


def _select_reduction(frame: pd.DataFrame, reduction: Reduction) -> pd.DataFrame:
    time = pd.to_numeric(frame["time"], errors="coerce")
    if time.isna().any():
        return frame.iloc[0:0]
    if isinstance(reduction, EndpointReduction):
        return frame.loc[(time - reduction.recorded_time_h).abs().le(1e-9)].copy()
    return frame.loc[
        time.ge(reduction.recorded_start_time_h - 1e-9) & time.le(reduction.recorded_end_time_h + 1e-9)
    ].copy()


def _condition_summary(
    frame: pd.DataFrame,
    ontology: ReporterResponseConditionOntology,
    *,
    reduction: Reduction,
    protocol: MetastudyProtocol,
):
    channels = (ontology.reporter_channel, ontology.normalizer_channel, ontology.ratio_channel)
    if frame.empty or set(frame["channel"].astype(str)) != set(channels):
        return None
    by_observation: dict[str, tuple[float, float, float]] = {}
    temporal_policy = reduction.temporal_policy
    if temporal_policy is None:
        return None
    for observation_identity in sorted(set(frame["position"].astype(str))):
        observation = frame.loc[frame["position"].astype(str).eq(observation_identity)]
        time_sets = {
            channel: tuple(sorted(pd.to_numeric(observation.loc[observation["channel"].eq(channel), "time"]).tolist()))
            for channel in channels
        }
        if not time_sets[channels[0]] or len(set(time_sets.values())) != 1:
            return None
        observed_times = time_sets[channels[0]]
        if not all(math.isfinite(value) for value in observed_times):
            return None
        if len(observed_times) != len(set(observed_times)):
            return None
        values: list[float] = []
        for channel in channels:
            channel_rows = observation.loc[observation["channel"].eq(channel)]
            trace = tuple(
                TemporalSelectedRow(
                    observation_identity=observation_identity,
                    time_h=float(row.time),
                    value=float(row.value),
                    value_policy_clipped=bool(getattr(row, "value_policy_clipped", False)),
                    value_instrument_overflow=bool(getattr(row, "value_instrument_overflow", False)),
                    value_bound_kind=str(getattr(row, "value_bound_kind", "exact")),  # type: ignore[arg-type]
                )
                for row in channel_rows.itertuples(index=False)
            )
            try:
                values.append(
                    reduce_temporal_input_trace(
                        trace,
                        policy=temporal_policy,
                        within_acquisition_statistic="median",
                    )
                )
            except ReporterResponseContractError:
                return None
        if not all(math.isfinite(value) for value in values) or values[1] <= 0.0:
            return None
        by_observation[observation_identity] = (values[0], values[1], values[2])
    if len(by_observation) < protocol.minimum_within_acquisition_observations_per_stratum:
        return None
    reporter = _reduce((row[0] for row in by_observation.values()), "median")
    normalizer = _reduce((row[1] for row in by_observation.values()), "median")
    ratio_values = tuple(row[2] for row in by_observation.values())
    ratio = (
        reporter / normalizer
        if isinstance(reduction, EndpointReduction)
        else _reduce(ratio_values, protocol.time_summary_statistic)
    )
    return reporter, normalizer, ratio, len(by_observation), max(ratio_values) - min(ratio_values)


def _contains_censored_values(frame: pd.DataFrame) -> bool:
    return bool(
        frame["value_policy_clipped"].astype(bool).any()
        or frame["value_instrument_overflow"].astype(bool).any()
        or frame["value_bound_kind"].astype(str).ne("exact").any()
    )


def _reduce(values: Iterable[float], statistic: str) -> float:
    rows = tuple(float(value) for value in values)
    return float(statistics.median(rows) if statistic == "median" else statistics.mean(rows))


def _descriptive_uncertainty(
    *,
    dose: float,
    values: list[tuple[float, float]],
    statistic: str,
    minimum_replicates: int,
    identity_complete: bool,
) -> DoseUncertainty:
    count = len(values)
    if count == 0:
        raise ValueError("dose uncertainty requires acquisition observations")
    if not identity_complete:
        reason = "biological_replicate_identity_unknown"
        biological_replicate_count = 0
    elif count < minimum_replicates:
        reason = "below_minimum_biological_replicates"
        biological_replicate_count = count
    else:
        # Materialization remains descriptive until a biological-replicate
        # resampling method is explicitly selected and validated.
        reason = "insufficient_valid_resamples"
        biological_replicate_count = count
    return DoseUncertainty(
        dose_uM=dose,
        biological_replicate_count=biological_replicate_count,
        normalized_reporter_response=NotEstimableMetricUncertainty(
            estimate=_reduce((row[0] for row in values), statistic),
            reason=reason,
        ),
        relative_od=NotEstimableMetricUncertainty(
            estimate=_reduce((row[1] for row in values), statistic),
            reason=reason,
        ),
    )


def _reduction_id(reduction: Reduction) -> str:
    if isinstance(reduction, EndpointReduction):
        return f"endpoint-{reduction.recorded_time_h:g}h"
    return f"window-{reduction.recorded_start_time_h:g}-{reduction.recorded_end_time_h:g}h"


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
    typed_omissions = tuple(
        sorted(
            omissions,
            key=lambda row: (row.subject_id, row.reduction_id, row.code),
        )
    )
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


__all__ = ["MaterializationReadiness", "materialize_record_evidence"]
