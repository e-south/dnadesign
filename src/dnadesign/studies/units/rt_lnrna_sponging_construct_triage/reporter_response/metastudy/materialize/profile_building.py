"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/reporter_response/metastudy/materialize/profile_building.py

Construction and audit binding for one descriptive reporter profile.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import statistics

import pandas as pd

from ....reader_evidence import ReaderDataframeRecordRef, ReaderEvidenceBindingSet
from ...building import build_reporter_response_profile
from ...measurement_profile import DescriptiveReporterProfile, build_reporter_measurement_profile
from ...policy import ReporterResponseObservationPolicy
from ...profile.measurement import ConditionMeasurement, EndpointReduction, Reduction
from ...profile.response import ControlAssignment, PairingPolicy
from ..audits import _build_derivation_closed_profile_audit
from ..condition_ontology import ReporterResponseConditionOntology
from ..contracts.profile import GrowthPhaseStratum, ProfileEvidence
from ..contracts.protocol import MetastudyProtocol
from .reference import resolve_reference_basis
from .temporal import _condition_summary, _contains_censored_values, _growth_phase_strata, _select_reduction
from .uncertainty import _descriptive_uncertainty


def _build_profile(
    frame: pd.DataFrame,
    *,
    reduction: Reduction,
    reference: EndpointReduction,
    record: ReaderDataframeRecordRef,
    bindings: ReaderEvidenceBindingSet,
    design_id: str,
    assay_subject_id: str | None,
    subject_id: str,
    ontology: ReporterResponseConditionOntology,
    policy: ReporterResponseObservationPolicy,
    protocol: MetastudyProtocol,
    include_sensitivity_doses: bool,
) -> ProfileEvidence | str:
    """Build one source-closed profile or return a typed omission code."""

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
    positive = ontology.positive_control
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
    included = {baseline.treatment_label}
    if positive is not None:
        included.add(positive.treatment_label)
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
    condition_definitions = (baseline, *((positive,) if positive is not None else ()), *dose_definitions)
    for definition in condition_definitions:
        label = definition.treatment_label
        selected_condition = selected.loc[selected["treatment"].eq(label)]
        reference_condition_rows = reference_rows.loc[reference_rows["treatment"].eq(label)]
        if definition.role == "positive_control" and (selected_condition.empty or reference_condition_rows.empty):
            continue
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
    positives = measurements_by_condition.get(positive.condition_id, []) if positive is not None else []
    if replicate_field is not None and policy.pairing_kind == "paired_by_design":
        return "explicit_paired_control_assignment_missing"
    reference_basis = resolve_reference_basis(baselines=baselines, positive=positive, positives=positives)
    if reference_basis.unavailable is not None:
        profile = build_reporter_measurement_profile(
            profile_id=f"{record.experiment_id}:{subject_id}:{_reduction_id(reduction)}",
            subject_id=subject_id,
            raw_design_id=design_id,
            raw_assay_subject_id=assay_subject_id,
            evidence_bindings=bindings,
            observation_policy=policy,
            reduction=reduction,
            dose_grid_uM=tuple(float(row.dose_uM) for row in dose_definitions),
            measurements=measurements,
            reference_normalization=reference_basis.unavailable,
            ineligibility_reasons=(
                "preference_objective_not_defined",
                f"reference_normalization_{reference_basis.unavailable.reason}",
            ),
        )
        return _profile_evidence(
            profile,
            selected=selected,
            ontology=ontology,
            within_acquisition_ranges=within_acquisition_ranges,
            reference_spans=reference_spans,
            growth_phase_strata=growth_phase_strata,
        )
    assert reference_basis.separation is not None
    separation = reference_basis.separation
    baseline_ratio = reference_basis.baseline_ratio
    baseline_od = reference_basis.baseline_od600
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
    return _profile_evidence(
        profile,
        selected=selected,
        ontology=ontology,
        within_acquisition_ranges=within_acquisition_ranges,
        reference_spans=reference_spans,
        growth_phase_strata=growth_phase_strata,
    )


def _profile_evidence(
    profile: DescriptiveReporterProfile,
    *,
    selected: pd.DataFrame,
    ontology: ReporterResponseConditionOntology,
    within_acquisition_ranges: list[float],
    reference_spans: list[float],
    growth_phase_strata: tuple[GrowthPhaseStratum, ...],
) -> ProfileEvidence:
    """Bind one profile to its derivation-closed quality audit."""

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


def _reduction_id(reduction: Reduction) -> str:
    """Return the stable profile identifier segment for one reduction."""

    if isinstance(reduction, EndpointReduction):
        return f"endpoint-{reduction.recorded_time_h:g}h"
    return f"window-{reduction.recorded_start_time_h:g}-{reduction.recorded_end_time_h:g}h"
