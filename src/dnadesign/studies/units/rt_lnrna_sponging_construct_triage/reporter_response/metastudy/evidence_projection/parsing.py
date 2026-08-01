"""Strict parsing of offline profile content projections."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict

from ..._contract_values import json_value
from ...canonical import comparability_key, derive_profile_rows
from ...measurement_profile import (
    MEASUREMENT_PROFILE_CONTRACT_ID,
    ReferenceNormalizationUnavailable,
    validate_measurement_profile_contract,
)
from ...policy import ReporterResponseObservationPolicy
from ...profile.measurement import ConditionMeasurement, EndpointReduction, TimeWindowReduction
from ...profile.response import ControlAssignment, DoseResponse, PairingPolicy
from ...profile.uncertainty import (
    DoseUncertainty,
    EstimatedMetricUncertainty,
    NotEstimableMetricUncertainty,
    ProfileEligibility,
    UncertaintyPolicy,
)
from ...temporal import TemporalPolicyProjection
from ._values import mapping, number_tuple, object_list, required_text, strict_dataclass, strict_object
from .audit_parsing import parse_profile_audit
from .contracts import (
    ProfileContentProjection,
    ProfileEvidenceProjection,
    ProfileProvenanceProjection,
)


def parse_profile_evidence_projection(value: object, *, index: int) -> ProfileEvidenceProjection:
    """Parse bundled evidence without minting live-source closure tokens."""

    row = strict_object(value, label=f"publication evidence profiles[{index}]", fields={"profile", "audit"})
    raw_profile = mapping(row["profile"], label=f"publication evidence profiles[{index}].profile")
    if raw_profile.get("contract_id") == MEASUREMENT_PROFILE_CONTRACT_ID:
        return _parse_measurement_profile_projection(row, raw_profile, index=index)
    profile_payload = strict_object(
        raw_profile,
        label=f"publication evidence profiles[{index}].profile",
        fields={
            "contract_id",
            "study_id",
            "profile_id",
            "subject_id",
            "provenance",
            "observation_policy",
            "reduction",
            "dose_grid_uM",
            "measurements",
            "pairing_policy",
            "dose_responses",
            "dose_uncertainties",
            "comparability_key",
            "eligibility",
        },
    )
    if profile_payload["contract_id"] != "rt_lnrna_reporter_response_profile.v3":
        raise ValueError(f"publication evidence profiles[{index}] profile contract_id changed")
    if profile_payload["study_id"] != "rt_lnrna_sponging_construct_triage":
        raise ValueError(f"publication evidence profiles[{index}] profile study_id changed")
    provenance = ProfileProvenanceProjection(
        **strict_dataclass(profile_payload["provenance"], ProfileProvenanceProjection)
    )
    policy = _parse_observation_policy(profile_payload["observation_policy"])
    reduction = _parse_reduction(profile_payload["reduction"])
    measurements = tuple(
        ConditionMeasurement(**strict_dataclass(item, ConditionMeasurement))
        for item in object_list(profile_payload["measurements"], label="measurements")
    )
    pairing_payload = strict_object(
        profile_payload["pairing_policy"],
        label="pairing_policy",
        fields={"kind", "assignments"},
    )
    pairing = PairingPolicy(
        kind=pairing_payload["kind"],
        assignments=tuple(
            _parse_control_assignment(item)
            for item in object_list(pairing_payload["assignments"], label="pairing_policy.assignments")
        ),
    )
    uncertainties = tuple(
        _parse_dose_uncertainty(item)
        for item in object_list(profile_payload["dose_uncertainties"], label="dose_uncertainties")
    )
    eligibility_values = strict_dataclass(profile_payload["eligibility"], ProfileEligibility)
    eligibility_values["reasons"] = tuple(eligibility_values["reasons"])
    eligibility = ProfileEligibility(**eligibility_values)
    dose_grid, canonical_measurements, responses, canonical_uncertainties = derive_profile_rows(
        reduction=reduction,
        dose_grid_uM=number_tuple(profile_payload["dose_grid_uM"], label="dose_grid_uM"),
        measurements=measurements,
        pairing_policy=pairing,
        observation_policy=policy,
        dose_uncertainties=uncertainties,
    )
    declared_responses = tuple(
        _parse_dose_response(item) for item in object_list(profile_payload["dose_responses"], label="dose_responses")
    )
    if declared_responses != responses:
        raise ValueError("serialized dose_responses differ from canonical profile derivation")
    expected_key = comparability_key(
        observation_policy_digest=policy.digest,
        reduction=reduction,
        dose_grid_uM=dose_grid,
        dose_uncertainties=canonical_uncertainties,
    )
    if profile_payload["comparability_key"] != expected_key:
        raise ValueError("serialized comparability_key differs from canonical profile derivation")
    projection = ProfileContentProjection(
        profile_id=required_text(profile_payload["profile_id"], label="profile_id"),
        subject_id=required_text(profile_payload["subject_id"], label="subject_id"),
        provenance=provenance,
        observation_policy=policy,
        reduction=reduction,
        dose_grid_uM=dose_grid,
        measurements=canonical_measurements,
        pairing_policy=pairing,
        dose_uncertainties=canonical_uncertainties,
        dose_responses=responses,
        reference_normalization=None,
        comparability_key=expected_key,
        serialized_payload=dict(profile_payload),
    )
    del eligibility
    audit = parse_profile_audit(row["audit"], profile_payload=profile_payload, profile=projection)
    return ProfileEvidenceProjection(profile=projection, audit=audit)


def _parse_measurement_profile_projection(
    row: Mapping[str, object],
    profile_payload: Mapping[str, object],
    *,
    index: int,
) -> ProfileEvidenceProjection:
    profile_payload = strict_object(
        profile_payload,
        label=f"publication evidence profiles[{index}].profile",
        fields={
            "contract_id",
            "study_id",
            "profile_id",
            "subject_id",
            "provenance",
            "observation_policy",
            "reduction",
            "dose_grid_uM",
            "measurements",
            "reference_normalization",
            "comparability_key",
            "eligibility",
        },
    )
    provenance = ProfileProvenanceProjection(
        **strict_dataclass(profile_payload["provenance"], ProfileProvenanceProjection)
    )
    policy = _parse_observation_policy(profile_payload["observation_policy"])
    reduction = _parse_reduction(profile_payload["reduction"])
    measurements = tuple(
        ConditionMeasurement(**strict_dataclass(item, ConditionMeasurement))
        for item in object_list(profile_payload["measurements"], label="measurements")
    )
    reference = ReferenceNormalizationUnavailable(
        **strict_dataclass(profile_payload["reference_normalization"], ReferenceNormalizationUnavailable)
    )
    eligibility_values = strict_dataclass(profile_payload["eligibility"], ProfileEligibility)
    eligibility_values["reasons"] = tuple(eligibility_values["reasons"])
    eligibility = ProfileEligibility(**eligibility_values)
    dose_grid = number_tuple(profile_payload["dose_grid_uM"], label="dose_grid_uM")
    expected_key = validate_measurement_profile_contract(
        contract_id=profile_payload["contract_id"],
        study_id=profile_payload["study_id"],
        profile_id=profile_payload["profile_id"],
        subject_id=profile_payload["subject_id"],
        observation_policy=policy,
        reduction=reduction,
        dose_grid_uM=dose_grid,
        measurements=measurements,
        reference_normalization=reference,
        eligibility=eligibility,
    )
    if profile_payload["comparability_key"] != expected_key:
        raise ValueError("serialized comparability_key differs from canonical measurement-profile derivation")
    projection = ProfileContentProjection(
        profile_id=required_text(profile_payload["profile_id"], label="profile_id"),
        subject_id=required_text(profile_payload["subject_id"], label="subject_id"),
        provenance=provenance,
        observation_policy=policy,
        reduction=reduction,
        dose_grid_uM=dose_grid,
        measurements=measurements,
        pairing_policy=None,
        dose_uncertainties=(),
        dose_responses=(),
        reference_normalization=reference,
        comparability_key=expected_key,
        serialized_payload=dict(profile_payload),
    )
    audit = parse_profile_audit(row["audit"], profile_payload=profile_payload, profile=projection)
    return ProfileEvidenceProjection(profile=projection, audit=audit)


def _parse_observation_policy(value: object) -> ReporterResponseObservationPolicy:
    payload = strict_object(
        value,
        label="observation_policy",
        fields={
            "policy_id",
            "pairing_kind",
            "within_acquisition_reduction_statistic",
            "biological_replicate_uncertainty_policy",
            "contract_id",
            "normalized_reporter_formula",
            "relative_od_formula",
            "clipping_policy",
            "digest",
        },
    )
    uncertainty = UncertaintyPolicy(
        **strict_dataclass(payload["biological_replicate_uncertainty_policy"], UncertaintyPolicy)
    )
    policy = ReporterResponseObservationPolicy(
        policy_id=payload["policy_id"],
        pairing_kind=payload["pairing_kind"],
        within_acquisition_reduction_statistic=payload["within_acquisition_reduction_statistic"],
        biological_replicate_uncertainty_policy=uncertainty,
    )
    if payload != json_value(asdict(policy)):
        raise ValueError("serialized observation policy differs from its canonical form")
    return policy


def _parse_reduction(value: object) -> EndpointReduction | TimeWindowReduction:
    payload = mapping(value, label="reduction")
    kind = payload.get("kind")
    if kind == "endpoint":
        values = strict_dataclass(payload, EndpointReduction)
        values["temporal_policy"] = _parse_temporal_policy(values["temporal_policy"])
        return EndpointReduction(**values)
    if kind == "time_window":
        values = strict_dataclass(payload, TimeWindowReduction)
        values["temporal_policy"] = _parse_temporal_policy(values["temporal_policy"])
        return TimeWindowReduction(**values)
    raise ValueError("reduction.kind must be endpoint or time_window")


def _parse_temporal_policy(value: object) -> TemporalPolicyProjection:
    payload = strict_object(
        value,
        label="temporal_policy",
        fields={"selection", "method", "output_space", "support", "digest"},
    )
    projection = TemporalPolicyProjection.from_reader_mapping(
        {key: item for key, item in payload.items() if key != "digest"}
    )
    if payload != json_value(asdict(projection)):
        raise ValueError("serialized temporal policy differs from its canonical form")
    return projection


def _parse_control_assignment(value: object) -> ControlAssignment:
    values = strict_dataclass(value, ControlAssignment)
    values["baseline_observation_ids"] = tuple(values["baseline_observation_ids"])
    values["positive_control_observation_ids"] = tuple(values["positive_control_observation_ids"])
    return ControlAssignment(**values)


def _parse_dose_response(value: object) -> DoseResponse:
    values = strict_dataclass(value, DoseResponse)
    values["baseline_observation_ids"] = tuple(values["baseline_observation_ids"])
    values["positive_control_observation_ids"] = tuple(values["positive_control_observation_ids"])
    return DoseResponse(**values)


def _parse_dose_uncertainty(value: object) -> DoseUncertainty:
    payload = strict_object(
        value,
        label="dose_uncertainty",
        fields={
            "dose_uM",
            "biological_replicate_count",
            "normalized_reporter_response",
            "relative_od",
        },
    )
    return DoseUncertainty(
        dose_uM=payload["dose_uM"],
        biological_replicate_count=payload["biological_replicate_count"],
        normalized_reporter_response=_parse_metric_uncertainty(payload["normalized_reporter_response"]),
        relative_od=_parse_metric_uncertainty(payload["relative_od"]),
    )


def _parse_metric_uncertainty(value: object) -> EstimatedMetricUncertainty | NotEstimableMetricUncertainty:
    payload = mapping(value, label="metric_uncertainty")
    if payload.get("status") == "estimated":
        return EstimatedMetricUncertainty(**strict_dataclass(payload, EstimatedMetricUncertainty))
    if payload.get("status") == "not_estimable":
        return NotEstimableMetricUncertainty(**strict_dataclass(payload, NotEstimableMetricUncertainty))
    raise ValueError("metric uncertainty status is invalid")


__all__ = ["parse_profile_evidence_projection"]
