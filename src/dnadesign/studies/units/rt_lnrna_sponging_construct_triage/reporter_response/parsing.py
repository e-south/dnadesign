"""Strict parsing and source-closed revalidation of serialized reporter profiles."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, fields

from ..reader_evidence import ReaderEvidenceBindingSet
from ._contract_values import ReporterResponseContractError
from ._contract_values import json_value as _json_value
from .measurement_profile import (
    MEASUREMENT_PROFILE_CONTRACT_ID,
    ReferenceNormalizationUnavailable,
    ReporterMeasurementProfile,
)
from .policy import ReporterResponseObservationPolicy
from .profile.measurement import ConditionMeasurement, EndpointReduction, Reduction, TimeWindowReduction
from .profile.normalized import ReporterResponseProfile
from .profile.provenance import ReaderEvidenceProvenance
from .profile.response import ControlAssignment, DoseResponse, PairingPolicy
from .profile.uncertainty import (
    DoseUncertainty,
    EstimatedMetricUncertainty,
    NotEstimableMetricUncertainty,
    ProfileEligibility,
    UncertaintyPolicy,
)
from .temporal import TemporalPolicyProjection


def profile_from_dict(
    payload: Mapping[str, object],
    *,
    evidence_bindings: ReaderEvidenceBindingSet,
) -> ReporterResponseProfile | ReporterMeasurementProfile:
    """Parse and canonically revalidate one serialized profile payload."""

    try:
        return _parse_profile(payload, evidence_bindings=evidence_bindings)
    except ReporterResponseContractError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise ReporterResponseContractError("profile payload is malformed") from exc


def _parse_profile(
    payload: Mapping[str, object],
    *,
    evidence_bindings: ReaderEvidenceBindingSet,
) -> ReporterResponseProfile | ReporterMeasurementProfile:
    if payload.get("contract_id") == MEASUREMENT_PROFILE_CONTRACT_ID:
        return _parse_measurement_profile(payload, evidence_bindings=evidence_bindings)
    root = _strict_object(
        payload,
        name="profile",
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
    provenance_payload = _mapping(root["provenance"], name="provenance")
    provenance = ReaderEvidenceProvenance._from_source_closed_bindings(
        evidence_bindings=evidence_bindings,
        subject_id=root["subject_id"],
        raw_design_id=provenance_payload.get("raw_design_id"),
        raw_assay_subject_id=provenance_payload.get("raw_assay_subject_id"),
    )
    if root["provenance"] != _public_provenance_payload(provenance):
        raise ReporterResponseContractError(
            "serialized provenance must equal provenance rederived from the verified evidence-binding artifact"
        )
    observation_policy = _observation_policy(root["observation_policy"])
    reduction_payload = _mapping(root["reduction"], name="reduction")
    reduction: Reduction
    if reduction_payload.get("kind") == "endpoint":
        values = _strict_dataclass(reduction_payload, EndpointReduction)
        values["temporal_policy"] = _temporal_policy(values["temporal_policy"])
        reduction = EndpointReduction(**values)
    elif reduction_payload.get("kind") == "time_window":
        values = _strict_dataclass(reduction_payload, TimeWindowReduction)
        values["temporal_policy"] = _temporal_policy(values["temporal_policy"])
        reduction = TimeWindowReduction(**values)
    else:
        raise ReporterResponseContractError("reduction.kind must be endpoint or time_window")
    measurements = tuple(
        ConditionMeasurement(**_strict_dataclass(row, ConditionMeasurement))
        for row in _object_list(root["measurements"], name="measurements")
    )
    pairing_payload = _strict_object(root["pairing_policy"], name="pairing_policy", fields={"kind", "assignments"})
    pairing = PairingPolicy(
        kind=pairing_payload["kind"],
        assignments=tuple(
            _control_assignment(row)
            for row in _object_list(pairing_payload["assignments"], name="pairing_policy.assignments")
        ),
    )
    responses = tuple(_dose_response(row) for row in _object_list(root["dose_responses"], name="dose_responses"))
    uncertainties: list[DoseUncertainty] = []
    for row in _object_list(root["dose_uncertainties"], name="dose_uncertainties"):
        item = _strict_object(
            row,
            name="dose_uncertainty",
            fields={
                "dose_uM",
                "biological_replicate_count",
                "normalized_reporter_response",
                "relative_od",
            },
        )
        uncertainties.append(
            DoseUncertainty(
                dose_uM=item["dose_uM"],
                biological_replicate_count=item["biological_replicate_count"],
                normalized_reporter_response=_metric_uncertainty(item["normalized_reporter_response"]),
                relative_od=_metric_uncertainty(item["relative_od"]),
            )
        )
    eligibility_payload = _strict_dataclass(root["eligibility"], ProfileEligibility)
    eligibility_payload["reasons"] = tuple(eligibility_payload["reasons"])
    profile = ReporterResponseProfile(
        contract_id=root["contract_id"],
        study_id=root["study_id"],
        profile_id=root["profile_id"],
        subject_id=root["subject_id"],
        provenance=provenance,
        observation_policy=observation_policy,
        reduction=reduction,
        dose_grid_uM=_number_tuple(root["dose_grid_uM"], name="dose_grid_uM"),
        measurements=measurements,
        pairing_policy=pairing,
        dose_uncertainties=tuple(uncertainties),
        eligibility=ProfileEligibility(**eligibility_payload),
    )
    if responses != profile.dose_responses:
        raise ReporterResponseContractError(
            "serialized dose_responses must equal canonical responses recomputed from measurements and pairing"
        )
    if root["comparability_key"] != profile.comparability_key:
        raise ReporterResponseContractError("serialized comparability_key must equal the canonical comparison identity")
    return profile


def _parse_measurement_profile(
    payload: Mapping[str, object],
    *,
    evidence_bindings: ReaderEvidenceBindingSet,
) -> ReporterMeasurementProfile:
    root = _strict_object(
        payload,
        name="measurement profile",
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
    provenance_payload = _mapping(root["provenance"], name="provenance")
    provenance = ReaderEvidenceProvenance._from_source_closed_bindings(
        evidence_bindings=evidence_bindings,
        subject_id=root["subject_id"],
        raw_design_id=provenance_payload.get("raw_design_id"),
        raw_assay_subject_id=provenance_payload.get("raw_assay_subject_id"),
    )
    if root["provenance"] != _public_provenance_payload(provenance):
        raise ReporterResponseContractError(
            "serialized provenance must equal provenance rederived from the verified evidence-binding artifact"
        )
    reduction_payload = _mapping(root["reduction"], name="reduction")
    reduction: Reduction
    if reduction_payload.get("kind") == "endpoint":
        values = _strict_dataclass(reduction_payload, EndpointReduction)
        values["temporal_policy"] = _temporal_policy(values["temporal_policy"])
        reduction = EndpointReduction(**values)
    elif reduction_payload.get("kind") == "time_window":
        values = _strict_dataclass(reduction_payload, TimeWindowReduction)
        values["temporal_policy"] = _temporal_policy(values["temporal_policy"])
        reduction = TimeWindowReduction(**values)
    else:
        raise ReporterResponseContractError("reduction.kind must be endpoint or time_window")
    reference_values = _strict_dataclass(
        root["reference_normalization"],
        ReferenceNormalizationUnavailable,
    )
    eligibility_values = _strict_dataclass(root["eligibility"], ProfileEligibility)
    eligibility_values["reasons"] = tuple(eligibility_values["reasons"])
    profile = ReporterMeasurementProfile(
        contract_id=root["contract_id"],
        study_id=root["study_id"],
        profile_id=root["profile_id"],
        subject_id=root["subject_id"],
        provenance=provenance,
        observation_policy=_observation_policy(root["observation_policy"]),
        reduction=reduction,
        dose_grid_uM=_number_tuple(root["dose_grid_uM"], name="dose_grid_uM"),
        measurements=tuple(
            ConditionMeasurement(**_strict_dataclass(row, ConditionMeasurement))
            for row in _object_list(root["measurements"], name="measurements")
        ),
        reference_normalization=ReferenceNormalizationUnavailable(**reference_values),
        eligibility=ProfileEligibility(**eligibility_values),
    )
    if root["comparability_key"] != profile.comparability_key:
        raise ReporterResponseContractError("serialized comparability_key must equal the canonical comparison identity")
    return profile


def _public_provenance_payload(provenance: ReaderEvidenceProvenance) -> dict[str, object]:
    return {
        "raw_design_id": provenance.raw_design_id,
        "raw_assay_subject_id": provenance.raw_assay_subject_id,
        "reader_experiment_id": provenance.reader_experiment_id,
        "reader_protocol_id": provenance.reader_protocol_id,
        "reader_record_id": provenance.reader_record_id,
        "reader_record_kind": provenance.reader_record_kind,
        "reader_record_revision": provenance.reader_record_revision,
        "reader_record_revision_digest": provenance.reader_record_revision_digest,
        "reader_record_config_digest": provenance.reader_record_config_digest,
        "reader_record_producer_config_digest": provenance.reader_record_producer_config_digest,
        "reader_record_producer": provenance.reader_record_producer.to_dict(),
        "reader_record_inputs": [item.to_dict() for item in provenance.reader_record_inputs],
        "reader_record_content_digest": provenance.reader_record_content_digest,
        "reader_record_schema_version": provenance.reader_record_schema_version,
        "reader_record_contract_id": provenance.reader_record_contract_id,
        "reader_record_path": provenance.reader_record_path,
        "evidence_binding_artifact_id": provenance.evidence_binding_artifact_id,
        "evidence_binding_artifact_digest": provenance.evidence_binding_artifact_digest,
    }


def _observation_policy(value: object) -> ReporterResponseObservationPolicy:
    payload = _strict_object(
        value,
        name="observation_policy",
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
        **_strict_dataclass(payload["biological_replicate_uncertainty_policy"], UncertaintyPolicy)
    )
    policy = ReporterResponseObservationPolicy(
        policy_id=payload["policy_id"],
        pairing_kind=payload["pairing_kind"],
        within_acquisition_reduction_statistic=payload["within_acquisition_reduction_statistic"],
        biological_replicate_uncertainty_policy=uncertainty,
    )
    expected = _json_value(asdict(policy))
    if payload != expected:
        raise ReporterResponseContractError(
            "serialized observation policy digest and fixed semantics must equal the canonical policy"
        )
    return policy


def _temporal_policy(value: object) -> TemporalPolicyProjection:
    payload = _strict_object(
        value,
        name="temporal policy",
        fields={"selection", "method", "output_space", "support", "digest"},
    )
    projection = TemporalPolicyProjection.from_reader_mapping(
        {key: item for key, item in payload.items() if key != "digest"}
    )
    expected = _json_value(asdict(projection))
    if value != expected:
        raise ReporterResponseContractError("serialized temporal policy digest must equal the canonical projection")
    return projection


def _metric_uncertainty(value: object) -> EstimatedMetricUncertainty | NotEstimableMetricUncertainty:
    payload = _mapping(value, name="metric_uncertainty")
    if payload.get("status") == "estimated":
        return EstimatedMetricUncertainty(**_strict_dataclass(payload, EstimatedMetricUncertainty))
    if payload.get("status") == "not_estimable":
        return NotEstimableMetricUncertainty(**_strict_dataclass(payload, NotEstimableMetricUncertainty))
    raise ReporterResponseContractError("metric uncertainty status must be estimated or not_estimable")


def _control_assignment(value: object) -> ControlAssignment:
    payload = _strict_dataclass(value, ControlAssignment)
    payload["baseline_observation_ids"] = tuple(payload["baseline_observation_ids"])
    payload["positive_control_observation_ids"] = tuple(payload["positive_control_observation_ids"])
    return ControlAssignment(**payload)


def _dose_response(value: object) -> DoseResponse:
    payload = _strict_dataclass(value, DoseResponse)
    payload["baseline_observation_ids"] = tuple(payload["baseline_observation_ids"])
    payload["positive_control_observation_ids"] = tuple(payload["positive_control_observation_ids"])
    return DoseResponse(**payload)


def _strict_dataclass(value: object, cls: type[object]) -> dict[str, object]:
    declared_fields = fields(cls)
    payload = _strict_object(value, name=cls.__name__, fields={item.name for item in declared_fields})
    return {item.name: payload[item.name] for item in declared_fields if item.init}


def _strict_object(value: object, *, name: str, fields: set[str]) -> dict[str, object]:
    payload = _mapping(value, name=name)
    actual = set(payload)
    if actual != fields:
        missing = sorted(fields - actual)
        extra = sorted(actual - fields)
        raise ReporterResponseContractError(f"{name} fields must match exactly; missing={missing}, extra={extra}")
    return dict(payload)


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ReporterResponseContractError(f"{name} must be an object with string keys")
    return value


def _object_list(value: object, *, name: str) -> tuple[Mapping[str, object], ...]:
    if not isinstance(value, list):
        raise ReporterResponseContractError(f"{name} must be an array")
    return tuple(_mapping(row, name=name) for row in value)


def _number_tuple(value: object, *, name: str) -> tuple[float, ...]:
    if not isinstance(value, list):
        raise ReporterResponseContractError(f"{name} must be an array")
    return tuple(value)


__all__ = ["profile_from_dict"]
