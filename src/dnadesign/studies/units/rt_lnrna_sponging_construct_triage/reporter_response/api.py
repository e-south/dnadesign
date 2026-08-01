"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/reporter_response/api.py

Public construction, parsing, serialization, and comparison surfaces.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict

from ..reader_evidence import ReaderEvidenceBindingSet
from ._contract_values import json_value as _json_value
from ._contract_values import required_text as _required_text
from .policy import ReporterResponseObservationPolicy
from .profile import (
    CONTRACT_ID,
    STUDY_ID,
    ConditionMeasurement,
    ControlAssignment,
    DoseResponse,
    DoseUncertainty,
    EndpointReduction,
    EstimatedMetricUncertainty,
    NotEstimableMetricUncertainty,
    PairingPolicy,
    ProfileEligibility,
    ReaderEvidenceProvenance,
    Reduction,
    ReporterResponseContractError,
    ReporterResponseProfile,
    TimeWindowReduction,
    UncertaintyPolicy,
)
from .temporal import TemporalPolicyProjection


def build_reporter_response_profile(
    *,
    profile_id: str,
    subject_id: str,
    raw_design_id: str | None,
    raw_assay_subject_id: str | None,
    evidence_bindings: ReaderEvidenceBindingSet,
    observation_policy: ReporterResponseObservationPolicy,
    reduction: Reduction,
    dose_grid_uM: Iterable[float],
    measurements: Iterable[ConditionMeasurement],
    pairing_policy: PairingPolicy,
    dose_uncertainties: Iterable[DoseUncertainty],
    ineligibility_reasons: Iterable[str],
) -> ReporterResponseProfile:
    """Build a profile from bindings returned by source derivation or revalidating load."""

    _required_text(profile_id, field_name="profile_id")
    _required_text(subject_id, field_name="subject_id")
    provenance = ReaderEvidenceProvenance._from_source_closed_bindings(
        evidence_bindings=evidence_bindings,
        subject_id=subject_id,
        raw_design_id=raw_design_id,
        raw_assay_subject_id=raw_assay_subject_id,
    )
    eligibility = ProfileEligibility(
        evidence_use="descriptive",
        optimization_status="ineligible",
        reasons=tuple(ineligibility_reasons),
    )
    return ReporterResponseProfile(
        contract_id=CONTRACT_ID,
        study_id=STUDY_ID,
        profile_id=profile_id,
        subject_id=subject_id,
        provenance=provenance,
        observation_policy=observation_policy,
        reduction=reduction,
        dose_grid_uM=tuple(dose_grid_uM),
        measurements=tuple(measurements),
        pairing_policy=pairing_policy,
        dose_uncertainties=tuple(dose_uncertainties),
        eligibility=eligibility,
    )


def profile_to_dict(profile: ReporterResponseProfile) -> dict[str, object]:
    """Serialize one already-validated profile without objective fields."""

    if not isinstance(profile, ReporterResponseProfile):
        raise ReporterResponseContractError("profile must be ReporterResponseProfile")
    payload = _json_value(asdict(profile))
    assert isinstance(payload, dict)
    provenance = payload["provenance"]
    assert isinstance(provenance, dict)
    provenance.pop("_bound_subject_id", None)
    provenance.pop("_source_closed", None)
    provenance.pop("_declared_biological_replicate_scopes", None)
    return payload


def profile_from_dict(
    payload: Mapping[str, object],
    *,
    evidence_bindings: ReaderEvidenceBindingSet,
) -> ReporterResponseProfile:
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
) -> ReporterResponseProfile:
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
    eligibility = ProfileEligibility(**eligibility_payload)
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
        eligibility=eligibility,
    )
    if responses != profile.dose_responses:
        raise ReporterResponseContractError(
            "serialized dose_responses must equal canonical responses recomputed from measurements and pairing"
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


def require_comparable_profiles(profiles: Iterable[ReporterResponseProfile]) -> str:
    """Return the shared comparability key or reject aggregation."""

    rows = tuple(profiles)
    if len(rows) < 2:
        raise ReporterResponseContractError("cross-profile aggregation requires at least two profiles")
    if not all(isinstance(row, ReporterResponseProfile) for row in rows):
        raise ReporterResponseContractError("aggregation inputs must be ReporterResponseProfile values")
    expected = rows[0].comparability_key
    mismatches = [row.profile_id for row in rows[1:] if row.comparability_key != expected]
    if mismatches:
        raise ReporterResponseContractError(
            "cross-profile aggregation requires exactly matching comparability keys; "
            f"mismatched profiles: {', '.join(mismatches)}"
        )
    return expected


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
    from dataclasses import fields

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


__all__ = [
    "build_reporter_response_profile",
    "profile_from_dict",
    "profile_to_dict",
    "require_comparable_profiles",
]
