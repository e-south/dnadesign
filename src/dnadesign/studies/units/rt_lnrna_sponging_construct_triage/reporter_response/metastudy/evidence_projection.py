"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/reporter_response/metastudy/evidence_projection.py

Typed content projections for offline meta-study decision re-evaluation.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from .._contract_values import json_value
from ..canonical import comparability_key, derive_profile_rows
from ..policy import ReporterResponseObservationPolicy
from ..profile import (
    ConditionMeasurement,
    ControlAssignment,
    DoseResponse,
    DoseUncertainty,
    EndpointReduction,
    EstimatedMetricUncertainty,
    NotEstimableMetricUncertainty,
    PairingPolicy,
    ProfileEligibility,
    TimeWindowReduction,
    UncertaintyPolicy,
)
from ..temporal import TemporalPolicyProjection
from .contracts import (
    GrowthPhaseStratum,
    ProfileAuditArtifact,
    canonical_digest,
)


@dataclass(frozen=True, slots=True)
class ProfileProvenanceProjection:
    """Serialized provenance identity without source-closure authority."""

    raw_design_id: str | None
    raw_assay_subject_id: str | None
    reader_experiment_id: str
    reader_protocol_id: str
    reader_record_id: str
    reader_record_kind: str
    reader_record_revision: int
    reader_record_revision_digest: str
    reader_record_content_digest: str
    reader_record_schema_version: int
    reader_record_contract_id: str
    reader_record_path: str
    evidence_binding_artifact_id: str
    evidence_binding_artifact_digest: str

    def __post_init__(self) -> None:
        if self.raw_design_id is None and self.raw_assay_subject_id is None:
            raise ValueError("provenance requires at least one raw Reader identity")
        for name in ("raw_design_id", "raw_assay_subject_id"):
            value = getattr(self, name)
            if value is not None:
                _required_text(value, label=name)
        for name in (
            "reader_experiment_id",
            "reader_protocol_id",
            "reader_record_id",
            "reader_record_kind",
            "reader_record_contract_id",
            "reader_record_path",
        ):
            _required_text(getattr(self, name), label=name)
        if self.reader_record_kind != "dataframe_artifact":
            raise ValueError("reader_record_kind must equal dataframe_artifact")
        record_path = Path(self.reader_record_path)
        if record_path.is_absolute() or ".." in record_path.parts:
            raise ValueError("reader_record_path must be outputs-relative")


@dataclass(frozen=True, slots=True)
class ProfileContentProjection:
    """Canonical profile content sufficient to repeat meta-study evaluation.

    This projection proves internal content and decision consistency only. It is
    deliberately not a ``ReporterResponseProfile`` and cannot claim that Reader
    records, evidence bindings, or raw-observation audits remain source closed.
    """

    profile_id: str
    subject_id: str
    provenance: ProfileProvenanceProjection
    observation_policy: ReporterResponseObservationPolicy
    reduction: EndpointReduction | TimeWindowReduction
    dose_grid_uM: tuple[float, ...]
    measurements: tuple[ConditionMeasurement, ...]
    pairing_policy: PairingPolicy
    dose_uncertainties: tuple[DoseUncertainty, ...]
    dose_responses: tuple[DoseResponse, ...]
    comparability_key: str
    serialized_payload: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class ProfileEvidenceProjection:
    """One structurally verified profile and audit for offline re-evaluation."""

    profile: ProfileContentProjection
    audit: ProfileAuditArtifact


def parse_profile_evidence_projection(value: object, *, index: int) -> ProfileEvidenceProjection:
    """Parse bundled evidence without minting live-source closure tokens."""

    row = _strict_object(value, label=f"publication evidence profiles[{index}]", fields={"profile", "audit"})
    profile_payload = _strict_object(
        row["profile"],
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
        **_strict_dataclass(profile_payload["provenance"], ProfileProvenanceProjection)
    )
    policy = _parse_observation_policy(profile_payload["observation_policy"])
    reduction = _parse_reduction(profile_payload["reduction"])
    measurements = tuple(
        ConditionMeasurement(**_strict_dataclass(item, ConditionMeasurement))
        for item in _object_list(profile_payload["measurements"], label="measurements")
    )
    pairing_payload = _strict_object(
        profile_payload["pairing_policy"],
        label="pairing_policy",
        fields={"kind", "assignments"},
    )
    pairing = PairingPolicy(
        kind=pairing_payload["kind"],
        assignments=tuple(
            _parse_control_assignment(item)
            for item in _object_list(pairing_payload["assignments"], label="pairing_policy.assignments")
        ),
    )
    uncertainties = tuple(
        _parse_dose_uncertainty(item)
        for item in _object_list(profile_payload["dose_uncertainties"], label="dose_uncertainties")
    )
    eligibility_values = _strict_dataclass(profile_payload["eligibility"], ProfileEligibility)
    eligibility_values["reasons"] = tuple(eligibility_values["reasons"])
    eligibility = ProfileEligibility(**eligibility_values)
    dose_grid, canonical_measurements, responses, canonical_uncertainties = derive_profile_rows(
        reduction=reduction,
        dose_grid_uM=_number_tuple(profile_payload["dose_grid_uM"], label="dose_grid_uM"),
        measurements=measurements,
        pairing_policy=pairing,
        observation_policy=policy,
        dose_uncertainties=uncertainties,
    )
    declared_responses = tuple(
        _parse_dose_response(item) for item in _object_list(profile_payload["dose_responses"], label="dose_responses")
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
        profile_id=_required_text(profile_payload["profile_id"], label="profile_id"),
        subject_id=_required_text(profile_payload["subject_id"], label="subject_id"),
        provenance=provenance,
        observation_policy=policy,
        reduction=reduction,
        dose_grid_uM=dose_grid,
        measurements=canonical_measurements,
        pairing_policy=pairing,
        dose_uncertainties=canonical_uncertainties,
        dose_responses=responses,
        comparability_key=expected_key,
        serialized_payload=dict(profile_payload),
    )
    # Constructing eligibility is itself the exact schema and fixed-value check.
    del eligibility
    audit_payload = _strict_object(
        row["audit"],
        label="audit",
        fields={item.name for item in fields(ProfileAuditArtifact) if item.name != "_derivation_closure"},
    )
    audit_values = {item.name: audit_payload[item.name] for item in fields(ProfileAuditArtifact) if item.init}
    audit_values["growth_phase_strata"] = tuple(
        GrowthPhaseStratum(**_strict_dataclass(item, GrowthPhaseStratum))
        for item in _object_list(audit_values["growth_phase_strata"], label="audit.growth_phase_strata")
    )
    audit = ProfileAuditArtifact(**audit_values)
    audit_without_digest = dict(audit_payload)
    artifact_digest = audit_without_digest.pop("artifact_digest")
    if artifact_digest != canonical_digest(audit_without_digest):
        raise ValueError("publication evidence audit digest mismatch")
    if audit.profile_digest != canonical_digest(profile_payload):
        raise ValueError("publication evidence profile digest mismatch")
    if audit.profile_source_digest != canonical_digest(profile_source_identity_projection(projection)):
        raise ValueError("publication evidence profile source digest mismatch")
    return ProfileEvidenceProjection(profile=projection, audit=audit)


def profile_source_identity_projection(profile: ProfileContentProjection) -> dict[str, object]:
    """Return the same source-identity coordinate without claiming authenticity."""

    provenance = profile.provenance
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
        "observation_policy_identity": profile.observation_policy.digest,
    }


def _parse_observation_policy(value: object) -> ReporterResponseObservationPolicy:
    payload = _strict_object(
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
        **_strict_dataclass(payload["biological_replicate_uncertainty_policy"], UncertaintyPolicy)
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
    payload = _mapping(value, label="reduction")
    kind = payload.get("kind")
    cls = EndpointReduction if kind == "endpoint" else TimeWindowReduction if kind == "time_window" else None
    if cls is None:
        raise ValueError("reduction.kind must be endpoint or time_window")
    values = _strict_dataclass(payload, cls)
    temporal = _strict_object(
        values["temporal_policy"],
        label="temporal_policy",
        fields={"selection", "method", "output_space", "support", "digest"},
    )
    projection = TemporalPolicyProjection.from_reader_mapping(
        {key: item for key, item in temporal.items() if key != "digest"}
    )
    if temporal != json_value(asdict(projection)):
        raise ValueError("serialized temporal policy differs from its canonical form")
    values["temporal_policy"] = projection
    return cls(**values)


def _parse_control_assignment(value: object) -> ControlAssignment:
    values = _strict_dataclass(value, ControlAssignment)
    values["baseline_observation_ids"] = tuple(values["baseline_observation_ids"])
    values["positive_control_observation_ids"] = tuple(values["positive_control_observation_ids"])
    return ControlAssignment(**values)


def _parse_dose_response(value: object) -> DoseResponse:
    values = _strict_dataclass(value, DoseResponse)
    values["baseline_observation_ids"] = tuple(values["baseline_observation_ids"])
    values["positive_control_observation_ids"] = tuple(values["positive_control_observation_ids"])
    return DoseResponse(**values)


def _parse_dose_uncertainty(value: object) -> DoseUncertainty:
    payload = _strict_object(
        value,
        label="dose_uncertainty",
        fields={"dose_uM", "biological_replicate_count", "normalized_reporter_response", "relative_od"},
    )
    return DoseUncertainty(
        dose_uM=payload["dose_uM"],
        biological_replicate_count=payload["biological_replicate_count"],
        normalized_reporter_response=_parse_metric_uncertainty(payload["normalized_reporter_response"]),
        relative_od=_parse_metric_uncertainty(payload["relative_od"]),
    )


def _parse_metric_uncertainty(value: object) -> EstimatedMetricUncertainty | NotEstimableMetricUncertainty:
    payload = _mapping(value, label="metric_uncertainty")
    if payload.get("status") == "estimated":
        return EstimatedMetricUncertainty(**_strict_dataclass(payload, EstimatedMetricUncertainty))
    if payload.get("status") == "not_estimable":
        return NotEstimableMetricUncertainty(**_strict_dataclass(payload, NotEstimableMetricUncertainty))
    raise ValueError("metric uncertainty status is invalid")


def _strict_dataclass(value: object, cls: type[object]) -> dict[str, object]:
    declared = fields(cls)
    payload = _strict_object(value, label=cls.__name__, fields={item.name for item in declared})
    return {item.name: payload[item.name] for item in declared if item.init}


def _strict_object(value: object, *, label: str, fields: set[str]) -> dict[str, object]:
    payload = _mapping(value, label=label)
    if set(payload) != fields:
        raise ValueError(f"{label} fields do not match the exact contract")
    return payload


def _mapping(value: object, *, label: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return dict(value)


def _object_list(value: object, *, label: str) -> tuple[dict[str, object], ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{label} must be an array")
    return tuple(_mapping(item, label=f"{label}[]") for item in value)


def _number_tuple(value: object, *, label: str) -> tuple[float, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be an array")
    return tuple(value)


def _required_text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be non-empty text")
    return value


__all__ = [
    "ProfileContentProjection",
    "ProfileEvidenceProjection",
    "ProfileProvenanceProjection",
    "parse_profile_evidence_projection",
    "profile_source_identity_projection",
]
