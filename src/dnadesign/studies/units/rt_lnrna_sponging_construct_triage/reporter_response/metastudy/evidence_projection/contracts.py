"""Non-authoritative profile content projections for offline evaluation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from ...measurement_profile import ReferenceNormalizationUnavailable
from ...policy import ReporterResponseObservationPolicy
from ...profile.measurement import ConditionMeasurement, EndpointReduction, TimeWindowReduction
from ...profile.response import DoseResponse, PairingPolicy
from ...profile.uncertainty import DoseUncertainty
from ..contracts.profile import ProfileAuditArtifact
from ._values import required_text


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
                required_text(value, label=name)
        for name in (
            "reader_experiment_id",
            "reader_protocol_id",
            "reader_record_id",
            "reader_record_kind",
            "reader_record_contract_id",
            "reader_record_path",
        ):
            required_text(getattr(self, name), label=name)
        if self.reader_record_kind != "dataframe_artifact":
            raise ValueError("reader_record_kind must equal dataframe_artifact")
        record_path = Path(self.reader_record_path)
        if record_path.is_absolute() or ".." in record_path.parts:
            raise ValueError("reader_record_path must be outputs-relative")


@dataclass(frozen=True, slots=True)
class ProfileContentProjection:
    """Canonical profile content sufficient to repeat offline evaluation."""

    profile_id: str
    subject_id: str
    provenance: ProfileProvenanceProjection
    observation_policy: ReporterResponseObservationPolicy
    reduction: EndpointReduction | TimeWindowReduction
    dose_grid_uM: tuple[float, ...]
    measurements: tuple[ConditionMeasurement, ...]
    pairing_policy: PairingPolicy | None
    dose_uncertainties: tuple[DoseUncertainty, ...]
    dose_responses: tuple[DoseResponse, ...]
    reference_normalization: ReferenceNormalizationUnavailable | None
    comparability_key: str
    serialized_payload: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class ProfileEvidenceProjection:
    """One structurally verified profile and audit for offline re-evaluation."""

    profile: ProfileContentProjection
    audit: ProfileAuditArtifact


def profile_source_identity_projection(profile: ProfileContentProjection) -> dict[str, object]:
    """Return the source coordinate without claiming current authenticity."""

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


__all__ = [
    "ProfileContentProjection",
    "ProfileEvidenceProjection",
    "ProfileProvenanceProjection",
    "profile_source_identity_projection",
]
