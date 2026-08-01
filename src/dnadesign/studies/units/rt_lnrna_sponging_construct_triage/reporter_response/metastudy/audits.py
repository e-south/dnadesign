"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/reporter_response/metastudy/audits.py

Canonical profile-bound audit construction and identity.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Literal

from .. import profile_to_dict
from ..profile import ReporterResponseProfile, TimeWindowReduction
from .contracts._values import canonical_digest
from .contracts.profile import GrowthPhaseStratum, ProfileAuditArtifact
from .contracts.protocol import CANONICAL_CONDITION_ONTOLOGY_DIGEST


def profile_source_identity_payload(profile: ReporterResponseProfile) -> dict[str, object]:
    """Return the exact source identity to which one profile audit must bind."""

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
        "observation_policy_identity": _profile_policy_identity(profile),
    }


def _profile_policy_identity(profile: ReporterResponseProfile) -> object:
    """Single compatibility seam for the canonical reporter-response observation policy."""

    return profile.observation_policy.digest


def profile_digest(profile: ReporterResponseProfile) -> str:
    """Digest the complete canonical profile, including measurements and reduction."""

    return canonical_digest(profile_to_dict(profile))


def build_profile_audit_artifact(
    profile: ReporterResponseProfile,
    *,
    method_id: Literal["synthetic_profile_audit_v1", "canonical_profile_observation_audit_v1"],
    within_acquisition_observation_range: float,
    reference_within_acquisition_observation_range: float,
    required_observation_count: int,
    overflow_observation_count: int,
    clipped_observation_count: int,
    growth_phase_strata: tuple[GrowthPhaseStratum, ...] | None = None,
) -> ProfileAuditArtifact:
    """Build explicitly synthetic audit data for contract-level tests only."""

    if method_id != "synthetic_profile_audit_v1":
        raise ValueError("canonical audits are derived only by the source-closed materializer")
    phase = growth_phase_strata
    if phase is None:
        phase = (
            (GrowthPhaseStratum("synthetic", 1.0, 0.5),) if isinstance(profile.reduction, TimeWindowReduction) else ()
        )
    return _build_profile_audit(
        profile,
        method_id=method_id,
        condition_ontology_digest=CANONICAL_CONDITION_ONTOLOGY_DIGEST,
        within_acquisition_observation_range=within_acquisition_observation_range,
        reference_within_acquisition_observation_range=reference_within_acquisition_observation_range,
        required_observation_count=required_observation_count,
        overflow_observation_count=overflow_observation_count,
        clipped_observation_count=clipped_observation_count,
        growth_phase_strata=phase,
        source_closed=False,
    )


def _build_derivation_closed_profile_audit(
    profile: ReporterResponseProfile,
    *,
    method_id: object | None = None,
    within_acquisition_observation_range: float,
    reference_within_acquisition_observation_range: float,
    required_observation_count: int,
    overflow_observation_count: int,
    clipped_observation_count: int,
    growth_phase_strata: tuple[GrowthPhaseStratum, ...] | None = None,
    condition_ontology_digest: str = CANONICAL_CONDITION_ONTOLOGY_DIGEST,
) -> ProfileAuditArtifact:
    """Build the audit used only after raw source observations were rederived."""

    if method_id not in {None, "synthetic_profile_audit_v1", "canonical_profile_observation_audit_v1"}:
        raise ValueError("unknown internal audit method marker")
    phase = growth_phase_strata
    if phase is None:
        phase = (
            (GrowthPhaseStratum("synthetic", 1.0, 0.5),) if isinstance(profile.reduction, TimeWindowReduction) else ()
        )
    return _build_profile_audit(
        profile,
        method_id="canonical_profile_observation_audit_v1",
        condition_ontology_digest=condition_ontology_digest,
        within_acquisition_observation_range=within_acquisition_observation_range,
        reference_within_acquisition_observation_range=reference_within_acquisition_observation_range,
        required_observation_count=required_observation_count,
        overflow_observation_count=overflow_observation_count,
        clipped_observation_count=clipped_observation_count,
        growth_phase_strata=phase,
        source_closed=True,
    )


def _build_profile_audit(
    profile: ReporterResponseProfile,
    *,
    method_id: Literal["synthetic_profile_audit_v1", "canonical_profile_observation_audit_v1"],
    condition_ontology_digest: str,
    within_acquisition_observation_range: float,
    reference_within_acquisition_observation_range: float,
    required_observation_count: int,
    overflow_observation_count: int,
    clipped_observation_count: int,
    growth_phase_strata: tuple[GrowthPhaseStratum, ...],
    source_closed: bool,
) -> ProfileAuditArtifact:
    values = {
        "contract_id": "rt_lnrna_reporter_response_profile_audit.v3",
        "method_id": method_id,
        "profile_source_digest": canonical_digest(profile_source_identity_payload(profile)),
        "profile_digest": profile_digest(profile),
        "condition_ontology_digest": condition_ontology_digest,
        "within_acquisition_observation_range": within_acquisition_observation_range,
        "reference_within_acquisition_observation_range": reference_within_acquisition_observation_range,
        "required_observation_count": required_observation_count,
        "overflow_observation_count": overflow_observation_count,
        "clipped_observation_count": clipped_observation_count,
        "growth_phase_strata": growth_phase_strata,
    }
    digest_values = {
        **values,
        "growth_phase_strata": [asdict(row) for row in growth_phase_strata],
    }
    values["artifact_digest"] = canonical_digest(digest_values)
    if source_closed:
        return ProfileAuditArtifact._from_canonical_derivation(**values)
    return ProfileAuditArtifact(**values)


def profile_audit_payload(audit: ProfileAuditArtifact, *, include_digest: bool = True) -> dict[str, object]:
    """Serialize one profile audit canonically."""

    payload = asdict(audit)
    payload.pop("_derivation_closure", None)
    if not include_digest:
        payload.pop("artifact_digest")
    return payload


__all__ = [
    "build_profile_audit_artifact",
    "profile_audit_payload",
    "profile_digest",
    "profile_source_identity_payload",
]
