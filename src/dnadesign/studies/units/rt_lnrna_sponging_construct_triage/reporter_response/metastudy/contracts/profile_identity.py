"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/reporter_response/metastudy/contracts/profile_identity.py

Canonical identity payloads for profile-bound meta-study evidence.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from ...measurement_profile import DescriptiveReporterProfile
from ...serialization import profile_to_dict
from ._values import canonical_digest


def profile_source_identity_payload(profile: DescriptiveReporterProfile) -> dict[str, object]:
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
        "observation_policy_identity": profile.observation_policy.digest,
    }


def profile_digest(profile: DescriptiveReporterProfile) -> str:
    """Digest the complete canonical profile, including measurements and reduction."""

    return canonical_digest(profile_to_dict(profile))


__all__ = ["profile_digest", "profile_source_identity_payload"]
