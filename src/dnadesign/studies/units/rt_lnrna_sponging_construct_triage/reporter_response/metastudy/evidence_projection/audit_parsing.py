"""Digest-closed parsing for serialized profile audit projections."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields

from ..contracts._values import canonical_digest
from ..contracts.profile import GrowthPhaseStratum, ProfileAuditArtifact
from ._values import object_list, strict_dataclass, strict_object
from .contracts import ProfileContentProjection
from .provenance import profile_source_identity_projection


def parse_profile_audit(
    value: object,
    *,
    profile_payload: Mapping[str, object],
    profile: ProfileContentProjection,
) -> ProfileAuditArtifact:
    audit_payload = strict_object(
        value,
        label="audit",
        fields={item.name for item in fields(ProfileAuditArtifact) if item.name != "_derivation_closure"},
    )
    audit_values = {item.name: audit_payload[item.name] for item in fields(ProfileAuditArtifact) if item.init}
    audit_values["growth_phase_strata"] = tuple(
        GrowthPhaseStratum(**strict_dataclass(item, GrowthPhaseStratum))
        for item in object_list(audit_values["growth_phase_strata"], label="audit.growth_phase_strata")
    )
    audit = ProfileAuditArtifact(**audit_values)
    audit_without_digest = dict(audit_payload)
    artifact_digest = audit_without_digest.pop("artifact_digest")
    if artifact_digest != canonical_digest(audit_without_digest):
        raise ValueError("publication evidence audit digest mismatch")
    if audit.profile_digest != canonical_digest(profile_payload):
        raise ValueError("publication evidence profile digest mismatch")
    if audit.profile_source_digest != canonical_digest(profile_source_identity_projection(profile)):
        raise ValueError("publication evidence profile source digest mismatch")
    return audit


__all__ = ["parse_profile_audit"]
