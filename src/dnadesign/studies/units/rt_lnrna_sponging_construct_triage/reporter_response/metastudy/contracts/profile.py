"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/reporter_response/metastudy/contracts/profile.py

Profile evidence and canonical profile-audit contracts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Literal

from ...measurement_profile import DescriptiveReporterProfile, ReporterMeasurementProfile
from ...profile.measurement import EndpointReduction, TimeWindowReduction
from ...profile.normalized import ReporterResponseProfile
from ._values import MetastudyContractError, _digest, _nonnegative, _required_text, canonical_digest
from .profile_identity import profile_digest, profile_source_identity_payload

_AUDIT_DERIVATION_TOKEN = object()


@dataclass(frozen=True, slots=True)
class GrowthPhaseStratum:
    """Normalized one-hour log-normalizer slopes for one study condition."""

    condition_id: str
    normalized_start_slope: float
    normalized_end_slope: float

    def __post_init__(self) -> None:
        _required_text(self.condition_id, label="growth-phase condition_id")
        for name in ("normalized_start_slope", "normalized_end_slope"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise MetastudyContractError(f"{name} must be finite")


@dataclass(frozen=True, slots=True)
class ProfileAuditArtifact:
    """One canonical audit artifact bound to an exact profile source identity."""

    contract_id: Literal["rt_lnrna_reporter_response_profile_audit.v3"]
    method_id: Literal["synthetic_profile_audit_v1", "canonical_profile_observation_audit_v1"]
    profile_source_digest: str
    profile_digest: str
    condition_ontology_digest: str
    within_acquisition_observation_range: float
    reference_within_acquisition_observation_range: float
    required_observation_count: int
    overflow_observation_count: int
    clipped_observation_count: int
    growth_phase_strata: tuple[GrowthPhaseStratum, ...]
    artifact_digest: str
    _derivation_closure: object | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.contract_id != "rt_lnrna_reporter_response_profile_audit.v3":
            raise MetastudyContractError("profile audit contract_id changed")
        if self.method_id not in {"synthetic_profile_audit_v1", "canonical_profile_observation_audit_v1"}:
            raise MetastudyContractError("profile audit method_id is not enumerated")
        _digest(self.profile_source_digest, label="profile audit profile_source_digest")
        _digest(self.profile_digest, label="profile audit profile_digest")
        _digest(self.condition_ontology_digest, label="profile audit condition_ontology_digest")
        width = _nonnegative(
            self.within_acquisition_observation_range,
            label="profile audit within_acquisition_observation_range",
        )
        reference = _nonnegative(
            self.reference_within_acquisition_observation_range,
            label="profile audit reference_within_acquisition_observation_range",
        )
        if reference == 0.0 and width != 0.0:
            raise MetastudyContractError("a zero reference observation range cannot support a nonzero range")
        for name in (
            "required_observation_count",
            "overflow_observation_count",
            "clipped_observation_count",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise MetastudyContractError(f"{name} must be a non-negative integer")
        if not isinstance(self.growth_phase_strata, tuple) or not all(
            isinstance(row, GrowthPhaseStratum) for row in self.growth_phase_strata
        ):
            raise MetastudyContractError("profile audit growth_phase_strata must be a typed tuple")
        condition_ids = tuple(row.condition_id for row in self.growth_phase_strata)
        if condition_ids != tuple(sorted(condition_ids)) or len(condition_ids) != len(set(condition_ids)):
            raise MetastudyContractError("profile audit growth-phase strata must use unique canonical condition order")
        _digest(self.artifact_digest, label="profile audit artifact_digest")

    @classmethod
    def _from_canonical_derivation(cls, **values: object) -> ProfileAuditArtifact:
        audit = cls(**values)
        object.__setattr__(audit, "_derivation_closure", _AUDIT_DERIVATION_TOKEN)
        return audit

    @property
    def is_derivation_closed(self) -> bool:
        return self._derivation_closure is _AUDIT_DERIVATION_TOKEN


def profile_audit_payload(audit: ProfileAuditArtifact, *, include_digest: bool = True) -> dict[str, object]:
    """Serialize one profile audit canonically."""

    payload = asdict(audit)
    payload.pop("_derivation_closure", None)
    if not include_digest:
        payload.pop("artifact_digest")
    return payload


@dataclass(frozen=True, slots=True)
class ProfileEvidence:
    """One canonical profile plus digest-bound within-acquisition range evidence."""

    profile: DescriptiveReporterProfile
    audit: ProfileAuditArtifact

    def __post_init__(self) -> None:
        if not isinstance(self.profile, (ReporterResponseProfile, ReporterMeasurementProfile)):
            raise MetastudyContractError("profile evidence must contain a typed reporter profile")
        if not isinstance(self.audit, ProfileAuditArtifact):
            raise MetastudyContractError("profile evidence requires ProfileAuditArtifact")
        expected_source = canonical_digest(profile_source_identity_payload(self.profile))
        if self.audit.profile_source_digest != expected_source:
            raise MetastudyContractError("profile audit source identity digest mismatch")
        if self.audit.profile_digest != profile_digest(self.profile):
            raise MetastudyContractError("profile audit full profile digest mismatch")
        if self.audit.artifact_digest != canonical_digest(profile_audit_payload(self.audit, include_digest=False)):
            raise MetastudyContractError("profile audit artifact digest mismatch")
        if isinstance(self.profile.reduction, TimeWindowReduction) and not self.audit.growth_phase_strata:
            raise MetastudyContractError("time-window profile evidence requires growth-phase strata")
        if isinstance(self.profile.reduction, EndpointReduction) and self.audit.growth_phase_strata:
            raise MetastudyContractError("endpoint profile evidence cannot contain growth-phase strata")


__all__ = [
    "GrowthPhaseStratum",
    "ProfileAuditArtifact",
    "ProfileEvidence",
    "profile_audit_payload",
]
