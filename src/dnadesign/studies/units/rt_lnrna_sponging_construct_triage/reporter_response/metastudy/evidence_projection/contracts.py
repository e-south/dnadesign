"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/reporter_response/metastudy/evidence_projection/contracts.py

Non-authoritative profile content projections for offline evaluation.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from ...measurement_profile import ReferenceNormalizationUnavailable
from ...policy import ReporterResponseObservationPolicy
from ...profile.measurement import ConditionMeasurement, EndpointReduction, TimeWindowReduction
from ...profile.response import DoseResponse, PairingPolicy
from ...profile.uncertainty import DoseUncertainty
from ..contracts.profile import ProfileAuditArtifact
from .provenance import ProfileProvenanceProjection


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


__all__ = [
    "ProfileContentProjection",
    "ProfileEvidenceProjection",
]
