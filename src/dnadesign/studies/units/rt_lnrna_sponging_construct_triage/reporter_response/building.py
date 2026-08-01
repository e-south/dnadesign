"""Source-closed construction of descriptive reporter-response profiles."""

from __future__ import annotations

from collections.abc import Iterable

from ..reader_evidence import ReaderEvidenceBindingSet
from ._contract_values import required_text as _required_text
from .policy import ReporterResponseObservationPolicy
from .profile.measurement import ConditionMeasurement, Reduction
from .profile.normalized import CONTRACT_ID, STUDY_ID, ReporterResponseProfile
from .profile.provenance import ReaderEvidenceProvenance
from .profile.response import PairingPolicy
from .profile.uncertainty import DoseUncertainty, ProfileEligibility


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


__all__ = ["build_reporter_response_profile"]
