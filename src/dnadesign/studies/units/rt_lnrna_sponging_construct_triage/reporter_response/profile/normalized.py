"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/reporter_response/profile/normalized.py

Final normalized reporter-response profile contract.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .._contract_values import ReporterResponseContractError
from .._contract_values import required_text as _required_text
from .measurement import ConditionMeasurement, Reduction
from .provenance import ReaderEvidenceProvenance
from .response import DoseResponse, PairingPolicy
from .uncertainty import DoseUncertainty, ProfileEligibility

if TYPE_CHECKING:
    from ..policy import ReporterResponseObservationPolicy

CONTRACT_ID = "rt_lnrna_reporter_response_profile.v4"
STUDY_ID = "rt_lnrna_sponging_construct_triage"


@dataclass(frozen=True, slots=True)
class ReporterResponseProfile:
    """A descriptive dose profile with exact evidence and comparability identity."""

    contract_id: str
    study_id: str
    profile_id: str
    subject_id: str
    provenance: ReaderEvidenceProvenance
    observation_policy: ReporterResponseObservationPolicy
    reduction: Reduction
    dose_grid_uM: tuple[float, ...]
    measurements: tuple[ConditionMeasurement, ...]
    pairing_policy: PairingPolicy
    dose_uncertainties: tuple[DoseUncertainty, ...]
    eligibility: ProfileEligibility
    dose_responses: tuple[DoseResponse, ...] = field(init=False)
    comparability_key: str = field(init=False)

    def __post_init__(self) -> None:
        from ..canonical import comparability_key, derive_profile_rows
        from ..policy import ReporterResponseObservationPolicy

        if self.contract_id != CONTRACT_ID:
            raise ReporterResponseContractError(f"contract_id must equal {CONTRACT_ID!r}")
        if self.study_id != STUDY_ID:
            raise ReporterResponseContractError(f"study_id must equal {STUDY_ID!r}")
        _required_text(self.profile_id, field_name="profile_id")
        _required_text(self.subject_id, field_name="subject_id")
        if not isinstance(self.provenance, ReaderEvidenceProvenance) or not self.provenance.is_source_closed:
            raise ReporterResponseContractError(
                "provenance must be derived from a source-closed Reader evidence-binding set"
            )
        self.provenance.require_bound_subject(self.subject_id)
        if not isinstance(self.observation_policy, ReporterResponseObservationPolicy):
            raise ReporterResponseContractError("observation_policy must be ReporterResponseObservationPolicy")
        if not isinstance(self.eligibility, ProfileEligibility):
            raise ReporterResponseContractError("eligibility must be ProfileEligibility")
        self.provenance.require_biological_replicate_scopes(
            tuple((row.source_condition_value, row.biological_replicate_id) for row in self.measurements)
        )
        dose_grid, measurement_rows, expected_responses, uncertainty_rows = derive_profile_rows(
            reduction=self.reduction,
            dose_grid_uM=self.dose_grid_uM,
            measurements=self.measurements,
            pairing_policy=self.pairing_policy,
            observation_policy=self.observation_policy,
            dose_uncertainties=self.dose_uncertainties,
        )
        if self.dose_grid_uM != dose_grid:
            raise ReporterResponseContractError("dose_grid_uM must be stored as its canonical tuple")
        if self.measurements != measurement_rows:
            raise ReporterResponseContractError("measurements must be stored as their canonical tuple")
        if self.dose_uncertainties != uncertainty_rows:
            raise ReporterResponseContractError("dose_uncertainties must be stored as their canonical tuple")
        object.__setattr__(self, "dose_responses", expected_responses)
        object.__setattr__(
            self,
            "comparability_key",
            comparability_key(
                observation_policy_digest=self.observation_policy.digest,
                reduction=self.reduction,
                dose_grid_uM=self.dose_grid_uM,
                dose_uncertainties=self.dose_uncertainties,
            ),
        )
