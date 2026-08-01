"""Study-owned reporter measurements that do not require a positive control."""

from __future__ import annotations

import hashlib
import json
import statistics
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from typing import Literal, TypeAlias

from ._contract_values import ReporterResponseContractError, json_value, ordered_dose_grid, required_text
from .policy import ReporterResponseObservationPolicy
from .profile.measurement import (
    ConditionMeasurement,
    EndpointReduction,
    Reduction,
    TimeWindowReduction,
    validate_ratio_reduction_semantics,
)
from .profile.normalized import STUDY_ID, ReporterResponseProfile
from .profile.provenance import ReaderEvidenceProvenance
from .profile.uncertainty import ProfileEligibility

MEASUREMENT_PROFILE_CONTRACT_ID = "rt_lnrna_reporter_measurement_profile.v1"
ReferenceNormalizationAbsenceReason = Literal[
    "positive_control_not_declared",
    "positive_control_observations_missing",
    "positive_control_separation_not_positive",
]


@dataclass(frozen=True, slots=True)
class ReferenceNormalizationUnavailable:
    """Typed reason that the stricter reference-normalized projection is absent."""

    reason: ReferenceNormalizationAbsenceReason
    positive_control_condition_id: str | None
    status: Literal["unavailable"] = field(default="unavailable", init=False)

    def __post_init__(self) -> None:
        if self.reason not in {
            "positive_control_not_declared",
            "positive_control_observations_missing",
            "positive_control_separation_not_positive",
        }:
            raise ReporterResponseContractError("reference-normalization absence reason is undeclared")
        if self.reason == "positive_control_not_declared":
            if self.positive_control_condition_id is not None:
                raise ReporterResponseContractError(
                    "an absent positive control cannot declare positive_control_condition_id"
                )
        elif self.positive_control_condition_id is None:
            raise ReporterResponseContractError("declared positive-control unavailability requires its condition id")
        else:
            required_text(
                self.positive_control_condition_id,
                field_name="positive_control_condition_id",
            )


@dataclass(frozen=True, slots=True)
class ReporterMeasurementProfile:
    """Raw reporter, OD600, and reporter/OD summaries for declared conditions."""

    contract_id: Literal["rt_lnrna_reporter_measurement_profile.v1"]
    study_id: str
    profile_id: str
    subject_id: str
    provenance: ReaderEvidenceProvenance
    observation_policy: ReporterResponseObservationPolicy
    reduction: Reduction
    dose_grid_uM: tuple[float, ...]
    measurements: tuple[ConditionMeasurement, ...]
    reference_normalization: ReferenceNormalizationUnavailable
    eligibility: ProfileEligibility
    comparability_key: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.provenance, ReaderEvidenceProvenance) or not self.provenance.is_source_closed:
            raise ReporterResponseContractError("measurement profile requires source-closed Reader provenance")
        self.provenance.require_bound_subject(self.subject_id)
        comparability = validate_measurement_profile_contract(
            contract_id=self.contract_id,
            study_id=self.study_id,
            profile_id=self.profile_id,
            subject_id=self.subject_id,
            observation_policy=self.observation_policy,
            reduction=self.reduction,
            dose_grid_uM=self.dose_grid_uM,
            measurements=self.measurements,
            reference_normalization=self.reference_normalization,
            eligibility=self.eligibility,
        )
        self.provenance.require_biological_replicate_scopes(
            tuple((row.source_condition_value, row.biological_replicate_id) for row in self.measurements)
        )
        object.__setattr__(self, "comparability_key", comparability)


def validate_measurement_profile_contract(
    *,
    contract_id: object,
    study_id: object,
    profile_id: object,
    subject_id: object,
    observation_policy: object,
    reduction: Reduction,
    dose_grid_uM: tuple[float, ...],
    measurements: tuple[ConditionMeasurement, ...],
    reference_normalization: object,
    eligibility: object,
) -> str:
    """Validate canonical raw-profile content and return its comparison identity."""

    if contract_id != MEASUREMENT_PROFILE_CONTRACT_ID:
        raise ReporterResponseContractError("measurement profile contract_id changed")
    if study_id != STUDY_ID:
        raise ReporterResponseContractError("measurement profile study_id changed")
    required_text(profile_id, field_name="profile_id")
    required_text(subject_id, field_name="subject_id")
    if not isinstance(observation_policy, ReporterResponseObservationPolicy):
        raise ReporterResponseContractError("observation_policy must be ReporterResponseObservationPolicy")
    if not isinstance(reduction, (EndpointReduction, TimeWindowReduction)):
        raise ReporterResponseContractError("reduction must be endpoint or time_window")
    if not isinstance(reference_normalization, ReferenceNormalizationUnavailable):
        raise ReporterResponseContractError("measurement profile must type why reference normalization is absent")
    if not isinstance(eligibility, ProfileEligibility):
        raise ReporterResponseContractError("eligibility must be ProfileEligibility")
    dose_grid = ordered_dose_grid(dose_grid_uM)
    if dose_grid != dose_grid_uM:
        raise ReporterResponseContractError("dose_grid_uM must be stored as its canonical tuple")
    rows = measurements
    if not isinstance(rows, tuple) or not rows or not all(isinstance(row, ConditionMeasurement) for row in rows):
        raise ReporterResponseContractError("measurement profile requires typed condition measurements")
    validate_ratio_reduction_semantics(reduction, rows)
    if len({row.observation_id for row in rows}) != len(rows):
        raise ReporterResponseContractError("measurement observation_id values must be unique")
    baseline_rows = tuple(row for row in rows if row.role == "baseline")
    positive_rows = tuple(row for row in rows if row.role == "positive_control")
    dose_rows = tuple(row for row in rows if row.role == "dose")
    if not baseline_rows or not dose_rows:
        raise ReporterResponseContractError("measurement profile requires baseline and dose observations")
    observed_doses = tuple(sorted({float(row.dose_uM) for row in dose_rows if row.dose_uM is not None}))
    if observed_doses != dose_grid:
        raise ReporterResponseContractError("dose observations must cover the declared dose grid exactly")
    if reference_normalization.reason == "positive_control_not_declared" and positive_rows:
        raise ReporterResponseContractError("positive-control measurements contradict not-declared status")
    if reference_normalization.reason in {
        "positive_control_observations_missing",
        "positive_control_separation_not_positive",
    }:
        expected_id = reference_normalization.positive_control_condition_id
        if reference_normalization.reason == "positive_control_observations_missing" and positive_rows:
            raise ReporterResponseContractError("missing positive-control observations cannot be present")
        if reference_normalization.reason == "positive_control_separation_not_positive" and (
            not positive_rows or {row.condition_id for row in positive_rows} != {expected_id}
        ):
            raise ReporterResponseContractError(
                "non-positive separation status must name the observed positive-control condition"
            )
        if (
            positive_rows
            and statistics.median(row.rfp_over_od600 for row in positive_rows)
            - statistics.median(row.rfp_over_od600 for row in baseline_rows)
            > 0.0
        ):
            raise ReporterResponseContractError(
                "positive baseline separation requires the reference-normalized profile variant"
            )
    statistics_used = {row.within_acquisition_reduction_statistic for row in rows}
    if statistics_used != {observation_policy.within_acquisition_reduction_statistic}:
        raise ReporterResponseContractError("measurement reduction statistic must equal the observation policy")
    dose_unit_keys = tuple((float(row.dose_uM), row.acquisition_id, row.biological_replicate_id) for row in dose_rows)
    if len(dose_unit_keys) != len(set(dose_unit_keys)):
        raise ReporterResponseContractError(
            "duplicate dose rows for one scoped biological replicate and acquisition are not allowed"
        )
    payload = {
        "observation_policy_digest": observation_policy.digest,
        "reduction": json_value(asdict(reduction)),
        "dose_grid_uM": list(dose_grid),
        "reference_normalization_status": reference_normalization.status,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


DescriptiveReporterProfile: TypeAlias = ReporterMeasurementProfile | ReporterResponseProfile


def build_reporter_measurement_profile(
    *,
    profile_id: str,
    subject_id: str,
    raw_design_id: str | None,
    raw_assay_subject_id: str | None,
    evidence_bindings,
    observation_policy: ReporterResponseObservationPolicy,
    reduction: Reduction,
    dose_grid_uM: Iterable[float],
    measurements: Iterable[ConditionMeasurement],
    reference_normalization: ReferenceNormalizationUnavailable,
    ineligibility_reasons: Iterable[str],
) -> ReporterMeasurementProfile:
    """Build a source-closed descriptive profile without inventing a reference."""

    provenance = ReaderEvidenceProvenance._from_source_closed_bindings(
        evidence_bindings=evidence_bindings,
        subject_id=subject_id,
        raw_design_id=raw_design_id,
        raw_assay_subject_id=raw_assay_subject_id,
    )
    return ReporterMeasurementProfile(
        contract_id=MEASUREMENT_PROFILE_CONTRACT_ID,
        study_id=STUDY_ID,
        profile_id=profile_id,
        subject_id=subject_id,
        provenance=provenance,
        observation_policy=observation_policy,
        reduction=reduction,
        dose_grid_uM=tuple(dose_grid_uM),
        measurements=tuple(measurements),
        reference_normalization=reference_normalization,
        eligibility=ProfileEligibility(
            evidence_use="descriptive",
            optimization_status="ineligible",
            reasons=tuple(ineligibility_reasons),
        ),
    )


__all__ = [
    "MEASUREMENT_PROFILE_CONTRACT_ID",
    "DescriptiveReporterProfile",
    "ReferenceNormalizationUnavailable",
    "ReporterMeasurementProfile",
    "build_reporter_measurement_profile",
    "validate_measurement_profile_contract",
]
