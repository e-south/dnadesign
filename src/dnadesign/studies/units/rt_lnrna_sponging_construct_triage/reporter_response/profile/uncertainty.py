"""Uncertainty and descriptive-eligibility contracts for response profiles."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, TypeAlias

from .._contract_values import ReporterResponseContractError
from .._contract_values import explicit_id_set as _explicit_id_set
from .._contract_values import finite_number as _finite_number
from .._contract_values import positive_integer as _positive_integer
from .._contract_values import required_text as _required_text

BiologicalReplicateReductionStatistic: TypeAlias = Literal["median", "mean"]
ResamplingUnit: TypeAlias = Literal[
    "biological_replicate",
    "paired_biological_replicate",
]
NotEstimableReason: TypeAlias = Literal[
    "biological_replicate_identity_unknown",
    "below_minimum_biological_replicates",
    "insufficient_valid_resamples",
]


@dataclass(frozen=True, slots=True)
class UncertaintyPolicy:
    """Declared biological-replicate support required before uncertainty is estimable."""

    minimum_biological_replicates: int
    biological_replicate_reduction_statistic: BiologicalReplicateReductionStatistic

    def __post_init__(self) -> None:
        minimum_units = _positive_integer(
            self.minimum_biological_replicates,
            field_name="minimum_biological_replicates",
        )
        if minimum_units < 2:
            raise ReporterResponseContractError("minimum_biological_replicates must be at least 2")
        if self.biological_replicate_reduction_statistic not in {"median", "mean"}:
            raise ReporterResponseContractError("biological_replicate_reduction_statistic must be median or mean")


@dataclass(frozen=True, slots=True)
class EstimatedMetricUncertainty:
    """Caller-supplied interval metadata for one descriptive metric."""

    estimate: float
    interval_lower: float
    interval_upper: float
    confidence_level: float
    method: str
    resampling_unit: ResamplingUnit
    draws: int
    status: Literal["estimated"] = field(default="estimated", init=False)

    def __post_init__(self) -> None:
        estimate = _finite_number(self.estimate, field_name="uncertainty.estimate")
        lower = _finite_number(self.interval_lower, field_name="uncertainty.interval_lower")
        upper = _finite_number(self.interval_upper, field_name="uncertainty.interval_upper")
        if lower > upper or not lower <= estimate <= upper:
            raise ReporterResponseContractError("uncertainty interval must contain the estimate")
        confidence = _finite_number(self.confidence_level, field_name="uncertainty.confidence_level")
        if not 0.0 < confidence < 1.0:
            raise ReporterResponseContractError("uncertainty.confidence_level must be between zero and one")
        _required_text(self.method, field_name="uncertainty.method")
        if self.resampling_unit not in {
            "biological_replicate",
            "paired_biological_replicate",
        }:
            raise ReporterResponseContractError("uncertainty.resampling_unit must be a biological replicate")
        _positive_integer(self.draws, field_name="uncertainty.draws")


@dataclass(frozen=True, slots=True)
class NotEstimableMetricUncertainty:
    """Descriptive estimate with a typed reason that uncertainty is unavailable."""

    estimate: float
    reason: NotEstimableReason
    status: Literal["not_estimable"] = field(default="not_estimable", init=False)

    def __post_init__(self) -> None:
        _finite_number(self.estimate, field_name="uncertainty.estimate")
        if self.reason not in {
            "biological_replicate_identity_unknown",
            "below_minimum_biological_replicates",
            "insufficient_valid_resamples",
        }:
            raise ReporterResponseContractError("uncertainty.reason must be a declared not-estimable reason")


MetricUncertainty: TypeAlias = EstimatedMetricUncertainty | NotEstimableMetricUncertainty


@dataclass(frozen=True, slots=True)
class DoseUncertainty:
    """Supplied estimability state for both descriptive values at one dose."""

    dose_uM: float
    biological_replicate_count: int
    normalized_reporter_response: MetricUncertainty
    relative_od: MetricUncertainty

    def __post_init__(self) -> None:
        dose = _finite_number(self.dose_uM, field_name="dose_uncertainty.dose_uM")
        if dose <= 0.0:
            raise ReporterResponseContractError("dose_uncertainty.dose_uM must be positive")
        if type(self.biological_replicate_count) is not int or self.biological_replicate_count < 0:
            raise ReporterResponseContractError("dose_uncertainty.biological_replicate_count must be non-negative")
        for name in ("normalized_reporter_response", "relative_od"):
            if not isinstance(
                getattr(self, name),
                (EstimatedMetricUncertainty, NotEstimableMetricUncertainty),
            ):
                raise ReporterResponseContractError(
                    f"dose_uncertainty.{name} must declare estimated or not_estimable uncertainty"
                )
        if self.relative_od.estimate < 0.0:
            raise ReporterResponseContractError("dose_uncertainty.relative_od estimate must be non-negative")


@dataclass(frozen=True, slots=True)
class ProfileEligibility:
    """Profiles are descriptive evidence and not optimization objectives."""

    evidence_use: Literal["descriptive"]
    optimization_status: Literal["ineligible"]
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.evidence_use != "descriptive":
            raise ReporterResponseContractError("evidence_use must equal descriptive")
        if self.optimization_status != "ineligible":
            raise ReporterResponseContractError("optimization_status must equal ineligible")
        _explicit_id_set(self.reasons, field_name="eligibility.reasons")
