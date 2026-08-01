"""Temporal reductions and condition measurements for reporter-response profiles."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal, TypeAlias

from .._contract_values import ReporterResponseContractError
from .._contract_values import finite_number as _finite_number
from .._contract_values import nonnegative_number as _nonnegative_number
from .._contract_values import positive_integer as _positive_integer
from .._contract_values import required_text as _required_text
from ..temporal import (
    TemporalPolicyProjection,
    endpoint_temporal_policy_projection,
    window_temporal_policy_projection,
)

ConditionRole: TypeAlias = Literal["baseline", "positive_control", "dose"]
WithinAcquisitionReductionStatistic: TypeAlias = Literal["median"]
TimeSummaryStatistic: TypeAlias = Literal["median"]
RatioReductionOrder: TypeAlias = Literal[
    "ratio_then_reduce",
    "reduce_channels_then_ratio",
]


@dataclass(frozen=True, slots=True)
class EndpointReduction:
    """One recorded endpoint with an optional, separately estimated elapsed time."""

    recorded_time_h: float
    estimated_elapsed_time_h: float | None = None
    ratio_reduction_order: Literal["reduce_channels_then_ratio"] = "reduce_channels_then_ratio"
    expected_cadence_h: float = 1.0 / 6.0
    temporal_policy: TemporalPolicyProjection | None = None
    kind: Literal["endpoint"] = field(default="endpoint", init=False)

    def __post_init__(self) -> None:
        _nonnegative_number(self.recorded_time_h, field_name="recorded_time_h")
        if self.estimated_elapsed_time_h is not None:
            _nonnegative_number(self.estimated_elapsed_time_h, field_name="estimated_elapsed_time_h")
        if self.ratio_reduction_order != "reduce_channels_then_ratio":
            raise ReporterResponseContractError("endpoint ratio_reduction_order must equal reduce_channels_then_ratio")
        cadence = _finite_number(self.expected_cadence_h, field_name="expected_cadence_h")
        if cadence <= 0.0:
            raise ReporterResponseContractError("expected_cadence_h must be positive")
        expected = endpoint_temporal_policy_projection(time_h=self.recorded_time_h)
        if self.temporal_policy is None:
            object.__setattr__(self, "temporal_policy", expected)
        elif not isinstance(self.temporal_policy, TemporalPolicyProjection):
            raise ReporterResponseContractError("endpoint temporal_policy must be a typed projection")
        assert self.temporal_policy is not None
        if self.temporal_policy != expected:
            raise ReporterResponseContractError("endpoint temporal_policy is incompatible with endpoint reduction")


@dataclass(frozen=True, slots=True)
class TimeWindowReduction:
    """One recorded time window with explicit summary and ratio-reduction semantics."""

    recorded_start_time_h: float
    recorded_end_time_h: float
    summary_statistic: TimeSummaryStatistic
    ratio_reduction_order: RatioReductionOrder
    estimated_elapsed_start_h: float | None = None
    estimated_elapsed_end_h: float | None = None
    expected_cadence_h: float = 1.0 / 6.0
    temporal_policy: TemporalPolicyProjection | None = None
    kind: Literal["time_window"] = field(default="time_window", init=False)

    def __post_init__(self) -> None:
        recorded_start = _nonnegative_number(self.recorded_start_time_h, field_name="recorded_start_time_h")
        recorded_end = _nonnegative_number(self.recorded_end_time_h, field_name="recorded_end_time_h")
        if recorded_start >= recorded_end:
            raise ReporterResponseContractError("recorded window start must be less than its end")
        if self.summary_statistic != "median":
            raise ReporterResponseContractError("time-window summary_statistic must equal median")
        if self.ratio_reduction_order != "ratio_then_reduce":
            raise ReporterResponseContractError("time-window ratio_reduction_order must equal ratio_then_reduce")
        cadence = _finite_number(self.expected_cadence_h, field_name="expected_cadence_h")
        if cadence <= 0.0:
            raise ReporterResponseContractError("expected_cadence_h must be positive")
        estimated = (self.estimated_elapsed_start_h, self.estimated_elapsed_end_h)
        if (estimated[0] is None) != (estimated[1] is None):
            raise ReporterResponseContractError("estimated elapsed window bounds must be provided together")
        if estimated[0] is not None and estimated[1] is not None:
            estimated_start = _nonnegative_number(estimated[0], field_name="estimated_elapsed_start_h")
            estimated_end = _nonnegative_number(estimated[1], field_name="estimated_elapsed_end_h")
            if estimated_start >= estimated_end:
                raise ReporterResponseContractError("estimated elapsed window start must be less than its end")
        expected = window_temporal_policy_projection(
            start_h=recorded_start,
            end_h=recorded_end,
            expected_cadence_h=cadence,
        )
        if self.temporal_policy is None:
            object.__setattr__(self, "temporal_policy", expected)
        elif not isinstance(self.temporal_policy, TemporalPolicyProjection):
            raise ReporterResponseContractError("time-window temporal_policy must be a typed projection")
        assert self.temporal_policy is not None
        if self.temporal_policy != expected:
            raise ReporterResponseContractError("time-window temporal_policy is incompatible with window reduction")


Reduction: TypeAlias = EndpointReduction | TimeWindowReduction


@dataclass(frozen=True, slots=True)
class ConditionMeasurement:
    """One condition summary from an acquisition and optional biological replicate."""

    observation_id: str
    condition_id: str
    source_condition_value: str
    role: ConditionRole
    dose_uM: float | None
    biological_replicate_id: str | None
    acquisition_id: str
    within_acquisition_observation_count: int
    within_acquisition_reduction_statistic: WithinAcquisitionReductionStatistic
    rfp: float
    od600: float
    rfp_over_od600: float

    def __post_init__(self) -> None:
        for name in (
            "observation_id",
            "condition_id",
            "source_condition_value",
            "acquisition_id",
        ):
            _required_text(getattr(self, name), field_name=name)
        if self.biological_replicate_id is not None:
            _required_text(self.biological_replicate_id, field_name="biological_replicate_id")
        if self.role not in {"baseline", "positive_control", "dose"}:
            raise ReporterResponseContractError("role must be baseline, positive_control, or dose")
        if self.role == "dose":
            dose = _finite_number(self.dose_uM, field_name="dose_uM")
            if dose <= 0.0:
                raise ReporterResponseContractError("dose observations require a positive dose_uM")
        elif self.dose_uM is not None:
            raise ReporterResponseContractError("baseline and positive_control observations require dose_uM=null")
        _positive_integer(
            self.within_acquisition_observation_count,
            field_name="within_acquisition_observation_count",
        )
        if self.within_acquisition_reduction_statistic != "median":
            raise ReporterResponseContractError("within_acquisition_reduction_statistic must equal median")
        _finite_number(self.rfp, field_name="rfp")
        od600 = _finite_number(self.od600, field_name="od600")
        if od600 <= 0.0:
            raise ReporterResponseContractError("od600 must be positive when rfp_over_od600 is declared")
        _finite_number(self.rfp_over_od600, field_name="rfp_over_od600")


def validate_ratio_reduction_semantics(
    reduction: Reduction,
    measurements: tuple[ConditionMeasurement, ...],
) -> None:
    """Require channel-derived ratios when the declared reduction orders them that way."""

    ratio_must_follow_channel_reduction = isinstance(reduction, EndpointReduction) or (
        isinstance(reduction, TimeWindowReduction) and reduction.ratio_reduction_order == "reduce_channels_then_ratio"
    )
    if not ratio_must_follow_channel_reduction:
        return
    for row in measurements:
        expected_ratio = row.rfp / row.od600
        if not math.isclose(row.rfp_over_od600, expected_ratio, rel_tol=1e-12, abs_tol=1e-12):
            raise ReporterResponseContractError(
                f"{row.observation_id}: rfp_over_od600 must equal rfp / od600 for "
                "endpoint or reduce_channels_then_ratio summaries"
            )
