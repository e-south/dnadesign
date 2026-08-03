"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/reporter_response/metastudy/acquisition_projection/contracts.py

Typed normalized and raw acquisition-projection contracts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

from ..contracts._values import MetastudyContractError, canonical_digest
from ._values import digest, finite, text, window

ACQUISITION_PROJECTION_CONTRACT_ID = "rt_lnrna_reporter_response_acquisition_projection.v3"
MetricSpace = Literal["raw_measurement", "reference_normalized"]


@dataclass(frozen=True, slots=True)
class AcquisitionContribution:
    """One acquisition-level value in exactly one declared metric space."""

    acquisition_id: str
    profile_id: str
    profile_digest: str
    declared_biological_replicate_ids: tuple[str, ...]
    rfp: float | None
    od600: float | None
    rfp_over_od600: float | None
    normalized_reporter_response: float | None
    relative_od: float | None

    def __post_init__(self) -> None:
        text(self.acquisition_id, "acquisition_id")
        text(self.profile_id, "profile_id")
        digest(self.profile_digest, "profile_digest")
        if self.declared_biological_replicate_ids != tuple(sorted(set(self.declared_biological_replicate_ids))):
            raise MetastudyContractError("declared biological-replicate ids must be unique and ordered")
        for replicate_id in self.declared_biological_replicate_ids:
            text(replicate_id, "declared_biological_replicate_ids[]")
        raw_values = (self.rfp, self.od600, self.rfp_over_od600)
        if len({value is None for value in raw_values}) != 1:
            raise MetastudyContractError("raw acquisition metrics must be available together")
        if (self.normalized_reporter_response is None) != (self.relative_od is None):
            raise MetastudyContractError("normalized acquisition metrics must be available together")
        if (self.rfp is None) == (self.normalized_reporter_response is None):
            raise MetastudyContractError("acquisition contribution requires exactly one declared metric space")
        if self.rfp is not None:
            finite(self.rfp, "rfp")
            assert self.od600 is not None and self.rfp_over_od600 is not None
            if finite(self.od600, "od600") <= 0.0:
                raise MetastudyContractError("od600 must be positive")
            finite(self.rfp_over_od600, "rfp_over_od600")
        if self.normalized_reporter_response is not None:
            finite(self.normalized_reporter_response, "normalized_reporter_response")
        if self.relative_od is not None and finite(self.relative_od, "relative_od") < 0.0:
            raise MetastudyContractError("relative_od must be non-negative")


@dataclass(frozen=True, slots=True)
class AcquisitionMetricProjection:
    """Median across acquisitions plus leave-one-acquisition-out estimates."""

    estimate: float
    method: Literal["median_across_acquisitions"]
    acquisition_count: int
    leave_one_acquisition_out_estimates: tuple[float, ...]

    def __post_init__(self) -> None:
        finite(self.estimate, "acquisition metric estimate")
        if self.method != "median_across_acquisitions":
            raise MetastudyContractError("acquisition projection method changed")
        if type(self.acquisition_count) is not int or self.acquisition_count < 1:
            raise MetastudyContractError("acquisition_count must be positive")
        expected = self.acquisition_count if self.acquisition_count >= 2 else 0
        if len(self.leave_one_acquisition_out_estimates) != expected:
            raise MetastudyContractError("leave-one-acquisition-out estimates do not match acquisition support")
        for estimate in self.leave_one_acquisition_out_estimates:
            finite(estimate, "leave_one_acquisition_out_estimates[]")


@dataclass(frozen=True, slots=True)
class AcquisitionCoordinate:
    """One subject, condition, reduction, and metric space across acquisitions."""

    subject_id: str
    condition_role: Literal["dose"]
    metric_space: MetricSpace
    dose_uM: float
    reduction_id: str
    reduction_digest: str
    observation_policy_digest: str
    acquisition_ids: tuple[str, ...]
    contributions: tuple[AcquisitionContribution, ...]
    rfp: AcquisitionMetricProjection | None
    od600: AcquisitionMetricProjection | None
    rfp_over_od600: AcquisitionMetricProjection | None
    normalized_reporter_response: AcquisitionMetricProjection | None
    relative_od: AcquisitionMetricProjection | None

    def __post_init__(self) -> None:
        text(self.subject_id, "subject_id")
        if self.condition_role != "dose" or finite(self.dose_uM, "dose_uM") <= 0.0:
            raise MetastudyContractError("acquisition coordinates require one positive dose condition")
        if self.metric_space not in {"raw_measurement", "reference_normalized"}:
            raise MetastudyContractError("acquisition coordinate metric_space is undeclared")
        text(self.reduction_id, "reduction_id")
        digest(self.reduction_digest, "reduction_digest")
        digest(self.observation_policy_digest, "observation_policy_digest")
        if not isinstance(self.contributions, tuple) or not self.contributions:
            raise MetastudyContractError("acquisition contributions must be a non-empty tuple")
        acquisition_ids = tuple(row.acquisition_id for row in self.contributions)
        if acquisition_ids != tuple(sorted(set(acquisition_ids))) or self.acquisition_ids != acquisition_ids:
            raise MetastudyContractError("acquisition coordinates require unique ordered acquisitions")
        raw_metrics = (self.rfp, self.od600, self.rfp_over_od600)
        normalized_metrics = (self.normalized_reporter_response, self.relative_od)
        if len({metric is None for metric in raw_metrics}) != 1:
            raise MetastudyContractError("raw coordinate metrics must be available together")
        if (normalized_metrics[0] is None) != (normalized_metrics[1] is None):
            raise MetastudyContractError("normalized coordinate metrics must be available together")
        if (self.rfp is None) == (self.normalized_reporter_response is None):
            raise MetastudyContractError("acquisition coordinate requires exactly one declared metric space")
        expected_space = "raw_measurement" if self.rfp is not None else "reference_normalized"
        if self.metric_space != expected_space:
            raise MetastudyContractError("acquisition coordinate metrics differ from metric_space")
        metrics = tuple(metric for metric in raw_metrics + normalized_metrics if metric is not None)
        if any(metric.acquisition_count != len(acquisition_ids) for metric in metrics):
            raise MetastudyContractError("metric support must match acquisition contributions")
        available = {row.normalized_reporter_response is not None for row in self.contributions}
        if available != {self.normalized_reporter_response is not None}:
            raise MetastudyContractError("coordinate normalization availability differs from its contributions")


@dataclass(frozen=True, slots=True)
class AcquisitionProjection:
    """Selected-window descriptive projection without biological uncertainty claims."""

    contract_id: str
    selected_reduction: tuple[float, float]
    coordinates: tuple[AcquisitionCoordinate, ...]
    projection_digest: str = field(init=False)

    def __post_init__(self) -> None:
        if self.contract_id != ACQUISITION_PROJECTION_CONTRACT_ID:
            raise MetastudyContractError("acquisition projection contract_id changed")
        start_h, end_h = window(self.selected_reduction)
        if not isinstance(self.coordinates, tuple) or not self.coordinates:
            raise MetastudyContractError("acquisition projection requires coordinates")
        selected_id = f"window-{start_h:g}-{end_h:g}h"
        if any(row.reduction_id != selected_id for row in self.coordinates):
            raise MetastudyContractError("acquisition projection contains a non-selected reduction")
        keys = tuple(coordinate_key(row) for row in self.coordinates)
        if keys != tuple(sorted(keys)) or len(keys) != len(set(keys)):
            raise MetastudyContractError("acquisition projection coordinates must be unique and ordered")
        object.__setattr__(
            self,
            "projection_digest",
            canonical_digest(
                {
                    "contract_id": self.contract_id,
                    "selected_reduction": self.selected_reduction,
                    "coordinates": [coordinate_payload(row) for row in self.coordinates],
                }
            ),
        )


def coordinate_key(row: AcquisitionCoordinate) -> tuple[str, float, str, str, str, str]:
    return (
        row.subject_id,
        row.dose_uM,
        row.reduction_id,
        row.reduction_digest,
        row.observation_policy_digest,
        row.metric_space,
    )


def coordinate_payload(row: AcquisitionCoordinate) -> dict[str, object]:
    payload = {
        "subject_id": row.subject_id,
        "condition_role": row.condition_role,
        "metric_space": row.metric_space,
        "dose_uM": row.dose_uM,
        "reduction_id": row.reduction_id,
        "reduction_digest": row.reduction_digest,
        "observation_policy_digest": row.observation_policy_digest,
        "acquisition_ids": row.acquisition_ids,
        "contributions": [contribution_payload(item) for item in row.contributions],
        "normalized_reporter_response": (
            asdict(row.normalized_reporter_response) if row.normalized_reporter_response is not None else None
        ),
        "relative_od": asdict(row.relative_od) if row.relative_od is not None else None,
    }
    if row.rfp is not None:
        assert row.od600 is not None and row.rfp_over_od600 is not None
        payload.update(rfp=asdict(row.rfp), od600=asdict(row.od600), rfp_over_od600=asdict(row.rfp_over_od600))
    return payload


def contribution_payload(row: AcquisitionContribution) -> dict[str, object]:
    payload = {
        "acquisition_id": row.acquisition_id,
        "profile_id": row.profile_id,
        "profile_digest": row.profile_digest,
        "declared_biological_replicate_ids": row.declared_biological_replicate_ids,
        "normalized_reporter_response": row.normalized_reporter_response,
        "relative_od": row.relative_od,
    }
    if row.rfp is not None:
        payload.update(rfp=row.rfp, od600=row.od600, rfp_over_od600=row.rfp_over_od600)
    return payload


__all__ = [
    "ACQUISITION_PROJECTION_CONTRACT_ID",
    "AcquisitionContribution",
    "AcquisitionCoordinate",
    "AcquisitionMetricProjection",
    "AcquisitionProjection",
    "coordinate_key",
    "coordinate_payload",
]
