"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/reporter_response/metastudy/acquisition_projection.py

Build descriptive acquisition projections without inventing replicate identity.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from typing import Literal

from ..profile import TimeWindowReduction
from .contracts._values import MetastudyContractError, canonical_digest
from .contracts.profile import ProfileEvidence
from .evidence_projection import ProfileEvidenceProjection

ACQUISITION_PROJECTION_CONTRACT_ID = "rt_lnrna_reporter_response_acquisition_projection.v1"


@dataclass(frozen=True, slots=True)
class AcquisitionContribution:
    """One immutable acquisition-level descriptive value."""

    acquisition_id: str
    profile_id: str
    profile_digest: str
    declared_biological_replicate_ids: tuple[str, ...]
    normalized_reporter_response: float
    relative_od: float

    def __post_init__(self) -> None:
        _text(self.acquisition_id, "acquisition_id")
        _text(self.profile_id, "profile_id")
        _digest(self.profile_digest, "profile_digest")
        if self.declared_biological_replicate_ids != tuple(sorted(set(self.declared_biological_replicate_ids))):
            raise MetastudyContractError("declared biological-replicate ids must be unique and ordered")
        for replicate_id in self.declared_biological_replicate_ids:
            _text(replicate_id, "declared_biological_replicate_ids[]")
        _finite(self.normalized_reporter_response, "normalized_reporter_response")
        if _finite(self.relative_od, "relative_od") < 0.0:
            raise MetastudyContractError("relative_od must be non-negative")


@dataclass(frozen=True, slots=True)
class AcquisitionMetricProjection:
    """Median across acquisitions plus leave-one-acquisition-out estimates."""

    estimate: float
    method: Literal["median_across_acquisitions"]
    acquisition_count: int
    leave_one_acquisition_out_estimates: tuple[float, ...]

    def __post_init__(self) -> None:
        _finite(self.estimate, "acquisition metric estimate")
        if self.method != "median_across_acquisitions":
            raise MetastudyContractError("acquisition projection method changed")
        if type(self.acquisition_count) is not int or self.acquisition_count < 1:
            raise MetastudyContractError("acquisition_count must be positive")
        expected_loo_count = self.acquisition_count if self.acquisition_count >= 2 else 0
        if len(self.leave_one_acquisition_out_estimates) != expected_loo_count:
            raise MetastudyContractError("leave-one-acquisition-out estimates do not match acquisition support")
        for estimate in self.leave_one_acquisition_out_estimates:
            _finite(estimate, "leave_one_acquisition_out_estimates[]")


@dataclass(frozen=True, slots=True)
class AcquisitionCoordinate:
    """One subject, condition, and reduction summarized across acquisitions."""

    subject_id: str
    condition_role: Literal["dose"]
    dose_uM: float
    reduction_id: str
    reduction_digest: str
    observation_policy_digest: str
    acquisition_ids: tuple[str, ...]
    contributions: tuple[AcquisitionContribution, ...]
    normalized_reporter_response: AcquisitionMetricProjection
    relative_od: AcquisitionMetricProjection

    def __post_init__(self) -> None:
        _text(self.subject_id, "subject_id")
        if self.condition_role != "dose" or _finite(self.dose_uM, "dose_uM") <= 0.0:
            raise MetastudyContractError("acquisition coordinates require one positive dose condition")
        _text(self.reduction_id, "reduction_id")
        _digest(self.reduction_digest, "reduction_digest")
        _digest(self.observation_policy_digest, "observation_policy_digest")
        if not isinstance(self.contributions, tuple) or not self.contributions:
            raise MetastudyContractError("acquisition contributions must be a non-empty tuple")
        acquisition_ids = tuple(row.acquisition_id for row in self.contributions)
        if acquisition_ids != tuple(sorted(set(acquisition_ids))) or self.acquisition_ids != acquisition_ids:
            raise MetastudyContractError("acquisition coordinates require unique ordered acquisitions")
        if any(
            metric.acquisition_count != len(acquisition_ids)
            for metric in (self.normalized_reporter_response, self.relative_od)
        ):
            raise MetastudyContractError("metric support must match acquisition contributions")


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
        start_h, end_h = _window(self.selected_reduction)
        if not isinstance(self.coordinates, tuple) or not self.coordinates:
            raise MetastudyContractError("acquisition projection requires coordinates")
        selected_id = f"window-{start_h:g}-{end_h:g}h"
        if any(row.reduction_id != selected_id for row in self.coordinates):
            raise MetastudyContractError("acquisition projection contains a non-selected reduction")
        keys = tuple(_coordinate_key(row) for row in self.coordinates)
        if keys != tuple(sorted(keys)) or len(keys) != len(set(keys)):
            raise MetastudyContractError("acquisition projection coordinates must be unique and ordered")
        object.__setattr__(
            self,
            "projection_digest",
            canonical_digest(
                {
                    "contract_id": self.contract_id,
                    "selected_reduction": self.selected_reduction,
                    "coordinates": [asdict(row) for row in self.coordinates],
                }
            ),
        )


def build_acquisition_projection(
    evidence: Iterable[ProfileEvidence | ProfileEvidenceProjection],
    *,
    selected_reduction: tuple[float, float],
) -> AcquisitionProjection:
    """Summarize the selected window across acquisitions without resampling them as replicates."""

    selected_window = _window(selected_reduction)
    grouped: dict[tuple[str, float, str, str, str], list[AcquisitionContribution]] = defaultdict(list)
    for row in evidence:
        if not isinstance(row, (ProfileEvidence, ProfileEvidenceProjection)):
            raise MetastudyContractError("acquisition projection requires typed profile evidence")
        profile = row.profile
        reduction = profile.reduction
        if not isinstance(reduction, TimeWindowReduction):
            raise MetastudyContractError("acquisition projection accepts time-window profiles only")
        if (reduction.recorded_start_time_h, reduction.recorded_end_time_h) != selected_window:
            continue
        reduction_id = f"window-{reduction.recorded_start_time_h:g}-{reduction.recorded_end_time_h:g}h"
        reduction_digest = canonical_digest(asdict(reduction))
        acquisition_id = profile.provenance.reader_experiment_id
        responses_by_dose: dict[float, list[object]] = defaultdict(list)
        for response in profile.dose_responses:
            if response.acquisition_id != acquisition_id:
                raise MetastudyContractError("profile response acquisition differs from Reader provenance")
            responses_by_dose[float(response.dose_uM)].append(response)
        for dose, responses in responses_by_dose.items():
            replicate_ids = tuple(
                sorted(
                    {
                        response.biological_replicate_id
                        for response in responses
                        if response.biological_replicate_id is not None
                    }
                )
            )
            key = (
                profile.subject_id,
                dose,
                reduction_id,
                reduction_digest,
                profile.observation_policy.digest,
            )
            grouped[key].append(
                AcquisitionContribution(
                    acquisition_id=acquisition_id,
                    profile_id=profile.profile_id,
                    profile_digest=row.audit.profile_digest,
                    declared_biological_replicate_ids=replicate_ids,
                    normalized_reporter_response=float(
                        statistics.median(response.normalized_reporter_response for response in responses)
                    ),
                    relative_od=float(statistics.median(response.relative_od for response in responses)),
                )
            )
    if not grouped:
        raise MetastudyContractError("acquisition projection requires evidence for the selected reduction")

    coordinates: list[AcquisitionCoordinate] = []
    for (subject_id, dose, reduction_id, reduction_digest, policy_digest), values in sorted(grouped.items()):
        contributions = tuple(sorted(values, key=lambda row: row.acquisition_id))
        acquisition_ids = tuple(row.acquisition_id for row in contributions)
        if len(acquisition_ids) != len(set(acquisition_ids)):
            raise MetastudyContractError(
                f"acquisition coordinate contains duplicate acquisition: {(subject_id, dose, reduction_id)!r}"
            )
        coordinates.append(
            AcquisitionCoordinate(
                subject_id=subject_id,
                condition_role="dose",
                dose_uM=dose,
                reduction_id=reduction_id,
                reduction_digest=reduction_digest,
                observation_policy_digest=policy_digest,
                acquisition_ids=acquisition_ids,
                contributions=contributions,
                normalized_reporter_response=_metric_projection(
                    tuple(row.normalized_reporter_response for row in contributions)
                ),
                relative_od=_metric_projection(tuple(row.relative_od for row in contributions)),
            )
        )
    return AcquisitionProjection(
        contract_id=ACQUISITION_PROJECTION_CONTRACT_ID,
        selected_reduction=selected_window,
        coordinates=tuple(sorted(coordinates, key=_coordinate_key)),
    )


def acquisition_projection_payload(
    projection: AcquisitionProjection,
    include_digest: bool = True,
) -> dict[str, object]:
    if not isinstance(projection, AcquisitionProjection):
        raise MetastudyContractError("acquisition projection must be typed")
    payload: dict[str, object] = {
        "contract_id": projection.contract_id,
        "selected_reduction": list(projection.selected_reduction),
        "coordinates": json.loads(json.dumps([asdict(row) for row in projection.coordinates], allow_nan=False)),
        "projection_digest": projection.projection_digest,
    }
    if not include_digest:
        payload.pop("projection_digest", None)
    return payload


def validate_acquisition_projection_payload(value: object) -> AcquisitionProjection:
    root = _object(
        value,
        {"contract_id", "selected_reduction", "coordinates", "projection_digest"},
        "acquisition projection",
    )
    coordinate_rows = root["coordinates"]
    if not isinstance(coordinate_rows, list) or not coordinate_rows:
        raise MetastudyContractError("acquisition projection coordinates must be a non-empty array")
    reduction = root["selected_reduction"]
    if not isinstance(reduction, list):
        raise MetastudyContractError("acquisition selected_reduction must be an array")
    projection = AcquisitionProjection(
        contract_id=root["contract_id"],
        selected_reduction=tuple(reduction),
        coordinates=tuple(_parse_coordinate(row) for row in coordinate_rows),
    )
    if root["projection_digest"] != projection.projection_digest:
        raise MetastudyContractError("acquisition projection digest changed")
    return projection


def _parse_coordinate(value: object) -> AcquisitionCoordinate:
    fields = set(AcquisitionCoordinate.__dataclass_fields__)
    row = _object(value, fields, "acquisition coordinate")
    contributions = row["contributions"]
    if not isinstance(contributions, list):
        raise MetastudyContractError("acquisition contributions must be an array")
    return AcquisitionCoordinate(
        **{
            **{
                key: row[key]
                for key in fields - {"acquisition_ids", "contributions", "normalized_reporter_response", "relative_od"}
            },
            "acquisition_ids": tuple(row["acquisition_ids"]),
            "contributions": tuple(
                AcquisitionContribution(
                    **{
                        **_object(item, set(AcquisitionContribution.__dataclass_fields__), "acquisition contribution"),
                        "declared_biological_replicate_ids": tuple(item["declared_biological_replicate_ids"]),
                    }
                )
                for item in contributions
            ),
            "normalized_reporter_response": _parse_metric(row["normalized_reporter_response"]),
            "relative_od": _parse_metric(row["relative_od"]),
        }
    )


def _parse_metric(value: object) -> AcquisitionMetricProjection:
    payload = _object(value, set(AcquisitionMetricProjection.__dataclass_fields__), "acquisition metric")
    payload["leave_one_acquisition_out_estimates"] = tuple(payload["leave_one_acquisition_out_estimates"])
    return AcquisitionMetricProjection(**payload)


def _metric_projection(values: tuple[float, ...]) -> AcquisitionMetricProjection:
    estimate = float(statistics.median(values))
    loo = (
        tuple(float(statistics.median(values[:index] + values[index + 1 :])) for index in range(len(values)))
        if len(values) >= 2
        else ()
    )
    return AcquisitionMetricProjection(
        estimate=estimate,
        method="median_across_acquisitions",
        acquisition_count=len(values),
        leave_one_acquisition_out_estimates=loo,
    )


def _coordinate_key(row: AcquisitionCoordinate) -> tuple[str, float, str, str, str]:
    return (row.subject_id, row.dose_uM, row.reduction_id, row.reduction_digest, row.observation_policy_digest)


def _object(value: object, fields: set[str], label: str) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise MetastudyContractError(f"{label} fields do not match the exact contract")
    return dict(value)


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise MetastudyContractError(f"{label} must be trimmed non-empty text")
    return value


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 71 or not value.startswith("sha256:"):
        raise MetastudyContractError(f"{label} must be a canonical SHA-256 digest")
    try:
        int(value[7:], 16)
    except ValueError as exc:
        raise MetastudyContractError(f"{label} must be a canonical SHA-256 digest") from exc
    return value


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise MetastudyContractError(f"{label} must be finite")
    return float(value)


def _window(value: object) -> tuple[float, float]:
    if not isinstance(value, tuple) or len(value) != 2:
        raise MetastudyContractError("acquisition selected_reduction must contain two values")
    start_h = _finite(value[0], "selected_reduction start")
    end_h = _finite(value[1], "selected_reduction end")
    if start_h < 0.0 or end_h <= start_h:
        raise MetastudyContractError("selected_reduction must be an ordered non-negative window")
    return (start_h, end_h)


__all__ = [
    "ACQUISITION_PROJECTION_CONTRACT_ID",
    "AcquisitionContribution",
    "AcquisitionCoordinate",
    "AcquisitionMetricProjection",
    "AcquisitionProjection",
    "acquisition_projection_payload",
    "build_acquisition_projection",
    "validate_acquisition_projection_payload",
]
