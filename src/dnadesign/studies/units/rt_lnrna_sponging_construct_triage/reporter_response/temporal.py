"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/reporter_response/temporal.py

Study-owned neutral temporal-policy projection and comparison receipts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Literal, TypeAlias

from ._contract_values import ReporterResponseContractError

_BOUNDARY_ATOL_H = 1e-9


@dataclass(frozen=True, slots=True)
class EndpointTemporalSelection:
    """Reader-compatible exact endpoint selection."""

    time_h: float
    kind: Literal["endpoint"] = field(default="endpoint", init=False)
    time_basis: Literal["absolute"] = field(default="absolute", init=False)
    mode: Literal["exact"] = field(default="exact", init=False)
    tolerance_h: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.time_h)) or self.time_h < 0.0:
            raise ReporterResponseContractError("endpoint temporal selection time_h must be non-negative and finite")


@dataclass(frozen=True, slots=True)
class IntervalTemporalSelection:
    """Reader-compatible inclusive interval selection."""

    start_h: float
    end_h: float
    kind: Literal["interval"] = field(default="interval", init=False)
    time_basis: Literal["absolute"] = field(default="absolute", init=False)
    boundary: Literal["inclusive"] = field(default="inclusive", init=False)

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.start_h)) or not math.isfinite(float(self.end_h)):
            raise ReporterResponseContractError("interval temporal selection bounds must be finite")
        if self.start_h < 0.0 or self.start_h >= self.end_h:
            raise ReporterResponseContractError("interval temporal selection requires 0 <= start_h < end_h")


TemporalSelection: TypeAlias = EndpointTemporalSelection | IntervalTemporalSelection


@dataclass(frozen=True, slots=True)
class TemporalSupportProjection:
    """Reader-compatible support requirements, copied without importing Reader."""

    boundary_support: Literal["none", "observed"]
    minimum_observations: int
    maximum_interior_gap_h: float | None
    positive_floor: float | None
    positive_value_scope: Literal["selected_support", "entire_trace"]
    censored_values: Literal["allow", "reject"]

    def __post_init__(self) -> None:
        if self.boundary_support not in {"none", "observed"}:
            raise ReporterResponseContractError("temporal boundary_support must be none or observed")
        if (
            isinstance(self.minimum_observations, bool)
            or not isinstance(self.minimum_observations, int)
            or self.minimum_observations < 1
        ):
            raise ReporterResponseContractError("temporal minimum_observations must be a positive integer")
        for name in ("maximum_interior_gap_h", "positive_floor"):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(float(value)) or float(value) <= 0.0):
                raise ReporterResponseContractError(f"temporal {name} must be null or positive and finite")
        if self.positive_value_scope not in {"selected_support", "entire_trace"}:
            raise ReporterResponseContractError("temporal positive_value_scope is invalid")
        if self.censored_values not in {"allow", "reject"}:
            raise ReporterResponseContractError("temporal censored_values must be allow or reject")


@dataclass(frozen=True, slots=True)
class TemporalPolicyProjection:
    """Complete Reader-compatible neutral temporal operator embedded in a profile."""

    selection: TemporalSelection
    method: Literal["identity", "observed_median"]
    output_space: Literal["linear"]
    support: TemporalSupportProjection
    digest: str = field(default="", init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.selection, (EndpointTemporalSelection, IntervalTemporalSelection)):
            raise ReporterResponseContractError("temporal selection must be a typed endpoint or interval")
        if self.method not in {"identity", "observed_median"}:
            raise ReporterResponseContractError("temporal method must be identity or observed_median")
        if self.output_space != "linear":
            raise ReporterResponseContractError("temporal output_space must equal linear")
        if not isinstance(self.support, TemporalSupportProjection):
            raise ReporterResponseContractError("temporal support must be a typed projection")
        if self.method == "identity":
            if not isinstance(self.selection, EndpointTemporalSelection):
                raise ReporterResponseContractError("identity temporal reduction requires endpoint selection")
            if (
                self.support.boundary_support != "none"
                or self.support.minimum_observations != 1
                or self.support.maximum_interior_gap_h is not None
            ):
                raise ReporterResponseContractError(
                    "identity temporal reduction requires no interval support and one observation"
                )
        else:
            if not isinstance(self.selection, IntervalTemporalSelection):
                raise ReporterResponseContractError("observed temporal reduction requires interval selection")
            if self.support.boundary_support != "observed" or self.support.maximum_interior_gap_h is None:
                raise ReporterResponseContractError("observed temporal reduction requires observed interval support")
        object.__setattr__(self, "digest", _digest_payload(self.to_reader_mapping()))

    def to_reader_mapping(self) -> dict[str, object]:
        """Return the exact mapping consumed by Reader TemporalReductionSpec."""

        return {
            "selection": asdict(self.selection),
            "method": self.method,
            "output_space": self.output_space,
            "support": asdict(self.support),
        }

    @classmethod
    def from_reader_mapping(cls, value: object) -> TemporalPolicyProjection:
        """Strictly parse and canonicalize a Reader TemporalReductionSpec mapping."""

        payload = _exact_mapping(
            value,
            name="temporal operator",
            fields={"selection", "method", "output_space", "support"},
        )
        selection_payload = _mapping(payload["selection"], name="temporal selection")
        kind = selection_payload.get("kind")
        if kind == "endpoint":
            selection_values = _exact_mapping(
                selection_payload,
                name="endpoint temporal selection",
                fields={"kind", "time_basis", "time_h", "mode", "tolerance_h"},
            )
            if (
                selection_values["time_basis"] != "absolute"
                or selection_values["mode"] != "exact"
                or selection_values["tolerance_h"] != 0.0
            ):
                raise ReporterResponseContractError("endpoint temporal selection is not the neutral exact operator")
            selection: TemporalSelection = EndpointTemporalSelection(time_h=selection_values["time_h"])
        elif kind == "interval":
            selection_values = _exact_mapping(
                selection_payload,
                name="interval temporal selection",
                fields={"kind", "time_basis", "start_h", "end_h", "boundary"},
            )
            if selection_values["time_basis"] != "absolute" or selection_values["boundary"] != "inclusive":
                raise ReporterResponseContractError("interval temporal selection is not absolute and inclusive")
            selection = IntervalTemporalSelection(
                start_h=selection_values["start_h"],
                end_h=selection_values["end_h"],
            )
        else:
            raise ReporterResponseContractError("temporal selection kind must be endpoint or interval")
        support_values = _exact_mapping(
            payload["support"],
            name="temporal support",
            fields={
                "boundary_support",
                "minimum_observations",
                "maximum_interior_gap_h",
                "positive_floor",
                "positive_value_scope",
                "censored_values",
            },
        )
        projection = cls(
            selection=selection,
            method=payload["method"],
            output_space=payload["output_space"],
            support=TemporalSupportProjection(**support_values),
        )
        if projection.to_reader_mapping() != payload:
            raise ReporterResponseContractError("temporal operator does not round-trip canonically")
        return projection


@dataclass(frozen=True, slots=True)
class TemporalSelectedRow:
    """One canonical metric trace row used to derive a temporal outcome."""

    observation_identity: str
    time_h: float
    value: float
    value_policy_clipped: bool = False
    value_instrument_overflow: bool = False
    value_bound_kind: Literal["exact"] = "exact"

    def __post_init__(self) -> None:
        if not isinstance(self.observation_identity, str) or not self.observation_identity.strip():
            raise ReporterResponseContractError("temporal row observation_identity must be non-empty text")
        if not math.isfinite(float(self.time_h)) or not math.isfinite(float(self.value)):
            raise ReporterResponseContractError("temporal row time_h and value must be finite")
        if type(self.value_policy_clipped) is not bool or type(self.value_instrument_overflow) is not bool:
            raise ReporterResponseContractError("temporal row quality flags must be boolean")
        if self.value_bound_kind != "exact":
            raise ReporterResponseContractError("temporal row value_bound_kind must equal exact")


def endpoint_temporal_policy_projection(*, time_h: float) -> TemporalPolicyProjection:
    projection = TemporalPolicyProjection(
        selection=EndpointTemporalSelection(time_h=time_h),
        method="identity",
        output_space="linear",
        support=TemporalSupportProjection(
            boundary_support="none",
            minimum_observations=1,
            maximum_interior_gap_h=None,
            positive_floor=None,
            positive_value_scope="selected_support",
            censored_values="reject",
        ),
    )
    return TemporalPolicyProjection.from_reader_mapping(projection.to_reader_mapping())


def window_temporal_policy_projection(
    *,
    start_h: float,
    end_h: float,
    expected_cadence_h: float,
) -> TemporalPolicyProjection:
    if not math.isfinite(float(expected_cadence_h)) or expected_cadence_h <= 0.0:
        raise ReporterResponseContractError("expected_cadence_h must be positive and finite")
    selection = IntervalTemporalSelection(start_h=start_h, end_h=end_h)
    interval_count = (selection.end_h - selection.start_h) / expected_cadence_h
    nearest_interval_count = round(interval_count)
    if math.isclose(
        interval_count,
        nearest_interval_count,
        rel_tol=0.0,
        abs_tol=_BOUNDARY_ATOL_H / expected_cadence_h,
    ):
        interval_count = nearest_interval_count
    projection = TemporalPolicyProjection(
        selection=selection,
        method="observed_median",
        output_space="linear",
        support=TemporalSupportProjection(
            boundary_support="observed",
            minimum_observations=math.ceil(interval_count) + 1,
            maximum_interior_gap_h=expected_cadence_h,
            positive_floor=None,
            positive_value_scope="selected_support",
            censored_values="reject",
        ),
    )
    return TemporalPolicyProjection.from_reader_mapping(projection.to_reader_mapping())


def reduce_temporal_input_trace(
    rows: tuple[TemporalSelectedRow, ...],
    *,
    policy: TemporalPolicyProjection,
    within_acquisition_statistic: Literal["median"],
) -> float:
    if within_acquisition_statistic != "median":
        raise ReporterResponseContractError("temporal comparison supports only median within-acquisition reduction")
    if policy.support.censored_values == "reject" and any(
        row.value_policy_clipped or row.value_instrument_overflow or row.value_bound_kind != "exact" for row in rows
    ):
        raise ReporterResponseContractError("temporal comparison rejects censored input trace rows")
    by_observation: dict[str, list[TemporalSelectedRow]] = defaultdict(list)
    for row in rows:
        by_observation[row.observation_identity].append(row)
    observation_outputs: list[float] = []
    for observation_identity, trace in sorted(by_observation.items()):
        trace.sort(key=lambda row: row.time_h)
        if len({row.time_h for row in trace}) != len(trace):
            raise ReporterResponseContractError(
                f"temporal trace {observation_identity} contains duplicate time coordinates"
            )
        selected = _select_trace_rows(trace, policy=policy)
        if len(selected) < policy.support.minimum_observations:
            raise ReporterResponseContractError(f"temporal trace {observation_identity} has insufficient observations")
        if isinstance(policy.selection, IntervalTemporalSelection):
            if policy.support.boundary_support == "observed" and (
                not _time_equal(selected[0].time_h, policy.selection.start_h)
                or not _time_equal(selected[-1].time_h, policy.selection.end_h)
            ):
                raise ReporterResponseContractError(
                    f"temporal trace {observation_identity} does not observe both boundaries"
                )
            maximum_gap = policy.support.maximum_interior_gap_h
            assert maximum_gap is not None
            if any(
                right.time_h - left.time_h > maximum_gap + _BOUNDARY_ATOL_H
                for left, right in zip(selected, selected[1:], strict=False)
            ):
                raise ReporterResponseContractError(
                    f"temporal trace {observation_identity} exceeds maximum interior gap"
                )
        positive_rows = trace if policy.support.positive_value_scope == "entire_trace" else selected
        if policy.support.positive_floor is not None and any(
            row.value <= policy.support.positive_floor for row in positive_rows
        ):
            raise ReporterResponseContractError(f"temporal trace {observation_identity} violates positive floor")
        values = [row.value for row in selected]
        observation_outputs.append(values[0] if policy.method == "identity" else float(statistics.median(values)))
    if not observation_outputs:
        raise ReporterResponseContractError("temporal comparison requires at least one within-acquisition observation")
    return float(statistics.median(observation_outputs))


def _select_trace_rows(
    trace: list[TemporalSelectedRow],
    *,
    policy: TemporalPolicyProjection,
) -> list[TemporalSelectedRow]:
    selection = policy.selection
    if isinstance(selection, EndpointTemporalSelection):
        selected = [row for row in trace if _time_equal(row.time_h, selection.time_h)]
        if len(selected) != 1:
            raise ReporterResponseContractError("exact endpoint requires one value per within-acquisition observation")
        return selected
    return [
        row
        for row in trace
        if row.time_h >= selection.start_h - _BOUNDARY_ATOL_H and row.time_h <= selection.end_h + _BOUNDARY_ATOL_H
    ]


def _time_equal(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=0.0, abs_tol=_BOUNDARY_ATOL_H)


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ReporterResponseContractError(f"{name} must be an object with string keys")
    return value


def _exact_mapping(value: object, *, name: str, fields: set[str]) -> dict[str, object]:
    payload = _mapping(value, name=name)
    if set(payload) != fields:
        raise ReporterResponseContractError(f"{name} fields must match exactly")
    return dict(payload)


def _digest_payload(payload: dict[str, object]) -> str:
    try:
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ReporterResponseContractError("temporal identity must be JSON-compatible") from exc
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


__all__ = [
    "EndpointTemporalSelection",
    "IntervalTemporalSelection",
    "TemporalPolicyProjection",
    "TemporalSelectedRow",
    "TemporalSupportProjection",
    "endpoint_temporal_policy_projection",
    "reduce_temporal_input_trace",
    "window_temporal_policy_projection",
]
