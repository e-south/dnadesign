"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/response_window_observations/reader_record_validation.py

Validate Reader event and reduction semantics against the study projection.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from numbers import Real

import pandas as pd

from .reader_record_relations import validate_reader_record_relations
from .reader_record_structure import REDUCTION_COLUMNS, validate_reader_frame_structure


def validate_reader_response_frames(
    *,
    designs: pd.DataFrame,
    draws: pd.DataFrame,
    wells: pd.DataFrame,
    traces: pd.DataFrame,
    events: pd.DataFrame,
    primary_reduction_id: str,
    reference_design_id: str,
    source_experiment_ids: Sequence[str],
    event: Mapping[str, object],
    aggregation: Mapping[str, object],
    reductions: Sequence[Mapping[str, object]],
) -> None:
    """Reject any scientific-contract drift in the five canonical records."""

    frames = {
        "designs": designs,
        "descriptive_resampling_draws": draws,
        "wells": wells,
        "traces": traces,
        "events": events,
    }
    validate_reader_frame_structure(
        frames,
        source_experiment_ids=source_experiment_ids,
        reference_design_id=reference_design_id,
    )
    expected_event = _expected_event(event)
    observed_events = {
        (
            _text(row.event_id),
            _text(row.event_kind),
            _text(row.event_time_estimate_method),
            _text(row.declaration),
        )
        for row in events.itertuples(index=False)
    }
    if observed_events != {expected_event}:
        raise ValueError("Reader event identity disagrees with the study projection.")
    if set(designs["event_id"].astype(str)) != {expected_event[0]}:
        raise ValueError("Reader design event identity disagrees with the study projection.")

    _validate_aggregation(designs, expected=aggregation)
    expected_reductions = {_expected_reduction(item, event_id=expected_event[0]) for item in reductions}
    _validate_reduction_contract(designs, label="designs", expected=expected_reductions, include_event=True)
    _validate_reduction_contract(wells, label="wells", expected=expected_reductions, include_event=False)
    reduction_ids = {str(item[0]) for item in expected_reductions}
    validate_reader_record_relations(
        designs=designs,
        draws=draws,
        wells=wells,
        traces=traces,
        sources=source_experiment_ids,
        reductions=reduction_ids,
        expected_draws=int(aggregation["descriptive_resampling_draws"]),
    )
    primary = set(designs.loc[designs["reduction_role"].astype(str).eq("primary"), "reduction_id"].astype(str))
    if primary != {primary_reduction_id}:
        raise ValueError("Reader primary reduction disagrees with the study projection.")
    if set(designs["reference_design_id"].astype(str)) != {reference_design_id}:
        raise ValueError("Reader normalization reference disagrees with the study projection.")


def _validate_aggregation(frame: pd.DataFrame, *, expected: Mapping[str, object]) -> None:
    quality = expected["quality"]
    if not isinstance(quality, Mapping):
        raise ValueError("Reader projection aggregation quality is malformed.")
    values = {
        "observation_stat": _text(expected["observation_stat"]),
        "descriptive_resampling_draws": int(expected["descriptive_resampling_draws"]),
        "descriptive_interval_mass": _number(expected["descriptive_interval_mass"]),
        "positive_floor": _number(quality["positive_floor"]),
        "allowed_max_interior_gap_h": _number(quality["allowed_max_interior_gap_h"]),
        "required_min_observations_per_state": int(quality["required_min_observations_per_state"]),
    }
    for column, expected_value in values.items():
        observed = frame[column].drop_duplicates().tolist()
        if observed != [expected_value]:
            raise ValueError(f"Reader aggregation field {column!r} disagrees with the study projection.")


def _expected_event(value: Mapping[str, object]) -> tuple[str, str, str, str]:
    return (
        _text(value["event_id"]),
        _text(value["event_kind"]),
        _text(value["estimate_method"]),
        _text(value["declaration"]),
    )


def _expected_reduction(value: Mapping[str, object], *, event_id: str) -> tuple[object, ...]:
    return (
        _text(value["id"]),
        _text(value["method"]),
        _text(value["response_basis"]),
        _text(value["role"]),
        _number(value["window_start_event_h"]),
        _number(value["window_end_event_h"]),
        event_id,
    )


def _validate_reduction_contract(
    frame: pd.DataFrame,
    *,
    label: str,
    expected: set[tuple[object, ...]],
    include_event: bool,
) -> None:
    event_ids = {str(item[6]) for item in expected}
    if len(event_ids) != 1:
        raise ValueError("Reader projection reductions do not share one event identity.")
    event_id = next(iter(event_ids))
    columns = [*REDUCTION_COLUMNS, *(["event_id"] if include_event else [])]
    observed = {
        (
            _text(row[0]),
            _text(row[1]),
            _text(row[2]),
            _text(row[3]),
            _number(row[4]),
            _number(row[5]),
            _text(row[6]) if include_event else event_id,
        )
        for row in frame.loc[:, columns].drop_duplicates().itertuples(index=False, name=None)
    }
    if observed != expected:
        raise ValueError(f"Reader {label} reduction contract disagrees with the study projection.")


def _text(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Reader contract text values must be non-empty strings.")
    return value.strip()


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("Reader reduction values must be numeric.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("Reader reduction values must be finite.")
    return result


__all__ = ["validate_reader_response_frames"]
