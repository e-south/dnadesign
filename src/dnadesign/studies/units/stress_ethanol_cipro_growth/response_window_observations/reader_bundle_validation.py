"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/response_window_observations/reader_bundle_validation.py

Validate public Reader response-window record frames.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import pandas as pd

_STATE_ORDER = ("00", "10", "01", "11")
_VALUE_COLUMNS = tuple(f"r{state}" for state in _STATE_ORDER) + tuple(f"b{state}" for state in _STATE_ORDER)
_DESIGN_BOUND_COLUMNS = {
    f"{prefix}{state}_{suffix}"
    for prefix in ("r", "b")
    for state in _STATE_ORDER
    for suffix in ("has_policy_clipping", "has_instrument_overflow", "bound_kind")
}
_EVENT_CENSOR_COLUMNS = {
    f"{prefix}{state}_event_sensitivity_has_{cause}"
    for prefix in ("r", "b")
    for state in _STATE_ORDER
    for cause in ("policy_clipping", "instrument_overflow")
}


def validate_reader_bundle_frames(
    *,
    designs: pd.DataFrame,
    draws: pd.DataFrame,
    wells: pd.DataFrame,
    traces: pd.DataFrame,
    events: pd.DataFrame,
    payload: dict[str, object],
) -> None:
    requirements = (
        (
            "designs",
            designs,
            {
                "experiment_id",
                "design_id",
                "reduction_id",
                "reduction_role",
                "is_reference",
                *_VALUE_COLUMNS,
                *_DESIGN_BOUND_COLUMNS,
                *_EVENT_CENSOR_COLUMNS,
            },
        ),
        (
            "bootstrap_draws",
            draws,
            {"experiment_id", "design_id", "reduction_id", "draw_index", "is_reference", *_VALUE_COLUMNS},
        ),
        (
            "wells",
            wells,
            {
                "experiment_id",
                "design_id",
                "reduction_id",
                "state",
                "position",
                "response_well",
                "magnitude_well",
                "response_policy_clipped_point_count",
                "response_instrument_overflow_point_count",
                "response_bound_kind",
                "magnitude_policy_clipped_point_count",
                "magnitude_instrument_overflow_point_count",
                "magnitude_bound_kind",
                "is_reference",
            },
        ),
        (
            "traces",
            traces,
            {
                "experiment_id",
                "design_id",
                "position",
                "state",
                "time_from_event_h",
                "value",
                "value_policy_clipped",
                "value_instrument_overflow",
                "value_bound_kind",
                "signal_kind",
                "is_reference",
            },
        ),
        (
            "events",
            events,
            {
                "experiment_id",
                "event_id",
                "event_interval_start_assay_h",
                "event_interval_end_assay_h",
                "event_time_estimate_assay_h",
            },
        ),
    )
    for label, frame, required in requirements:
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"Reader {label} record lacks columns: {missing}")
    if designs.duplicated(subset=["experiment_id", "design_id", "reduction_id"]).any():
        raise ValueError("Reader design identities are not unique.")
    if draws.duplicated(subset=["experiment_id", "design_id", "reduction_id", "draw_index"]).any():
        raise ValueError("Reader bootstrap-draw identities are not unique.")
    if events["experiment_id"].duplicated().any():
        raise ValueError("Reader event identities are not unique.")
    counts = payload.get("counts")
    if not isinstance(counts, dict):
        raise ValueError("Reader bundle lacks row counts.")
    expected_counts = {
        "design_rows": len(designs),
        "bootstrap_draw_rows": len(draws),
        "well_rows": len(wells),
        "trace_rows": len(traces),
        "experiments": len(events),
    }
    for key, observed in expected_counts.items():
        if int(counts.get(key, -1)) != observed:
            raise ValueError(f"Reader bundle count mismatch for {key}: {counts.get(key)!r} != {observed}.")


__all__ = ["validate_reader_bundle_frames"]
