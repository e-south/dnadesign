"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/response_window_observations/reader_record_structure.py

Validate Reader event-window frame structure and source coverage.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd

STATE_ORDER = ("00", "10", "01", "11")
VALUE_COLUMNS = tuple(f"r{state}" for state in STATE_ORDER) + tuple(f"b{state}" for state in STATE_ORDER)
REDUCTION_COLUMNS = (
    "reduction_id",
    "reduction_method",
    "response_basis",
    "reduction_role",
    "window_start_event_h",
    "window_end_event_h",
)
_DESIGN_BOUND_COLUMNS = {
    f"{prefix}{state}_{suffix}"
    for prefix in ("r", "b")
    for state in STATE_ORDER
    for suffix in ("has_policy_clipping", "has_instrument_overflow", "bound_kind")
}
_EVENT_CENSOR_COLUMNS = {
    f"{prefix}{state}_event_sensitivity_has_{cause}"
    for prefix in ("r", "b")
    for state in STATE_ORDER
    for cause in ("policy_clipping", "instrument_overflow")
}
_REQUIREMENTS = {
    "designs": {
        "experiment_id",
        "design_id",
        "reference_design_id",
        *REDUCTION_COLUMNS,
        "event_id",
        "observation_stat",
        "descriptive_resampling_draws",
        "descriptive_interval_mass",
        "positive_floor",
        "allowed_max_interior_gap_h",
        "required_min_observations_per_state",
        "is_reference",
        *VALUE_COLUMNS,
        *_DESIGN_BOUND_COLUMNS,
        *_EVENT_CENSOR_COLUMNS,
    },
    "descriptive_resampling_draws": {
        "experiment_id",
        "design_id",
        "reduction_id",
        "draw_index",
        "is_reference",
        *VALUE_COLUMNS,
    },
    "wells": {
        "experiment_id",
        "design_id",
        *REDUCTION_COLUMNS,
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
    "traces": {
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
    "events": {
        "experiment_id",
        "event_id",
        "event_kind",
        "event_interval_start_assay_h",
        "event_interval_end_assay_h",
        "event_time_estimate_assay_h",
        "event_time_estimate_method",
        "declaration",
    },
}


def validate_reader_frame_structure(
    frames: Mapping[str, pd.DataFrame],
    *,
    source_experiment_ids: Sequence[str],
    reference_design_id: str,
) -> None:
    """Require complete sources, exact identities, and honest reference flags."""

    for label, frame in frames.items():
        missing = sorted(_REQUIREMENTS[label] - set(frame.columns))
        if missing:
            raise ValueError(f"Reader {label} record lacks columns: {missing}")
        _require_exact_sources(frame, label=label, expected=source_experiment_ids)
    if frames["designs"].duplicated(subset=["experiment_id", "design_id", "reduction_id"]).any():
        raise ValueError("Reader design identities are not unique.")
    if (
        frames["descriptive_resampling_draws"]
        .duplicated(subset=["experiment_id", "design_id", "reduction_id", "draw_index"])
        .any()
    ):
        raise ValueError("Reader descriptive-resampling draw identities are not unique.")
    if frames["events"]["experiment_id"].duplicated().any():
        raise ValueError("Reader event identities are not unique.")
    for label in ("designs", "descriptive_resampling_draws", "wells", "traces"):
        _require_reference_identity(
            frames[label],
            label=label,
            reference_design_id=reference_design_id,
            source_experiment_ids=source_experiment_ids,
        )
    state_groups = {
        "wells": ["experiment_id", "design_id", "reduction_id"],
        "traces": ["experiment_id", "design_id", "signal_kind"],
    }
    for label, group_fields in state_groups.items():
        for identity, group in frames[label].groupby(group_fields, sort=False):
            states = set(group["state"].astype(str))
            if states != set(STATE_ORDER):
                raise ValueError(
                    f"Reader {label} states for {identity!r} must be exactly {list(STATE_ORDER)!r}; "
                    f"observed {sorted(states)!r}."
                )


def _require_exact_sources(frame: pd.DataFrame, *, label: str, expected: Sequence[str]) -> None:
    expected_set = set(expected)
    observed = set(frame["experiment_id"].astype(str))
    if observed != expected_set:
        raise ValueError(
            f"Reader {label} source experiments disagree with the study projection: "
            f"missing={sorted(expected_set - observed)!r}, unexpected={sorted(observed - expected_set)!r}."
        )


def _require_reference_identity(
    frame: pd.DataFrame,
    *,
    label: str,
    reference_design_id: str,
    source_experiment_ids: Sequence[str],
) -> None:
    reference = frame["is_reference"]
    if reference.isna().any() or not pd.api.types.is_bool_dtype(reference):
        raise ValueError(f"Reader {label} is_reference must be complete boolean data.")
    expected = frame["design_id"].astype(str).eq(reference_design_id)
    if not reference.reset_index(drop=True).equals(expected.reset_index(drop=True)):
        raise ValueError(f"Reader {label} reference flags disagree with reference_design_id.")
    observed_sources = set(frame.loc[reference, "experiment_id"].astype(str))
    if observed_sources != set(source_experiment_ids):
        raise ValueError(f"Reader {label} does not contain the projected reference in every source experiment.")


__all__ = ["REDUCTION_COLUMNS", "STATE_ORDER", "VALUE_COLUMNS", "validate_reader_frame_structure"]
