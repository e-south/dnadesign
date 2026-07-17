"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy/runtime/multistate_behavior_censor.py

Exact-only censor accounting for the behavior shadow cohort.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_OUTPUT_COLUMNS = (
    "candidate_id",
    "design_id",
    "reader_experiment_id",
    "reduction_id",
    "component",
    "bound_kind",
    "component_is_nonexact",
    "has_policy_clipping",
    "has_instrument_overflow",
    "event_sensitivity_has_policy_clipping",
    "event_sensitivity_has_instrument_overflow",
    "exclusion_reason",
)


def build_behavior_censor_exclusions(
    measurements: pd.DataFrame,
    *,
    primary_reduction_id: str,
    state_ids: tuple[str, ...],
) -> pd.DataFrame:
    """Account for each component of every nonexact primary evidence unit."""

    components = [f"r{state}" for state in state_ids] + [f"b{state}" for state in state_ids]
    central_boolean_fields = ("has_policy_clipping", "has_instrument_overflow")
    event_boolean_fields = (
        "event_sensitivity_has_policy_clipping",
        "event_sensitivity_has_instrument_overflow",
    )
    required = {
        "candidate_id",
        "design_id",
        "reader_experiment_id",
        "reduction_id",
        *(f"{component}_bound_kind" for component in components),
        *(
            f"{component}_{field}"
            for component in components
            for field in (*central_boolean_fields, *event_boolean_fields)
        ),
    }
    if missing := sorted(required - set(measurements.columns)):
        raise ValueError(f"behavior censor review lacks fields: {missing}")
    primary = measurements.loc[measurements["reduction_id"].astype(str).eq(primary_reduction_id)].copy()
    bound_columns = [f"{component}_bound_kind" for component in components]
    nonexact = primary.loc[~primary[bound_columns].astype(str).eq("exact").all(axis=1)]
    records: list[dict[str, object]] = []
    for row in nonexact.itertuples(index=False):
        for component in components:
            bound_kind = str(getattr(row, f"{component}_bound_kind"))
            if bound_kind not in {"exact", "lower", "upper", "indeterminate"}:
                raise ValueError(f"behavior censor component {component!r} has unsupported bound kind {bound_kind!r}.")
            central_flags = {
                field: _exact_bool(getattr(row, f"{component}_{field}"), field=field)
                for field in central_boolean_fields
            }
            component_is_nonexact = bound_kind != "exact"
            has_censor_flag = central_flags["has_policy_clipping"] or central_flags["has_instrument_overflow"]
            if has_censor_flag != component_is_nonexact:
                raise ValueError(f"behavior censor component {component!r} bound kind disagrees with censor flags.")
            records.append(
                {
                    "candidate_id": str(row.candidate_id),
                    "design_id": str(row.design_id),
                    "reader_experiment_id": str(row.reader_experiment_id),
                    "reduction_id": primary_reduction_id,
                    "component": component,
                    "bound_kind": bound_kind,
                    "component_is_nonexact": component_is_nonexact,
                    **central_flags,
                    **{
                        field: _exact_bool(getattr(row, f"{component}_{field}"), field=field)
                        for field in event_boolean_fields
                    },
                    "exclusion_reason": ("one_or_more_nonexact_components_excluded_from_exact_shadow_cohort"),
                }
            )
    return pd.DataFrame.from_records(records, columns=_OUTPUT_COLUMNS)


def _exact_bool(value: object, *, field: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"behavior censor field {field!r} must contain exact boolean values.")
    return bool(value)


__all__ = ["build_behavior_censor_exclusions"]
