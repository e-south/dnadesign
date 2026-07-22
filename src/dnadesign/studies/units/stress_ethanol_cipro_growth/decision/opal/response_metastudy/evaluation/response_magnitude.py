"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy/evaluation/response_magnitude.py

Study target margins over Reader-owned response-window summaries.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from dnadesign.opal import response_magnitude_feasibility_components

from ..core.contracts import STRESS_STATE_IDS, StressTargetView

_RAW_COLUMNS = tuple(f"r{state}" for state in STRESS_STATE_IDS) + tuple(f"b{state}" for state in STRESS_STATE_IDS)
RESPONSE_SEMANTICS = "global_target_state_separation"


def build_response_separation_rows(
    summaries: pd.DataFrame,
    *,
    target_views: Sequence[StressTargetView],
) -> pd.DataFrame:
    """Expose signed right-and-bright components without scalar compensation."""

    required = {"id", "design_id", "reduction_id", *_RAW_COLUMNS}
    missing = sorted(required - set(summaries.columns))
    if missing:
        raise ValueError(f"response-window summaries missing RMF input columns: {missing}")
    rows: list[dict[str, object]] = []
    for reduction_id, frame in summaries.groupby("reduction_id", sort=True):
        response_magnitude = frame.loc[:, list(_RAW_COLUMNS)].to_numpy(dtype=float)
        for target_view in target_views:
            components = response_magnitude_feasibility_components(
                response_magnitude,
                target_mask=target_view.target_mask,
            )
            for index, (_, source) in enumerate(frame.iterrows()):
                rows.append(
                    {
                        "id": str(source["id"]),
                        "design_id": str(source["design_id"]),
                        "reader_experiment_id": str(source["reader_experiment_id"]),
                        "reduction_id": str(reduction_id),
                        "selection_view_id": target_view.id,
                        "response_semantics": RESPONSE_SEMANTICS,
                        **{column: float(source[column]) for column in _RAW_COLUMNS},
                        "response_separation": float(components.response_separation[index]),
                        "on_magnitude_floor": float(components.on_magnitude_floor[index]),
                        "off_magnitude_ceiling": float(components.off_magnitude_ceiling[index]),
                        "passes_response_zero": bool(components.response_separation[index] >= 0.0),
                        "passes_on_magnitude_floor_zero": bool(components.on_magnitude_floor[index] >= 0.0),
                        "passes_off_magnitude_ceiling_zero": bool(components.off_magnitude_ceiling[index] <= 0.0),
                        "passes_all_zero_constraints": bool(
                            components.response_separation[index] >= 0.0
                            and components.on_magnitude_floor[index] >= 0.0
                            and components.off_magnitude_ceiling[index] <= 0.0
                        ),
                    }
                )
    return pd.DataFrame.from_records(rows)


def summarize_response_separation_stability(
    margin_rows: pd.DataFrame,
    *,
    primary_reduction_id: str,
) -> pd.DataFrame:
    """Compare signed metric components across prespecified reductions."""

    required = {
        "id",
        "reduction_id",
        "selection_view_id",
        "response_separation",
        "on_magnitude_floor",
        "off_magnitude_ceiling",
    }
    missing = sorted(required - set(margin_rows.columns))
    if missing:
        raise ValueError(f"RMF stability input missing columns: {missing}")
    rows: list[dict[str, object]] = []
    for selection_view_id, view_rows in margin_rows.groupby("selection_view_id", sort=True):
        baseline = view_rows.loc[view_rows["reduction_id"].astype(str).eq(primary_reduction_id)].set_index("id")
        if baseline.empty:
            raise ValueError(f"RMF rows lack primary reduction {primary_reduction_id!r}.")
        for reduction_id, frame in view_rows.groupby("reduction_id", sort=True):
            aligned = frame.set_index("id").loc[baseline.index]
            row: dict[str, object] = {
                "reduction_id": str(reduction_id),
                "selection_view_id": str(selection_view_id),
                "n": int(len(aligned)),
                "positive_response_count": int((aligned["response_separation"] > 0.0).sum()),
                "zero_constraint_feasible_count": int(aligned["passes_all_zero_constraints"].astype(bool).sum()),
                "median_response_separation": float(aligned["response_separation"].median()),
                "median_on_magnitude_floor": float(aligned["on_magnitude_floor"].median()),
                "median_off_magnitude_ceiling": float(aligned["off_magnitude_ceiling"].median()),
            }
            for component in ("response_separation", "on_magnitude_floor", "off_magnitude_ceiling"):
                row[f"{component}__spearman_to_primary"] = _rank_correlation(baseline[component], aligned[component])
            rows.append(row)
    return pd.DataFrame.from_records(rows)


def _rank_correlation(left: pd.Series, right: pd.Series) -> float:
    left_values = left.to_numpy(dtype=float)
    right_values = right.to_numpy(dtype=float)
    if np.ptp(left_values) == 0.0 or np.ptp(right_values) == 0.0:
        return float("nan")
    return float(spearmanr(left_values, right_values).statistic)


__all__ = [
    "RESPONSE_SEMANTICS",
    "build_response_separation_rows",
    "summarize_response_separation_stability",
]
