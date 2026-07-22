"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy/evaluation/response_examples.py

Select measured response-window examples for review plots.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd

from ..core.response_contracts import STRESS_STATE_IDS

_RAW_COLUMNS = tuple(f"r{state}" for state in STRESS_STATE_IDS) + tuple(f"b{state}" for state in STRESS_STATE_IDS)


def build_response_example_rows(
    response_rows: pd.DataFrame,
    *,
    examples: Mapping[str, str],
    selection_view_ids: Sequence[str],
) -> pd.DataFrame:
    """Select exact response rows without translating them into another metric."""

    required = {
        "id",
        "design_id",
        "reader_experiment_id",
        "selection_view_id",
        "response_separation",
        "on_magnitude_floor",
        "off_magnitude_ceiling",
        "passes_all_zero_constraints",
        *_RAW_COLUMNS,
    }
    missing = sorted(required - set(response_rows.columns))
    if missing:
        raise ValueError(f"response example rows lack fields: {missing}")
    if not examples:
        raise ValueError("response examples must declare at least one design.")
    view_ids = tuple(str(value) for value in selection_view_ids)
    if not view_ids or len(set(view_ids)) != len(view_ids):
        raise ValueError("response example selection views must be non-empty and unique.")

    rows = response_rows.loc[
        response_rows["selection_view_id"].astype(str).isin(view_ids)
        & response_rows["design_id"].astype(str).isin(examples)
    ].copy()
    rows["example_label"] = rows["design_id"].astype(str).map(dict(examples))
    rows["off_suppression"] = -rows["off_magnitude_ceiling"].astype(float)

    expected_pairs = {(str(design_id), view_id) for design_id in examples for view_id in view_ids}
    observed_pairs = set(rows.loc[:, ["design_id", "selection_view_id"]].astype(str).itertuples(index=False, name=None))
    missing_pairs = sorted(expected_pairs - observed_pairs)
    if missing_pairs:
        raise ValueError(f"response examples lack design and selection-view pairs: {missing_pairs}")
    key = ["design_id", "selection_view_id"]
    if rows.duplicated(subset=key).any():
        raise ValueError("response examples contain duplicate design and selection-view identities.")

    return rows.sort_values(["selection_view_id", "design_id", "id"], kind="mergesort").reset_index(drop=True)


__all__ = ["build_response_example_rows"]
