"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy/evaluation/metric_comparison.py

Comparable observed rows for canonical SFXI and response constraints.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from ..core.response_contracts import STRESS_STATE_IDS

_RAW_COLUMNS = tuple(f"r{state}" for state in STRESS_STATE_IDS) + tuple(f"b{state}" for state in STRESS_STATE_IDS)


def build_metric_comparison_rows(
    sfxi_rows: pd.DataFrame,
    response_rows: pd.DataFrame,
    *,
    primary_reduction_id: str,
    examples: Mapping[str, str],
) -> pd.DataFrame:
    sfxi_required = {
        "id",
        "design_id",
        "assay_summary_id",
        "selection_view_id",
        "logic_fidelity",
        "effect_scaled",
        "sfxi",
    }
    response_required = {
        "id",
        "design_id",
        "reader_experiment_id",
        "reduction_id",
        "selection_view_id",
        "response_separation",
        "on_magnitude_floor",
        "off_magnitude_ceiling",
        "feasibility_margin",
        "passes_all_zero_constraints",
        *_RAW_COLUMNS,
    }
    _require_columns(sfxi_rows, sfxi_required, context="SFXI comparison rows")
    _require_columns(response_rows, response_required, context="response comparison rows")
    sfxi = sfxi_rows.loc[sfxi_rows["assay_summary_id"].astype(str).eq(primary_reduction_id)].copy()
    if sfxi.empty:
        raise ValueError(f"SFXI comparison lacks primary reduction {primary_reduction_id!r}.")
    key = ["id", "design_id", "selection_view_id"]
    if sfxi.duplicated(subset=key).any():
        raise ValueError("SFXI primary comparison rows contain duplicate identities.")
    target_views = set(sfxi["selection_view_id"].astype(str))
    response = response_rows.loc[
        response_rows["reduction_id"].astype(str).eq(primary_reduction_id)
        & response_rows["selection_view_id"].astype(str).isin(target_views)
    ].copy()
    if response.duplicated(subset=key).any():
        raise ValueError("Response primary comparison rows contain duplicate identities.")
    result = sfxi.loc[:, [*key, "logic_fidelity", "effect_scaled", "sfxi"]].merge(
        response.loc[:, [*key, *sorted(response_required - set(key) - {"reduction_id"})]],
        on=key,
        how="left",
        validate="one_to_one",
    )
    if len(result) != len(sfxi) or result["response_separation"].isna().any():
        raise ValueError("SFXI and response primary comparison universes disagree.")
    result["off_suppression"] = -result["off_magnitude_ceiling"].astype(float)
    result["example_label"] = result["design_id"].astype(str).map(dict(examples)).fillna("")
    result["is_response_example"] = result["example_label"].ne("")
    missing_examples = sorted(set(examples) - set(result.loc[result["is_response_example"], "design_id"].astype(str)))
    if missing_examples:
        raise ValueError(f"response comparison lacks configured examples: {missing_examples}.")
    return result.sort_values(["selection_view_id", "design_id"], kind="mergesort").reset_index(drop=True)


def _require_columns(frame: pd.DataFrame, required: set[str], *, context: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{context} missing columns: {missing}")


__all__ = ["build_metric_comparison_rows"]
