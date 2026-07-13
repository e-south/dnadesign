"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy/reporting/plot_helpers.py

Shared plot helpers for the response metric metastudy.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..core.contracts import StressTargetView
from ..core.policies import CANONICAL_SFXI_POLICY_ID, primary_policy_ids


def focus_policy_ids(summary: pd.DataFrame, *, comparison_policy_id: str) -> list[str]:
    ids = [CANONICAL_SFXI_POLICY_ID]
    if comparison_policy_id not in set(summary["policy_id"].astype(str)):
        raise ValueError(f"comparison policy is absent from summary: {comparison_policy_id}")
    if comparison_policy_id not in ids:
        ids.append(comparison_policy_id)
    for policy_id in primary_policy_ids():
        if policy_id not in ids:
            ids.append(policy_id)
    return ids


def target_view_mask_map(target_views: tuple[StressTargetView, ...]) -> dict[str, tuple[float, float, float, float]]:
    return {target_view.id: target_view.target_mask for target_view in target_views}


def ordered_pivot(
    frame: pd.DataFrame,
    *,
    rows: tuple[str, ...],
    columns: tuple[str, ...],
) -> pd.DataFrame:
    """Apply a closed publication order to a complete pivot table."""

    row_order = [value for value in rows if value in frame.index]
    column_order = [value for value in columns if value in frame.columns]
    if set(row_order) != set(frame.index.astype(str)) or set(column_order) != set(frame.columns.astype(str)):
        raise ValueError("plot ordering contract does not cover every row and column.")
    return frame.reindex(index=row_order, columns=column_order)


def contrast_text_color(image, value: float) -> str:
    """Return legible annotation text for a matrix cell."""

    if not np.isfinite(value):
        return "#111827"
    red, green, blue, _ = image.cmap(image.norm(value))
    perceived_lightness = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    return "#111827" if perceived_lightness >= 0.52 else "#ffffff"


def require_columns(frame: pd.DataFrame, required: set[str], *, context: str) -> None:
    """Reject incomplete plot evidence rather than rendering a partial figure."""

    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{context} missing columns: {missing}")


__all__ = [
    "contrast_text_color",
    "focus_policy_ids",
    "ordered_pivot",
    "require_columns",
    "target_view_mask_map",
]
