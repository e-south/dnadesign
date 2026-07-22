"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy/runtime/campaign_calibration.py

Compare the retrospective screen calibration with the campaign contract.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from ..core.contracts import EXPECTED_STRESS_TARGET_VIEW_IDS

_CAMPAIGN_CALIBRATION_FIELDS = {
    "response_separation": ("response_separation_min", "response_separation_scale"),
    "on_magnitude_floor": ("on_magnitude_min", "on_magnitude_scale"),
    "off_magnitude_ceiling": ("off_magnitude_max", "off_magnitude_scale"),
}


def compare_campaign_to_screen_calibration(
    calibration: pd.DataFrame,
    *,
    configured_by_view: Mapping[str, Mapping[str, float]],
    absolute_tolerance: float = 5.0e-7,
) -> dict[str, object]:
    """Report, but do not gate on, campaign-to-screen calibration differences."""

    required = {"selection_view_id", "component", "threshold", "scale", "scale_basis"}
    missing = sorted(required - set(calibration.columns))
    if missing:
        raise ValueError(f"response calibration lacks campaign parity fields: {missing}")
    observed_views = set(configured_by_view)
    expected_views = set(EXPECTED_STRESS_TARGET_VIEW_IDS)
    if observed_views != expected_views:
        raise ValueError(
            "campaign RMF calibration views must match the configured stress views: "
            f"missing={sorted(expected_views - observed_views)}, extra={sorted(observed_views - expected_views)}"
        )
    differences: list[str] = []
    max_abs_error = 0.0
    for view_id in EXPECTED_STRESS_TARGET_VIEW_IDS:
        configured = configured_by_view[view_id]
        for component, (threshold_field, scale_field) in _CAMPAIGN_CALIBRATION_FIELDS.items():
            rows = calibration.loc[
                calibration["selection_view_id"].astype(str).eq(view_id)
                & calibration["component"].astype(str).eq(component)
            ]
            if len(rows) != 1:
                differences.append(f"{view_id}.{component}: expected one screen calibration row, found {len(rows)}")
                continue
            row = rows.iloc[0]
            for kind, configured_field, observed_column in (
                ("threshold", threshold_field, "threshold"),
                ("scale", scale_field, "scale"),
            ):
                configured_value = float(configured[configured_field])
                observed_value = float(row[observed_column])
                difference = abs(configured_value - observed_value)
                max_abs_error = max(max_abs_error, difference)
                if not np.isclose(configured_value, observed_value, rtol=0.0, atol=absolute_tolerance):
                    differences.append(
                        f"{view_id}.{component}.{kind}: campaign={configured_value:.12g}, "
                        f"screen={observed_value:.12g}, abs_error={difference:.12g}"
                    )
    return {
        "comparison_role": "diagnostic_only",
        "matches_screen_calibration": not differences,
        "absolute_tolerance": float(absolute_tolerance),
        "max_abs_error": float(max_abs_error),
        "scale_basis": sorted(calibration["scale_basis"].astype(str).unique()),
        "differences": differences,
        "production_calibration_authority": "declared_study_calibration_cohort",
    }


__all__ = ["compare_campaign_to_screen_calibration"]
