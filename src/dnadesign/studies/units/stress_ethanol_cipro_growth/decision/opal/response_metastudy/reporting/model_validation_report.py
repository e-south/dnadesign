"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy/reporting/model_validation_report.py

Report-facing summaries for held-out model validation.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import pandas as pd


def model_validation_summary_table(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby(["split_strategy", "scope", "metric_id"], sort=False)
        .agg(
            median_spearman=("spearman", "median"),
            min_spearman=("spearman", "min"),
            max_spearman=("spearman", "max"),
            median_r2=("r2", "median"),
            median_mae=("mae", "median"),
        )
        .reset_index()
    )
