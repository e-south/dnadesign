"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/tests/analysis/test_sfxi_diagnostics_charts.py

Smoke tests for SFXI diagnostic chart builders.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import numpy as np
import polars as pl

from dnadesign.opal.src.analysis.dashboard.charts import (
    sfxi_intensity_scaling,
    sfxi_setpoint_sweep,
    sfxi_support_diagnostics,
    sfxi_uncertainty,
)


def test_support_diagnostics_chart_smoke() -> None:
    df = pl.DataFrame(
        {
            "dist": [0.1, 0.4, 0.2],
            "score": [0.5, 0.7, 0.1],
            "effect_scaled": [0.3, 0.9, 0.2],
        }
    )
    fig = sfxi_support_diagnostics.make_support_diagnostics_figure(
        df,
        x_col="dist",
        y_col="score",
        hue_col="effect_scaled",
    )
    assert fig is not None


def test_uncertainty_chart_smoke() -> None:
    df = pl.DataFrame(
        {
            "uncertainty": [0.01, 0.02, 0.03],
            "score": [0.2, 0.5, 0.9],
            "logic_fidelity": [0.6, 0.8, 0.4],
        }
    )
    fig = sfxi_uncertainty.make_uncertainty_figure(
        df,
        x_col="uncertainty",
        y_col="score",
        hue_col="logic_fidelity",
    )
    assert fig is not None


def test_intensity_scaling_chart_smoke() -> None:
    sweep_df = pl.DataFrame(
        {
            "setpoint_name": ["0000", "1111"],
            "denom_used": [1.0, 2.0],
            "clip_lo_fraction": [0.0, 0.1],
            "clip_hi_fraction": [0.2, 0.3],
        }
    )
    label_effect_raw = np.array([0.1, 0.4, 0.8], dtype=float)
    fig = sfxi_intensity_scaling.make_intensity_scaling_figure(
        sweep_df,
        label_effect_raw=label_effect_raw,
        pool_effect_raw=None,
    )
    assert fig is not None


def test_setpoint_sweep_heatmap_smoke() -> None:
    df = pl.DataFrame(
        {
            "setpoint_name": ["0001", "1111"],
            "setpoint_label": ["[0, 0, 0, 1]", "[1, 1, 1, 1]"],
            "logic_fidelity": [0.5, 0.9],
            "effect_scaled": [0.2, 0.8],
            "score": [0.1, 0.7],
        }
    )
    fig = sfxi_setpoint_sweep.make_setpoint_sweep_figure(
        df,
        metrics=["logic_fidelity", "effect_scaled", "score"],
    )
    assert fig is not None
