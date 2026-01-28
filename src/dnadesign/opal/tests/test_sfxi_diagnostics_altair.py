# ABOUTME: Smoke tests for Altair-based SFXI diagnostics charts.
"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/tests/test_sfxi_diagnostics_altair.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import altair as alt
import polars as pl

from dnadesign.opal.src.analysis.dashboard.charts import sfxi_diagnostics_altair
from dnadesign.opal.src.analysis.dashboard.hues import HueOption


def test_factorial_effects_altair_smoke() -> None:
    df = pl.DataFrame(
        {
            "pred_y_hat": [
                [0.1, 0.2, 0.3, 0.4, 0.0, 0.0, 0.0, 0.0],
                [0.2, 0.2, 0.2, 0.4, 0.0, 0.0, 0.0, 0.0],
                [0.7, 0.1, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0],
            ],
            "opal__view__effect_scaled": [0.1, 0.4, 0.8],
            "opal__view__observed": [True, False, False],
            "opal__view__top_k": [False, True, False],
        }
    )
    chart = sfxi_diagnostics_altair.make_factorial_effects_chart(
        df,
        logic_col="pred_y_hat",
        size_col="opal__view__effect_scaled",
        label_col="opal__view__observed",
        plot_size=320,
    )
    assert isinstance(chart, alt.TopLevelMixin)


def test_support_diagnostics_altair_smoke() -> None:
    df = pl.DataFrame(
        {
            "dist": [0.1, 0.4, 0.2],
            "score": [0.5, 0.7, 0.1],
            "opal__view__observed": [True, False, False],
            "opal__view__top_k": [False, True, False],
            "opal__view__effect_scaled": [0.2, 0.5, 0.7],
        }
    )
    hue = HueOption(
        key="opal__view__effect_scaled",
        label="opal__view__effect_scaled",
        kind="numeric",
        dtype=pl.Float64,
    )
    chart = sfxi_diagnostics_altair.make_support_diagnostics_chart(
        df,
        x_col="dist",
        y_col="score",
        hue=hue,
        label_col="opal__view__observed",
        plot_size=320,
    )
    assert isinstance(chart, alt.TopLevelMixin)


def test_uncertainty_altair_smoke() -> None:
    df = pl.DataFrame(
        {
            "uncertainty": [0.01, 0.02, 0.03],
            "score": [0.2, 0.5, 0.9],
            "opal__view__observed": [True, False, False],
            "opal__view__top_k": [False, True, False],
            "opal__view__logic_fidelity": [0.6, 0.8, 0.4],
        }
    )
    hue = HueOption(
        key="opal__view__logic_fidelity",
        label="opal__view__logic_fidelity",
        kind="numeric",
        dtype=pl.Float64,
    )
    chart = sfxi_diagnostics_altair.make_uncertainty_chart(
        df,
        x_col="uncertainty",
        y_col="score",
        hue=hue,
        label_col="opal__view__observed",
        plot_size=320,
    )
    assert isinstance(chart, alt.TopLevelMixin)
