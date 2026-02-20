"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/tests/test_sfxi_diagnostics_guidance.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import polars as pl

from dnadesign.opal.src.analysis.dashboard.charts import diagnostics_guidance as diag


def test_support_and_uncertainty_ignore_non_pred_rows() -> None:
    df_pred = pl.DataFrame(
        {
            "id": ["a", "b"],
            "pred_y_hat": [
                [0.2, 0.3, 0.4, 0.1, 0.0, 0.1, 0.2, 0.3],
                [0.4, 0.2, 0.1, 0.3, 0.3, 0.2, 0.1, 0.0],
            ],
        }
    )
    df_view = pl.DataFrame(
        {
            "id": ["a", "b", "c"],
            "opal__view__score": [0.5, 0.6, None],
            "opal__view__effect_scaled": [0.4, 0.3, None],
            "opal__view__observed": [False, True, False],
            "opal__view__top_k": [False, True, False],
            "opal__sfxi__dist_to_labeled_logic": [0.1, 0.2, None],
            "opal__sfxi__uncertainty": [0.05, 0.07, None],
        }
    )

    panels = diag.build_diagnostics_panels(
        df_pred_selected=df_pred,
        df_view=df_view,
        support_y_col="opal__view__score",
        support_color=None,
        uncertainty_color=None,
        uncertainty_available=True,
    )

    assert panels.support.note is None
    assert panels.uncertainty.note is None


def test_uncertainty_hue_matches_x_column() -> None:
    df_pred = pl.DataFrame(
        {
            "id": ["a", "b"],
            "pred_y_hat": [
                [0.2, 0.3, 0.4, 0.1, 0.0, 0.1, 0.2, 0.3],
                [0.4, 0.2, 0.1, 0.3, 0.3, 0.2, 0.1, 0.0],
            ],
        }
    )
    df_view = pl.DataFrame(
        {
            "id": ["a", "b"],
            "opal__view__score": [0.5, 0.6],
            "opal__view__effect_scaled": [0.4, 0.3],
            "opal__view__observed": [False, True],
            "opal__view__top_k": [False, True],
            "opal__sfxi__dist_to_labeled_logic": [0.1, 0.2],
            "opal__sfxi__uncertainty": [0.05, 0.07],
        }
    )
    hue = diag.HueOption(
        key="opal__sfxi__uncertainty",
        label="opal__sfxi__uncertainty",
        kind="numeric",
        dtype=pl.Float64,
    )

    panels = diag.build_diagnostics_panels(
        df_pred_selected=df_pred,
        df_view=df_view,
        support_y_col="opal__view__score",
        support_color=None,
        uncertainty_color=hue,
        uncertainty_available=True,
    )

    assert panels.uncertainty.note is None
