"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/test_stage_a_strata_overview_plot.py

Stage-A strata overview plot behaviors for labels, scales, and legend placement.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import matplotlib
import pandas as pd

from dnadesign.densegen.src.viz.plotting import _build_stage_a_strata_overview_figure


def test_stage_a_strata_overview_axes_and_legend() -> None:
    matplotlib.use("Agg", force=True)
    sampling = {
        "backend": "fimo",
        "pvalue_strata": [1e-8, 1e-6, 1e-4],
        "retain_depth": 2,
        "eligible_bins": [
            {"regulator": "regA", "counts": [2, 1, 0]},
            {"regulator": "regB", "counts": [1, 0, 0]},
        ],
        "retained_bins": [
            {"regulator": "regA", "counts": [1, 1, 0]},
            {"regulator": "regB", "counts": [0, 0, 0]},
        ],
        "eligible_pvalue_hist": [
            {"regulator": "regA", "edges": [1e-8, 1e-7, 1e-6, 1e-5], "counts": [1, 2, 1]},
            {"regulator": "regB", "edges": [1e-8, 1e-7, 1e-6, 1e-5], "counts": [0, 1, 0]},
        ],
    }
    pool_df = pd.DataFrame(
        {
            "tf": ["regA", "regA", "regB"],
            "tfbs": ["AAAAAA", "AAAAAAA", "CCCCCC"],
        }
    )

    fig, ax_left, ax_right = _build_stage_a_strata_overview_figure(
        input_name="demo_input",
        pool_df=pool_df,
        sampling=sampling,
        style={},
    )

    try:
        assert ax_left.get_xscale() == "log"
        left_xlim = ax_left.get_xlim()
        assert left_xlim[0] > left_xlim[1]
        assert ax_left.get_xlabel() == "FIMO p-value"
        assert "eligible" not in ax_left.get_xlabel().lower()
        left_labels = [label.get_text() for label in ax_left.get_yticklabels() if label.get_text()]
        assert "regA" in left_labels
        assert "regB" in left_labels
        assert ax_right.get_xlim() == (0.0, 35.0)
        assert fig.legends
        assert ax_left.get_legend() is None
        assert len(ax_right.collections) > 0
        max_points = max(len(line.get_xdata()) for line in ax_right.lines)
        assert max_points > 100
    finally:
        fig.clf()
