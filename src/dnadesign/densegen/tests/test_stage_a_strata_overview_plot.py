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
        "tier_scheme": "pct_1_9_90",
        "eligibility_rule": "best_hit_score > 0 (and has at least one FIMO hit)",
        "retention_rule": "top_n_sites_by_best_hit_score",
        "fimo_thresh": 1.0,
        "eligible_score_hist": [
            {
                "regulator": "regA",
                "edges": [0.0, 2.0, 4.0, 6.0],
                "counts": [1, 2, 1],
                "tier0_score": 6.0,
                "tier1_score": 4.0,
            },
            {
                "regulator": "regB",
                "edges": [0.0, 1.0, 2.0],
                "counts": [0, 1],
                "tier0_score": 2.0,
                "tier1_score": 1.0,
            },
        ],
    }
    pool_df = pd.DataFrame(
        {
            "tf": ["regA", "regA", "regB"],
            "tfbs": ["AAAAAA", "AAAAAAA", "CCCCCC"],
            "best_hit_score": [5.0, 3.0, 1.5],
        }
    )

    fig, ax_left, ax_right = _build_stage_a_strata_overview_figure(
        input_name="demo_input",
        pool_df=pool_df,
        sampling=sampling,
        style={},
    )

    try:
        assert ax_left.get_xscale() == "linear"
        left_xlim = ax_left.get_xlim()
        assert left_xlim[0] < left_xlim[1]
        assert "score" in ax_left.get_xlabel().lower()
        left_labels = [label.get_text() for label in ax_left.get_yticklabels() if label.get_text()]
        assert "regA" in left_labels
        assert "regB" in left_labels
        assert ax_right.get_xlim()[1] > ax_right.get_xlim()[0]
        assert fig.legends
        assert ax_left.get_legend() is None
        assert len(ax_right.collections) > 0
        max_points = max(len(line.get_xdata()) for line in ax_right.lines)
        assert max_points > 100
    finally:
        fig.clf()


def test_stage_a_strata_overview_accepts_regulator_id_columns() -> None:
    matplotlib.use("Agg", force=True)
    sampling = {
        "backend": "fimo",
        "tier_scheme": "pct_1_9_90",
        "eligibility_rule": "best_hit_score > 0 (and has at least one FIMO hit)",
        "retention_rule": "top_n_sites_by_best_hit_score",
        "fimo_thresh": 1.0,
        "eligible_score_hist": [
            {
                "regulator": "regA",
                "edges": [0.0, 2.0, 4.0],
                "counts": [1, 1],
                "tier0_score": 4.0,
                "tier1_score": 2.0,
            }
        ],
    }
    pool_df = pd.DataFrame(
        {
            "regulator_id": ["regA"],
            "tfbs_sequence": ["AAAAAA"],
            "best_hit_score": [3.5],
        }
    )

    fig, _, _ = _build_stage_a_strata_overview_figure(
        input_name="demo_input",
        pool_df=pool_df,
        sampling=sampling,
        style={},
    )

    try:
        assert fig.axes
    finally:
        fig.clf()


def test_stage_a_strata_overview_length_axis_expands_for_long_tfbs() -> None:
    matplotlib.use("Agg", force=True)
    sampling = {
        "backend": "fimo",
        "tier_scheme": "pct_1_9_90",
        "eligibility_rule": "best_hit_score > 0 (and has at least one FIMO hit)",
        "retention_rule": "top_n_sites_by_best_hit_score",
        "fimo_thresh": 1.0,
        "eligible_score_hist": [
            {
                "regulator": "regA",
                "edges": [0.0, 1.0, 2.0],
                "counts": [1, 1],
                "tier0_score": 2.0,
                "tier1_score": 1.0,
            }
        ],
    }
    long_tfbs = "A" * 80
    pool_df = pd.DataFrame(
        {
            "tf": ["regA"],
            "tfbs": [long_tfbs],
            "best_hit_score": [1.5],
        }
    )

    fig, _, ax_right = _build_stage_a_strata_overview_figure(
        input_name="demo_input",
        pool_df=pool_df,
        sampling=sampling,
        style={},
    )

    try:
        assert ax_right.get_xlim()[1] >= len(long_tfbs)
    finally:
        fig.clf()
