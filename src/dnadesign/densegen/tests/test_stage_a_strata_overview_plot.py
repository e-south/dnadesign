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
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.offsetbox import AnchoredText

from dnadesign.densegen.src.viz.plotting import _build_stage_a_strata_overview_figure, _draw_tier_markers


def test_stage_a_strata_overview_axes_and_legend() -> None:
    matplotlib.use("Agg", force=True)
    sampling = {
        "backend": "fimo",
        "tier_scheme": "pct_0.1_1_9",
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
                "tier2_score": 2.0,
            },
            {
                "regulator": "regB",
                "edges": [0.0, 1.0, 2.0],
                "counts": [0, 1],
                "tier0_score": 2.0,
                "tier1_score": 1.0,
                "tier2_score": 0.5,
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
        label_texts = []
        for axis in fig.axes:
            for text in axis.texts:
                label_texts.append(text.get_text())
        assert any("regA" in text for text in label_texts)
        assert any("regB" in text for text in label_texts)
        assert ax_right.get_xlim()[1] > ax_right.get_xlim()[0]
        assert len(fig.axes) == 4
        assert len(ax_right.patches) > 0
    finally:
        fig.clf()


def test_stage_a_strata_overview_accepts_regulator_id_columns() -> None:
    matplotlib.use("Agg", force=True)
    sampling = {
        "backend": "fimo",
        "tier_scheme": "pct_0.1_1_9",
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
                "tier2_score": 1.0,
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
        "tier_scheme": "pct_0.1_1_9",
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
                "tier2_score": 0.5,
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


def test_stage_a_strata_overview_xlims_cover_all_regulators() -> None:
    matplotlib.use("Agg", force=True)
    sampling = {
        "backend": "fimo",
        "tier_scheme": "pct_0.1_1_9",
        "eligibility_rule": "best_hit_score > 0 (and has at least one FIMO hit)",
        "retention_rule": "top_n_sites_by_best_hit_score",
        "fimo_thresh": 1.0,
        "eligible_score_hist": [
            {
                "regulator": "lexA",
                "edges": [0.0, 5.0, 10.0, 15.0],
                "counts": [4, 3, 2],
                "tier0_score": 14.5,
                "tier1_score": 11.0,
                "tier2_score": 6.0,
            },
            {
                "regulator": "cpxR",
                "edges": [0.0, 2.0, 4.0, 6.0],
                "counts": [2, 1, 1],
                "tier0_score": 5.5,
                "tier1_score": 4.0,
                "tier2_score": 2.0,
            },
        ],
    }
    pool_df = pd.DataFrame(
        {
            "tf": ["lexA", "lexA", "cpxR"],
            "tfbs": ["AAAAAA", "AAAAAAA", "CCCCCC"],
            "best_hit_score": [14.0, 9.0, 5.0],
        }
    )

    fig, ax_left, _ = _build_stage_a_strata_overview_figure(
        input_name="demo_input",
        pool_df=pool_df,
        sampling=sampling,
        style={},
    )

    try:
        xlim = ax_left.get_xlim()
        assert xlim[1] >= 15.0
        dashed = [line for line in ax_left.lines if line.get_linestyle() == "--"]
        assert len(dashed) >= 2
    finally:
        fig.clf()


def test_draw_tier_markers_caps_height_and_adds_box() -> None:
    matplotlib.use("Agg", force=True)
    fig, ax = plt.subplots()
    ax.set_ylim(0.0, 1.0)

    _draw_tier_markers(
        ax,
        [("0.1%", 1.0, "5"), ("1%", 0.5, "12")],
        ymax_fraction=0.58,
        label_mode="box",
    )

    try:
        assert ax.lines
        for line in ax.lines:
            ydata = line.get_ydata()
            assert max(ydata) <= 0.58 + 1e-6
        assert any(isinstance(artist, AnchoredText) for artist in ax.artists)
    finally:
        fig.clf()
