"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/stage_a/test_stage_a_strata_overview_plot.py

Stage-A strata overview plot behaviors for labels, scales, and legend placement.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import pytest
from matplotlib.colors import to_hex
from matplotlib.offsetbox import AnchoredText

from dnadesign.densegen.src.viz.plotting import _build_stage_a_strata_overview_figure, _draw_tier_markers


def _vertical_marker_lines(ax, *, linestyle: str) -> list[matplotlib.lines.Line2D]:
    lines: list[matplotlib.lines.Line2D] = []
    for line in ax.lines:
        xdata = list(line.get_xdata())
        if len(xdata) < 2:
            continue
        if float(xdata[0]) != float(xdata[-1]):
            continue
        if line.get_linestyle() != linestyle:
            continue
        if to_hex(line.get_color()).lower() != "#222222":
            continue
        lines.append(line)
    return lines


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
                "tier_fractions": [0.001, 0.01, 0.09],
                "tier_fractions_source": "default",
            },
            {
                "regulator": "regB",
                "edges": [0.0, 1.0, 2.0],
                "counts": [0, 1],
                "tier0_score": 2.0,
                "tier1_score": 1.0,
                "tier2_score": 0.5,
                "tier_fractions": [0.001, 0.01, 0.09],
                "tier_fractions_source": "default",
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

    fig, axes_left, ax_right = _build_stage_a_strata_overview_figure(
        input_name="demo_input",
        pool_df=pool_df,
        sampling=sampling,
        style={},
    )

    try:
        ax_left = axes_left[0]
        ax_left_bottom = axes_left[-1]
        assert ax_left.get_xscale() == "linear"
        left_xlim = ax_left.get_xlim()
        assert left_xlim[0] < left_xlim[1]
        assert "score" in ax_left_bottom.get_xlabel().lower()
        label_texts = []
        for axis in fig.axes:
            for text in axis.texts:
                label_texts.append(text.get_text())
        assert any("regA" in text for text in label_texts)
        assert any("regB" in text for text in label_texts)
        assert ax_right.get_xlim()[1] > ax_right.get_xlim()[0]
        assert len(fig.axes) == 3
        assert len(ax_right.patches) > 0
        assert axes_left[-1].xaxis.label.get_size() == pytest.approx(ax_right.xaxis.label.get_size())
        left_x_ticks = [tick.get_size() for tick in axes_left[-1].get_xticklabels() if tick.get_text()]
        right_x_ticks = [tick.get_size() for tick in ax_right.get_xticklabels() if tick.get_text()]
        right_y_ticks = [tick.get_size() for tick in ax_right.get_yticklabels() if tick.get_text()]
        assert left_x_ticks and right_x_ticks and right_y_ticks
        assert left_x_ticks[0] == pytest.approx(right_x_ticks[0])
        assert right_x_ticks[0] == pytest.approx(right_y_ticks[0])
        tier_boxes = [artist for artist in axes_left[0].artists if isinstance(artist, AnchoredText)]
        assert tier_boxes
        assert all(box.loc == 2 for box in tier_boxes)
        assert all(box.txt._text.get_fontsize() >= 6.6 for box in tier_boxes)
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
                "tier_fractions": [0.001, 0.01, 0.09],
                "tier_fractions_source": "default",
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
                "tier_fractions": [0.001, 0.01, 0.09],
                "tier_fractions_source": "default",
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
                "tier_fractions": [0.001, 0.01, 0.09],
                "tier_fractions_source": "default",
            },
            {
                "regulator": "cpxR",
                "edges": [0.0, 2.0, 4.0, 6.0],
                "counts": [2, 1, 1],
                "tier0_score": 5.5,
                "tier1_score": 4.0,
                "tier2_score": 2.0,
                "tier_fractions": [0.001, 0.01, 0.09],
                "tier_fractions_source": "default",
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

    fig, axes_left, _ = _build_stage_a_strata_overview_figure(
        input_name="demo_input",
        pool_df=pool_df,
        sampling=sampling,
        style={},
    )

    try:
        xlim = axes_left[0].get_xlim()
        assert xlim[1] >= 15.0
        dashed = [line for line in axes_left[0].lines if line.get_linestyle() == "--"]
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
        assert ax.collections
        assert any(isinstance(artist, AnchoredText) for artist in ax.artists)
        labels = [text.get_text() for text in ax.texts]
        assert "0.1%" in labels
        assert "1%" in labels
        box = next(artist for artist in ax.artists if isinstance(artist, AnchoredText))
        edge = box.patch.get_edgecolor()
        assert edge[-1] == 0.0
    finally:
        fig.clf()


def test_stage_a_strata_excludes_background_regulators_from_fimo_panels() -> None:
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
                "counts": [2, 1],
                "tier0_score": 4.0,
                "tier1_score": 2.0,
                "tier2_score": 1.0,
                "tier_fractions": [0.001, 0.01, 0.09],
                "tier_fractions_source": "default",
            },
            {
                "regulator": "background",
                "edges": [0.0, 1.0, 2.0],
                "counts": [1, 1],
                "tier0_score": 2.0,
                "tier1_score": 1.5,
                "tier2_score": 1.0,
                "tier_fractions": [0.001, 0.01, 0.09],
                "tier_fractions_source": "default",
            },
        ],
    }
    pool_df = pd.DataFrame(
        {
            "tf": ["regA", "regA", "background"],
            "tfbs": ["AAAAAA", "AAAAAT", "CCCCCC"],
            "best_hit_score": [3.8, 2.2, 1.1],
        }
    )

    fig, axes_left, ax_right = _build_stage_a_strata_overview_figure(
        input_name="demo_input",
        pool_df=pool_df,
        sampling=sampling,
        style={},
    )

    try:
        assert len(axes_left) == 1
        legend = ax_right.get_legend()
        assert legend is not None
        labels = [text.get_text().lower() for text in legend.texts]
        assert "background" not in labels
    finally:
        fig.clf()


def test_stage_a_strata_marks_worst_retained_percentile_with_single_solid_lollipop() -> None:
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
                "tier0_score": 5.6,
                "tier1_score": 4.5,
                "tier2_score": 2.2,
                "tier_fractions": [0.001, 0.01, 0.09],
                "tier_fractions_source": "default",
            },
            {
                "regulator": "regB",
                "edges": [0.0, 2.0, 4.0, 6.0],
                "counts": [2, 1, 1],
                "tier0_score": 5.2,
                "tier1_score": 3.8,
                "tier2_score": 1.8,
                "tier_fractions": [0.001, 0.01, 0.09],
                "tier_fractions_source": "default",
            },
        ],
    }
    pool_df = pd.DataFrame(
        {
            "tf": ["regA", "regA", "regB", "regB"],
            "tfbs": ["AAAAAA", "AAAAAT", "CCCCCC", "CCCCCG"],
            "best_hit_score": [5.4, 4.9, 1.1, 3.0],
        }
    )

    fig, axes_left, _ = _build_stage_a_strata_overview_figure(
        input_name="demo_input",
        pool_df=pool_df,
        sampling=sampling,
        style={},
    )
    try:
        dashed_a = _vertical_marker_lines(axes_left[0], linestyle="--")
        dashed_b = _vertical_marker_lines(axes_left[1], linestyle="--")
        solid_a = _vertical_marker_lines(axes_left[0], linestyle="-")
        solid_b = _vertical_marker_lines(axes_left[1], linestyle="-")
        assert len(dashed_a) >= 3
        assert len(dashed_b) >= 3
        assert len(solid_a) == 0
        assert len(solid_b) == 1
    finally:
        fig.clf()
