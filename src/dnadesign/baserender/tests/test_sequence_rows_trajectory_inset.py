"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/baserender/tests/test_sequence_rows_trajectory_inset.py

Tests for sequence_rows "you are here" trajectory inset rendering behavior.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pytest

from dnadesign.baserender.src.config import resolve_style
from dnadesign.baserender.src.core import Display, Feature, Record, Span, TrajectoryInset
from dnadesign.baserender.src.render import Palette, render_record
from dnadesign.baserender.src.runtime import initialize_runtime


def test_sequence_rows_renders_corner_trajectory_inset_with_current_point() -> None:
    record = Record(
        id="row-1",
        alphabet="DNA",
        sequence="ACGTACGTACGT",
        features=(
            Feature(
                id="f1",
                kind="kmer",
                span=Span(start=2, end=6, strand="fwd"),
                label="GTAC",
                tags=("tf:lexA",),
            ),
        ),
        effects=(),
        display=Display(
            trajectory_inset=TrajectoryInset(
                x=(0.0, 1.0, 2.0, 3.0),
                y=(0.1, 0.2, 0.22, 0.4),
                point_index=2,
                corner="bottom_left",
                label="Best-so-far",
            )
        ),
    ).validate()
    style = resolve_style(preset="presentation_default", overrides={"show_reverse_complement": True})
    palette = Palette(style.palette)
    initialize_runtime()
    fig = render_record(record, renderer_name="sequence_rows", style=style, palette=palette)
    try:
        main_ax = fig.axes[0]
        assert len(main_ax.child_axes) == 1
        inset_ax = main_ax.child_axes[0]
        assert not any(text.get_text() == "Best-so-far" for text in inset_ax.texts)
        assert inset_ax.get_xlabel() == "Sweep"
        assert inset_ax.get_ylabel() == "Best score"
        assert inset_ax.xaxis.get_label().get_position()[1] < 0.0
        assert inset_ax.yaxis.get_label().get_position()[0] < 0.0
        assert inset_ax.get_zorder() < 10.0
        bounds = inset_ax.get_position().bounds
        assert bounds[2] == pytest.approx(0.22, abs=1.0e-6)
        assert bounds[3] == pytest.approx(0.14, abs=1.0e-6)
        assert bounds[1] == pytest.approx(0.10, abs=1.0e-6)
        assert inset_ax.spines["top"].get_visible() is False
        assert inset_ax.spines["right"].get_visible() is False
        assert inset_ax.collections
    finally:
        plt.close(fig)


def test_sequence_rows_honors_fixed_content_radius_meta_for_stable_canvas_height() -> None:
    style = resolve_style(preset="presentation_default", overrides={"show_reverse_complement": True})
    palette = Palette(style.palette)
    initialize_runtime()

    base_record = Record(
        id="base",
        alphabet="DNA",
        sequence="ACGTACGTACGTACGT",
        features=(
            Feature(
                id="base_fwd",
                kind="kmer",
                span=Span(start=2, end=6, strand="fwd"),
                label="GTAC",
                tags=("tf:lexA",),
            ),
        ),
        effects=(),
        display=Display(),
        meta={"fixed_content_radius_px": 500.0},
    ).validate()
    tall_record = Record(
        id="tall",
        alphabet="DNA",
        sequence="ACGTACGTACGTACGT",
        features=(
            Feature(
                id="tall_fwd_1",
                kind="kmer",
                span=Span(start=0, end=4, strand="fwd"),
                label="ACGT",
                tags=("tf:lexA",),
                render={"track": 0},
            ),
            Feature(
                id="tall_fwd_2",
                kind="kmer",
                span=Span(start=4, end=8, strand="fwd"),
                label="ACGT",
                tags=("tf:cpxR",),
                render={"track": 5},
            ),
        ),
        effects=(),
        display=Display(),
        meta={"fixed_content_radius_px": 500.0},
    ).validate()

    fig_base = render_record(base_record, renderer_name="sequence_rows", style=style, palette=palette)
    fig_tall = render_record(tall_record, renderer_name="sequence_rows", style=style, palette=palette)
    try:
        assert fig_base.get_size_inches().tolist() == fig_tall.get_size_inches().tolist()
    finally:
        plt.close(fig_base)
        plt.close(fig_tall)
