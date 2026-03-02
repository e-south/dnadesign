"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/baserender/tests/test_sequence_rows_trajectory_inset.py

Tests for sequence_rows trajectory panel rendering behavior.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pytest

from dnadesign.baserender.src.config import resolve_style
from dnadesign.baserender.src.core import Display, Feature, Record, Span, TrajectoryPanel
from dnadesign.baserender.src.render import Palette, render_record
from dnadesign.baserender.src.render.sequence_rows import _format_compact_axis_value
from dnadesign.baserender.src.runtime import initialize_runtime


def _bbox_overlaps(a, b) -> bool:
    return float(max(a.x0, b.x0)) < float(min(a.x1, b.x1)) and float(max(a.y0, b.y0)) < float(min(a.y1, b.y1))


def test_sequence_rows_renders_side_by_side_trajectory_panel_with_current_point() -> None:
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
            trajectory_panel=TrajectoryPanel(
                x=(0.0, 1.0, 2.0, 3.0),
                y=(0.1, 0.2, 0.22, 0.4),
                point_index=2,
            )
        ),
    ).validate()
    style = resolve_style(preset="presentation_default", overrides={"show_reverse_complement": True})
    palette = Palette(style.palette)
    initialize_runtime()
    fig = render_record(record, renderer_name="sequence_rows", style=style, palette=palette)
    try:
        assert len(fig.axes) == 2
        left_ax, right_ax = sorted(fig.axes, key=lambda axis: float(axis.get_position().x0))
        assert left_ax.get_xlabel() == "Sweep"
        assert left_ax.get_ylabel() == "Best objective"
        assert left_ax.lines
        assert left_ax.collections
        assert right_ax.axison is False
        left_bounds = left_ax.get_position().bounds
        right_bounds = right_ax.get_position().bounds
        assert float(left_bounds[0] + left_bounds[2]) < float(right_bounds[0])
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        xlabel_bbox = left_ax.xaxis.label.get_window_extent(renderer=renderer).transformed(fig.transFigure.inverted())
        ylabel_bbox = left_ax.yaxis.label.get_window_extent(renderer=renderer).transformed(fig.transFigure.inverted())
        assert float(xlabel_bbox.x0) >= 0.0
        assert float(xlabel_bbox.y0) >= 0.0
        assert float(xlabel_bbox.x1) <= 1.0
        assert float(xlabel_bbox.y1) <= 1.0
        assert float(ylabel_bbox.x0) >= 0.0
        assert float(ylabel_bbox.y0) >= 0.0
        assert float(ylabel_bbox.x1) <= 1.0
        assert float(ylabel_bbox.y1) <= 1.0
        xtick_bboxes = [
            label.get_window_extent(renderer=renderer)
            for label in left_ax.get_xticklabels()
            if str(label.get_text()).strip()
        ]
        ytick_bboxes = [
            label.get_window_extent(renderer=renderer)
            for label in left_ax.get_yticklabels()
            if str(label.get_text()).strip()
        ]
        x_label_box = left_ax.xaxis.label.get_window_extent(renderer=renderer)
        y_label_box = left_ax.yaxis.label.get_window_extent(renderer=renderer)
        assert all(not _bbox_overlaps(x_label_box, box) for box in xtick_bboxes)
        assert all(not _bbox_overlaps(y_label_box, box) for box in ytick_bboxes)
    finally:
        plt.close(fig)


def test_sequence_rows_trajectory_panel_uses_custom_axis_labels() -> None:
    record = Record(
        id="row-custom-panel-labels",
        alphabet="DNA",
        sequence="ACGTACGTACGT",
        effects=(),
        display=Display(
            trajectory_panel=TrajectoryPanel(
                x=(0.0, 10.0, 20.0, 30.0),
                y=(0.11, 0.23, 0.31, 0.35),
                point_index=3,
                x_label="Sweep",
                y_label="Best min-TF norm-LLR",
            )
        ),
    ).validate()
    style = resolve_style(preset="presentation_default", overrides={"show_reverse_complement": True})
    palette = Palette(style.palette)
    initialize_runtime()
    fig = render_record(record, renderer_name="sequence_rows", style=style, palette=palette)
    try:
        left_ax, _right_ax = sorted(fig.axes, key=lambda axis: float(axis.get_position().x0))
        assert left_ax.get_xlabel() == "Sweep"
        assert left_ax.get_ylabel() == "Best min-TF norm-LLR"
    finally:
        plt.close(fig)


def test_sequence_rows_trajectory_panel_uses_square_plot_box() -> None:
    record = Record(
        id="row-square-panel",
        alphabet="DNA",
        sequence="ACGTACGTACGTACGT",
        effects=(),
        display=Display(
            trajectory_panel=TrajectoryPanel(
                x=(0.0, 1.0, 2.0, 3.0, 4.0),
                y=(0.1, 0.15, 0.2, 0.28, 0.35),
                point_index=3,
            )
        ),
    ).validate()
    style = resolve_style(preset="presentation_default", overrides={"show_reverse_complement": True})
    palette = Palette(style.palette)
    initialize_runtime()
    fig = render_record(record, renderer_name="sequence_rows", style=style, palette=palette)
    try:
        left_ax, _right_ax = sorted(fig.axes, key=lambda axis: float(axis.get_position().x0))
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        left_box = left_ax.get_window_extent(renderer=renderer)
        assert float(left_box.width) == pytest.approx(float(left_box.height), rel=0.03, abs=2.0)
    finally:
        plt.close(fig)


def test_sequence_rows_trajectory_panel_preserves_sequence_axis_scale() -> None:
    record = Record(
        id="row-sequence-scale",
        alphabet="DNA",
        sequence="ACGTACGTACGTACGTACGTACGT",
        effects=(),
        display=Display(
            trajectory_panel=TrajectoryPanel(
                x=(0.0, 50_000.0, 100_000.0, 150_000.0, 200_000.0),
                y=(0.05, 0.21, 0.29, 0.38, 0.44),
                point_index=4,
            )
        ),
    ).validate()
    style = resolve_style(preset="presentation_default", overrides={"show_reverse_complement": True})
    palette = Palette(style.palette)
    initialize_runtime()
    fig = render_record(record, renderer_name="sequence_rows", style=style, palette=palette)
    try:
        _left_ax, right_ax = sorted(fig.axes, key=lambda axis: float(axis.get_position().x0))
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        right_box = right_ax.get_window_extent(renderer=renderer)
        x_min, x_max = right_ax.get_xlim()
        y_min, y_max = right_ax.get_ylim()
        x_scale = float(right_box.width) / max(1.0e-9, float(x_max - x_min))
        y_scale = float(right_box.height) / max(1.0e-9, float(y_max - y_min))
        assert x_scale == pytest.approx(y_scale, rel=1.0e-3, abs=1.0e-3)
    finally:
        plt.close(fig)


def test_sequence_rows_trajectory_panel_is_center_aligned_with_sequence_axis() -> None:
    record = Record(
        id="row-panel-center-alignment",
        alphabet="DNA",
        sequence="ACGTACGTACGTACGTACGT",
        effects=(),
        display=Display(
            trajectory_panel=TrajectoryPanel(
                x=(0.0, 1.0, 2.0, 3.0, 4.0),
                y=(0.1, 0.18, 0.23, 0.32, 0.37),
                point_index=4,
            )
        ),
    ).validate()
    style = resolve_style(preset="presentation_default", overrides={"show_reverse_complement": True})
    palette = Palette(style.palette)
    initialize_runtime()
    fig = render_record(record, renderer_name="sequence_rows", style=style, palette=palette)
    try:
        left_ax, right_ax = sorted(fig.axes, key=lambda axis: float(axis.get_position().x0))
        left_x0, left_y0, left_w, left_h = left_ax.get_position().bounds
        right_x0, right_y0, right_w, right_h = right_ax.get_position().bounds
        del left_x0, left_w, right_x0, right_w
        left_mid = float(left_y0 + (left_h / 2.0))
        right_mid = float(right_y0 + (right_h / 2.0))
        assert left_mid == pytest.approx(right_mid, abs=0.04)
    finally:
        plt.close(fig)


def test_sequence_rows_trajectory_panel_enforces_axis_label_ordering() -> None:
    record = Record(
        id="row-axis-order",
        alphabet="DNA",
        sequence="ACGTACGTACGTACGT",
        effects=(),
        display=Display(
            trajectory_panel=TrajectoryPanel(
                x=tuple(float(v) for v in range(0, 160000, 20000)),
                y=(0.12, 0.21, 0.30, 0.36, 0.41, 0.45, 0.48, 0.50),
                point_index=6,
            )
        ),
    ).validate()
    style = resolve_style(preset="presentation_default", overrides={"show_reverse_complement": True})
    palette = Palette(style.palette)
    initialize_runtime()
    fig = render_record(record, renderer_name="sequence_rows", style=style, palette=palette)
    try:
        left_ax, _right_ax = sorted(fig.axes, key=lambda axis: float(axis.get_position().x0))
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        for label_artist in (left_ax.xaxis.label, left_ax.yaxis.label):
            bbox = label_artist.get_window_extent(renderer=renderer).transformed(fig.transFigure.inverted())
            assert float(bbox.x0) >= 0.0
            assert float(bbox.y0) >= 0.0
            assert float(bbox.x1) <= 1.0
            assert float(bbox.y1) <= 1.0
        panel_bbox = left_ax.get_window_extent(renderer=renderer)
        x_label_bbox = left_ax.xaxis.label.get_window_extent(renderer=renderer)
        y_label_bbox = left_ax.yaxis.label.get_window_extent(renderer=renderer)
        xtick_bboxes = [
            label.get_window_extent(renderer=renderer)
            for label in left_ax.get_xticklabels()
            if str(label.get_text()).strip()
        ]
        ytick_bboxes = [
            label.get_window_extent(renderer=renderer)
            for label in left_ax.get_yticklabels()
            if str(label.get_text()).strip()
        ]
        assert xtick_bboxes
        assert ytick_bboxes
        tol_px = 1.0
        assert max(float(box.y1) for box in xtick_bboxes) <= float(panel_bbox.y0) + tol_px
        assert float(x_label_bbox.y1) <= min(float(box.y0) for box in xtick_bboxes) + tol_px
        assert max(float(box.x1) for box in ytick_bboxes) <= float(panel_bbox.x0) + tol_px
        assert float(y_label_bbox.x1) <= min(float(box.x0) for box in ytick_bboxes) + tol_px
    finally:
        plt.close(fig)


def test_sequence_rows_trajectory_panel_uses_readable_axis_typography_without_collisions() -> None:
    record = Record(
        id="row-axis-typography",
        alphabet="DNA",
        sequence="ACGTACGTACGTACGT",
        effects=(),
        display=Display(
            trajectory_panel=TrajectoryPanel(
                x=tuple(float(v) for v in range(0, 200000, 20000)),
                y=(0.08, 0.14, 0.20, 0.24, 0.29, 0.32, 0.36, 0.39, 0.42, 0.45),
                point_index=8,
                x_label="Sweep",
                y_label="Best objective (norm-LLR)",
            )
        ),
    ).validate()
    style = resolve_style(preset="presentation_default", overrides={"show_reverse_complement": True})
    palette = Palette(style.palette)
    initialize_runtime()
    fig = render_record(record, renderer_name="sequence_rows", style=style, palette=palette)
    try:
        left_ax, _right_ax = sorted(fig.axes, key=lambda axis: float(axis.get_position().x0))
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        x_label_size = float(left_ax.xaxis.label.get_fontsize())
        y_label_size = float(left_ax.yaxis.label.get_fontsize())
        x_tick_sizes = [
            float(label.get_fontsize()) for label in left_ax.get_xticklabels() if str(label.get_text()).strip()
        ]
        y_tick_sizes = [
            float(label.get_fontsize()) for label in left_ax.get_yticklabels() if str(label.get_text()).strip()
        ]
        assert x_tick_sizes
        assert y_tick_sizes
        assert x_label_size >= 9.0
        assert y_label_size >= 9.0
        assert min(x_tick_sizes) >= 8.0
        assert min(y_tick_sizes) >= 8.0
        x_label_box = left_ax.xaxis.label.get_window_extent(renderer=renderer)
        y_label_box = left_ax.yaxis.label.get_window_extent(renderer=renderer)
        xtick_bboxes = [
            label.get_window_extent(renderer=renderer)
            for label in left_ax.get_xticklabels()
            if str(label.get_text()).strip()
        ]
        ytick_bboxes = [
            label.get_window_extent(renderer=renderer)
            for label in left_ax.get_yticklabels()
            if str(label.get_text()).strip()
        ]
        assert all(not _bbox_overlaps(x_label_box, box) for box in xtick_bboxes)
        assert all(not _bbox_overlaps(y_label_box, box) for box in ytick_bboxes)
    finally:
        plt.close(fig)


def test_sequence_rows_without_trajectory_panel_uses_single_axis() -> None:
    record = Record(
        id="single-panel",
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
        display=Display(),
    ).validate()
    style = resolve_style(preset="presentation_default", overrides={"show_reverse_complement": True})
    palette = Palette(style.palette)
    initialize_runtime()
    fig = render_record(record, renderer_name="sequence_rows", style=style, palette=palette)
    try:
        assert len(fig.axes) == 1
    finally:
        plt.close(fig)


def test_compact_axis_formatter_keeps_monotonic_thousands_labels() -> None:
    ticks = [60_000.0, 120_000.0, 180_000.0, 240_000.0, 300_000.0]
    labels = [_format_compact_axis_value(value, 0) for value in ticks]
    assert labels == ["60k", "120k", "180k", "240k", "300k"]


def test_sequence_rows_trajectory_panel_applies_slight_sweep_xlim_padding() -> None:
    record = Record(
        id="row-sweep-xpad",
        alphabet="DNA",
        sequence="ACGTACGTACGTACGT",
        effects=(),
        display=Display(
            trajectory_panel=TrajectoryPanel(
                x=(0.0, 60_000.0, 120_000.0, 180_000.0, 240_000.0),
                y=(0.08, 0.16, 0.21, 0.27, 0.34),
                point_index=4,
            )
        ),
    ).validate()
    style = resolve_style(preset="presentation_default", overrides={"show_reverse_complement": True})
    palette = Palette(style.palette)
    initialize_runtime()
    fig = render_record(record, renderer_name="sequence_rows", style=style, palette=palette)
    try:
        left_ax, _right_ax = sorted(fig.axes, key=lambda axis: float(axis.get_position().x0))
        x_low, x_high = left_ax.get_xlim()
        assert float(x_low) < 0.0
        assert float(x_high) > 240_000.0
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
