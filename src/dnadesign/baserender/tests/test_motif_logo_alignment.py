"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/baserender/tests/test_motif_logo_alignment.py

Regression tests for motif-logo alignment and overlap lane behavior.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pytest

from dnadesign.baserender.src.config import resolve_style
from dnadesign.baserender.src.core import Record, SchemaError, Span
from dnadesign.baserender.src.core.record import Display, Effect, Feature, revcomp
from dnadesign.baserender.src.render.effects.motif_logo import (
    _logo_stack_bits,
    compute_motif_logo_geometry,
)
from dnadesign.baserender.src.render.layout import compute_layout
from dnadesign.baserender.src.render.layout import span_to_x as layout_span_to_x
from dnadesign.baserender.src.render.palette import Palette
from dnadesign.baserender.src.render.sequence_rows import SequenceRowsRenderer


def _logo_matrix(length: int) -> list[list[float]]:
    row = [0.25, 0.25, 0.25, 0.25]
    return [row[:] for _ in range(length)]


def test_motif_logo_geometry_aligns_to_feature_span_grid() -> None:
    sequence = "TTACGTACGTTT"
    record = Record(
        id="align",
        alphabet="DNA",
        sequence=sequence,
        features=(
            Feature(
                id="k1",
                kind="kmer",
                span=Span(start=2, end=10, strand="fwd"),
                label=sequence[2:10],
                tags=("tf:lexA",),
                attrs={},
                render={},
            ),
        ),
        effects=(
            Effect(
                kind="motif_logo",
                target={"feature_id": "k1"},
                params={"matrix": _logo_matrix(8)},
                render={},
            ),
        ),
        display=Display(),
        meta={},
    )
    style = resolve_style(
        preset=None,
        overrides={"motif_logo": {"layout": "stack", "lane_mode": "follow_feature_track"}},
    )
    layout = compute_layout(record, style)

    geometry = compute_motif_logo_geometry(record=record, effect_index=0, layout=layout, style=style)
    expected_x0, expected_x1 = layout_span_to_x(layout, 2, 10)
    assert geometry.x0 == expected_x0
    assert geometry.x1 == expected_x1
    assert geometry.columns[0] == layout_span_to_x(layout, 2, 3)
    assert geometry.columns[-1] == layout_span_to_x(layout, 9, 10)


def test_overlapping_motif_logos_use_distinct_lanes_in_stack_mode() -> None:
    sequence = "CTGCATATATTTACAG"
    record = Record(
        id="stacked",
        alphabet="DNA",
        sequence=sequence,
        features=(
            Feature(
                id="a",
                kind="kmer",
                span=Span(start=2, end=12, strand="fwd"),
                label=sequence[2:12],
                tags=("tf:lexA",),
                attrs={},
                render={},
            ),
            Feature(
                id="b",
                kind="kmer",
                span=Span(start=6, end=16, strand="fwd"),
                label=sequence[6:16],
                tags=("tf:cpxR",),
                attrs={},
                render={},
            ),
        ),
        effects=(
            Effect(kind="motif_logo", target={"feature_id": "a"}, params={"matrix": _logo_matrix(10)}, render={}),
            Effect(kind="motif_logo", target={"feature_id": "b"}, params={"matrix": _logo_matrix(10)}, render={}),
        ),
        display=Display(),
        meta={},
    )
    style = resolve_style(
        preset=None,
        overrides={"motif_logo": {"layout": "stack", "lane_mode": "follow_feature_track"}},
    )
    layout = compute_layout(record, style)

    first = compute_motif_logo_geometry(record=record, effect_index=0, layout=layout, style=style)
    second = compute_motif_logo_geometry(record=record, effect_index=1, layout=layout, style=style)
    assert first.lane != second.lane


def test_follow_track_mode_anchors_logo_to_target_kmer_box() -> None:
    sequence = "ATGCATATATTTACAA"
    record = Record(
        id="anchor",
        alphabet="DNA",
        sequence=sequence,
        features=(
            Feature(
                id="k_top",
                kind="kmer",
                span=Span(start=1, end=9, strand="fwd"),
                label=sequence[1:9],
                tags=("tf:lexA",),
                attrs={},
                render={"track": 1},
            ),
            Feature(
                id="k_bottom",
                kind="kmer",
                span=Span(start=2, end=10, strand="rev"),
                label=revcomp(sequence[2:10]),
                tags=("tf:cpxR",),
                attrs={},
                render={"track": 0},
            ),
        ),
        effects=(
            Effect(kind="motif_logo", target={"feature_id": "k_top"}, params={"matrix": _logo_matrix(8)}, render={}),
            Effect(
                kind="motif_logo",
                target={"feature_id": "k_bottom"},
                params={"matrix": _logo_matrix(8)},
                render={},
            ),
        ),
        display=Display(),
        meta={},
    )
    style = resolve_style(
        preset=None,
        overrides={"motif_logo": {"layout": "stack", "lane_mode": "follow_feature_track"}},
    )
    layout = compute_layout(record, style)

    top_geo = compute_motif_logo_geometry(record=record, effect_index=0, layout=layout, style=style)
    bottom_geo = compute_motif_logo_geometry(record=record, effect_index=1, layout=layout, style=style)
    top_box = layout.feature_boxes["k_top"]
    bottom_box = layout.feature_boxes["k_bottom"]
    assert top_geo.y0 == pytest.approx(top_box[3] + layout.motif_logo_gap)
    assert bottom_geo.y0 == pytest.approx(bottom_box[1] - layout.motif_logo_gap - layout.motif_logo_height)
    assert top_geo.baseline == "bottom"
    assert bottom_geo.baseline == "top"


def test_follow_track_mode_prevents_logo_kmer_cross_overlap() -> None:
    sequence = "CTGCATATATTTACAG"
    record = Record(
        id="no_cross",
        alphabet="DNA",
        sequence=sequence,
        features=(
            Feature(
                id="f1",
                kind="kmer",
                span=Span(start=2, end=12, strand="fwd"),
                label=sequence[2:12],
                tags=("tf:lexA",),
                attrs={},
                render={},
            ),
            Feature(
                id="f2",
                kind="kmer",
                span=Span(start=6, end=16, strand="fwd"),
                label=sequence[6:16],
                tags=("tf:cpxR",),
                attrs={},
                render={},
            ),
        ),
        effects=(
            Effect(kind="motif_logo", target={"feature_id": "f1"}, params={"matrix": _logo_matrix(10)}, render={}),
            Effect(kind="motif_logo", target={"feature_id": "f2"}, params={"matrix": _logo_matrix(10)}, render={}),
        ),
        display=Display(),
        meta={},
    )
    style = resolve_style(
        preset=None,
        overrides={"motif_logo": {"layout": "stack", "lane_mode": "follow_feature_track"}},
    )
    layout = compute_layout(record, style)

    g1 = compute_motif_logo_geometry(record=record, effect_index=0, layout=layout, style=style)
    g2 = compute_motif_logo_geometry(record=record, effect_index=1, layout=layout, style=style)
    box1 = layout.feature_boxes["f1"]
    box2 = layout.feature_boxes["f2"]

    assert g1.y0 >= box2[3] or (g1.y0 + g1.height) <= box2[1]
    assert g2.y0 >= box1[3] or (g2.y0 + g2.height) <= box1[1]


def test_information_mode_uses_fixed_bit_scale_with_whitespace() -> None:
    low_info = _logo_stack_bits((0.25, 0.25, 0.25, 0.25), max_bits=2.0)
    high_info = _logo_stack_bits((1.0, 0.0, 0.0, 0.0), max_bits=2.0)
    assert sum(v for _, v in low_info) == pytest.approx(0.0)
    assert sum(v for _, v in high_info) == pytest.approx(2.0)


def test_sequence_and_logo_gap_style_keys_affect_vertical_layout() -> None:
    sequence = "CTGCATATATTTACAG"
    record = Record(
        id="gaps",
        alphabet="DNA",
        sequence=sequence,
        features=(
            Feature(
                id="k1",
                kind="kmer",
                span=Span(start=2, end=12, strand="fwd"),
                label=sequence[2:12],
                tags=("tf:lexA",),
                attrs={},
                render={},
            ),
        ),
        effects=(
            Effect(kind="motif_logo", target={"feature_id": "k1"}, params={"matrix": _logo_matrix(10)}, render={}),
        ),
        display=Display(),
        meta={},
    )
    compact_style = resolve_style(
        preset=None,
        overrides={
            "sequence": {"to_kmer_gap_cells": 0.10, "strand_gap_cells": 0.30},
            "kmer": {"to_logo_gap_cells": 0.08},
            "motif_logo": {"lane_mode": "follow_feature_track"},
        },
    )
    roomy_style = resolve_style(
        preset=None,
        overrides={
            "sequence": {"to_kmer_gap_cells": 1.10, "strand_gap_cells": 1.20},
            "kmer": {"to_logo_gap_cells": 0.90},
            "motif_logo": {"lane_mode": "follow_feature_track"},
        },
    )
    compact_layout = compute_layout(record, compact_style)
    roomy_layout = compute_layout(record, roomy_style)
    compact_box = compact_layout.feature_boxes["k1"]
    roomy_box = roomy_layout.feature_boxes["k1"]
    compact_logo = compute_motif_logo_geometry(
        record=record,
        effect_index=0,
        layout=compact_layout,
        style=compact_style,
    )
    roomy_logo = compute_motif_logo_geometry(record=record, effect_index=0, layout=roomy_layout, style=roomy_style)

    assert (roomy_box[1] - roomy_layout.y_forward) > (compact_box[1] - compact_layout.y_forward)
    assert (roomy_layout.y_forward - roomy_layout.y_reverse) > (compact_layout.y_forward - compact_layout.y_reverse)
    assert (roomy_logo.y0 - roomy_box[3]) > (compact_logo.y0 - compact_box[3])


def test_top_and_bottom_spacing_are_symmetric_for_equal_tracks() -> None:
    sequence = "CTGCATATATTTACAG"
    record = Record(
        id="symmetry",
        alphabet="DNA",
        sequence=sequence,
        features=(
            Feature(
                id="top",
                kind="kmer",
                span=Span(start=1, end=11, strand="fwd"),
                label=sequence[1:11],
                tags=("tf:lexA",),
                attrs={},
                render={"track": 0},
            ),
            Feature(
                id="bottom",
                kind="kmer",
                span=Span(start=3, end=13, strand="rev"),
                label=revcomp(sequence[3:13]),
                tags=("tf:cpxR",),
                attrs={},
                render={"track": 0},
            ),
        ),
        effects=(
            Effect(kind="motif_logo", target={"feature_id": "top"}, params={"matrix": _logo_matrix(10)}, render={}),
            Effect(kind="motif_logo", target={"feature_id": "bottom"}, params={"matrix": _logo_matrix(10)}, render={}),
        ),
        display=Display(),
        meta={},
    )
    style = resolve_style(
        preset=None,
        overrides={
            "sequence": {"to_kmer_gap_cells": 0.20, "strand_gap_cells": 0.40},
            "kmer": {"to_logo_gap_cells": 0.15},
            "motif_logo": {"lane_mode": "follow_feature_track"},
        },
    )
    layout = compute_layout(record, style)
    top_box = layout.feature_boxes["top"]
    bottom_box = layout.feature_boxes["bottom"]
    top_logo = compute_motif_logo_geometry(record=record, effect_index=0, layout=layout, style=style)
    bottom_logo = compute_motif_logo_geometry(record=record, effect_index=1, layout=layout, style=style)

    top_seq_boundary = layout.y_forward + layout.sequence_extent_up
    bottom_seq_boundary = layout.y_reverse - layout.sequence_extent_down

    top_seq_to_kmer = top_box[1] - top_seq_boundary
    bottom_seq_to_kmer = bottom_seq_boundary - bottom_box[3]
    assert top_seq_to_kmer == pytest.approx(bottom_seq_to_kmer)

    top_kmer_to_logo = top_logo.y0 - top_box[3]
    bottom_kmer_to_logo = bottom_box[1] - (bottom_logo.y0 + bottom_logo.height)
    assert top_kmer_to_logo == pytest.approx(bottom_kmer_to_logo)


def test_nearest_top_bottom_kmer_spacing_stays_symmetric_with_stacked_top_tracks() -> None:
    sequence = "CTGCATATATTTACAG"
    record = Record(
        id="symmetry_stacked_top",
        alphabet="DNA",
        sequence=sequence,
        features=(
            Feature(
                id="top1",
                kind="kmer",
                span=Span(start=1, end=11, strand="fwd"),
                label=sequence[1:11],
                tags=("tf:lexA",),
                attrs={},
                render={},
            ),
            Feature(
                id="top2",
                kind="kmer",
                span=Span(start=3, end=13, strand="fwd"),
                label=sequence[3:13],
                tags=("tf:cpxR",),
                attrs={},
                render={},
            ),
            Feature(
                id="bottom",
                kind="kmer",
                span=Span(start=2, end=12, strand="rev"),
                label=revcomp(sequence[2:12]),
                tags=("tf:rpoD",),
                attrs={},
                render={},
            ),
        ),
        effects=(),
        display=Display(),
        meta={},
    )
    style = resolve_style(
        preset=None,
        overrides={
            "sequence": {"to_kmer_gap_cells": 0.20, "strand_gap_cells": 0.40},
        },
    )
    layout = compute_layout(record, style)

    top_seq_boundary = layout.y_forward + layout.sequence_extent_up
    bottom_seq_boundary = layout.y_reverse - layout.sequence_extent_down
    top_boxes = [layout.feature_boxes["top1"], layout.feature_boxes["top2"]]
    bottom_box = layout.feature_boxes["bottom"]

    top_nearest_gap = min(box[1] - top_seq_boundary for box in top_boxes)
    bottom_nearest_gap = bottom_seq_boundary - bottom_box[3]
    assert top_nearest_gap == pytest.approx(bottom_nearest_gap)


def test_prime_labels_have_symmetric_left_right_offsets_per_row() -> None:
    sequence = "CTGCATATATTTACAG"
    record = Record(
        id="prime_label_symmetry",
        alphabet="DNA",
        sequence=sequence,
        features=(),
        effects=(),
        display=Display(),
        meta={},
    )
    style = resolve_style(preset=None, overrides={})
    layout = compute_layout(record, style)
    fig = SequenceRowsRenderer().render(record, style, Palette(style.palette))
    ax = fig.axes[0]

    seq_left = layout.x_left
    seq_right = layout.x_left + len(sequence) * layout.cw

    for row_y, left_token, right_token in (
        (layout.y_forward, "5'", "3'"),
        (layout.y_reverse, "3'", "5'"),
    ):
        left = next(
            t
            for t in ax.texts
            if t.get_text() == left_token and t.get_ha() == "right" and abs(t.get_position()[1] - row_y) < 1e-6
        )
        right = next(
            t
            for t in ax.texts
            if t.get_text() == right_token and t.get_ha() == "left" and abs(t.get_position()[1] - row_y) < 1e-6
        )
        left_gap = seq_left - left.get_position()[0]
        right_gap = right.get_position()[0] - seq_right
        assert left_gap == pytest.approx(right_gap)

    plt.close(fig)


def test_layout_sequence_half_height_covers_rendered_sequence_glyph_extents() -> None:
    sequence = "CTGCATATATTTACAG"
    record = Record(
        id="seq_envelope",
        alphabet="DNA",
        sequence=sequence,
        features=(),
        effects=(),
        display=Display(),
        meta={},
    )
    style = resolve_style(preset=None, overrides={})
    layout = compute_layout(record, style)
    fig = SequenceRowsRenderer().render(record, style, Palette(style.palette))
    ax = fig.axes[0]
    renderer = fig.canvas.get_renderer()

    fwd_patches = [p for p in ax.patches if (p.get_gid() or "").startswith("sequence:fwd:")]
    assert fwd_patches
    y_min = float("inf")
    y_max = float("-inf")
    inv = ax.transData.inverted()
    for patch in fwd_patches:
        bbox_disp = patch.get_window_extent(renderer=renderer)
        bbox_data = inv.transform_bbox(bbox_disp)
        y_min = min(y_min, bbox_data.y0)
        y_max = max(y_max, bbox_data.y1)

    assert (y_max - layout.y_forward) <= (layout.sequence_extent_up + 1e-3)
    assert (layout.y_forward - y_min) <= (layout.sequence_extent_down + 1e-3)


def test_multiline_overlay_text_reserves_additional_title_space() -> None:
    sequence = "TTGACAAAAAAAAAAAAAAAATATAAT"
    base_feature = Feature(
        id="k1",
        kind="kmer",
        span=Span(start=0, end=6, strand="fwd"),
        label="TTGACA",
        tags=("tf:lexA",),
        attrs={},
        render={},
    )
    base_effect = Effect(
        kind="motif_logo",
        target={"feature_id": "k1"},
        params={"matrix": _logo_matrix(6)},
        render={},
    )
    single = Record(
        id="single_line_overlay",
        alphabet="DNA",
        sequence=sequence,
        features=(base_feature,),
        effects=(base_effect,),
        display=Display(overlay_text="Source\nElite"),
        meta={},
    )
    multiline = Record(
        id="multi_line_overlay",
        alphabet="DNA",
        sequence=sequence,
        features=(base_feature,),
        effects=(base_effect,),
        display=Display(overlay_text="Source\nElite\nlexA=0.91 cpxR=0.87"),
        meta={},
    )
    style = resolve_style(
        preset=None,
        overrides={"motif_logo": {"layout": "stack", "lane_mode": "follow_feature_track"}},
    )
    single_layout = compute_layout(single, style)
    multiline_layout = compute_layout(multiline, style)

    one_line_px = max((style.font_size_label / 72.0 * style.dpi) * 1.05, single_layout.ch * 0.5)
    assert (multiline_layout.height - single_layout.height) >= (one_line_px * 0.9)


def test_style_accepts_new_logo_coloring_and_scale_bar_location() -> None:
    style = resolve_style(
        preset=None,
        overrides={
            "layout": {"outer_pad_cells": 0.35},
            "sequence": {"strand_gap_cells": 0.45, "to_kmer_gap_cells": 0.25},
            "kmer": {"to_logo_gap_cells": 0.12},
            "motif_logo": {
                "letter_coloring": {
                    "mode": "match_window_seq",
                    "other_color": "#d1d5db",
                    "observed_color_source": "feature_fill",
                },
                "scale_bar": {"enabled": True, "location": "left_of_logo"},
            },
        },
    )
    assert style.layout.outer_pad_cells == pytest.approx(0.35)
    assert style.sequence.strand_gap_cells == pytest.approx(0.45)
    assert style.sequence.to_kmer_gap_cells == pytest.approx(0.25)
    assert style.kmer.to_logo_gap_cells == pytest.approx(0.12)
    assert style.motif_logo.letter_coloring.mode == "match_window_seq"
    assert style.motif_logo.scale_bar.location == "left_of_logo"


def test_densegen_demo_style_preserves_top_bottom_sequence_to_kmer_symmetry() -> None:
    sequence = "CTGCATATATTTACAG"
    record = Record(
        id="densegen_symmetry",
        alphabet="DNA",
        sequence=sequence,
        features=(
            Feature(
                id="lexa",
                kind="kmer",
                span=Span(start=1, end=11, strand="fwd"),
                label=sequence[1:11],
                tags=("tf:lexa",),
                attrs={},
                render={},
            ),
            Feature(
                id="cpxr",
                kind="kmer",
                span=Span(start=3, end=13, strand="rev"),
                label=revcomp(sequence[3:13]),
                tags=("tf:cpxr",),
                attrs={},
                render={},
            ),
        ),
        effects=(),
        display=Display(),
        meta={},
    )
    style = resolve_style(
        preset="presentation_default",
        overrides={
            "palette": {"tf:lexa": "#1f77b4", "tf:cpxr": "#2ca02c"},
            "legend_font_size": 10,
            "connector_alpha": 0.35,
        },
    )
    layout = compute_layout(record, style)

    top_seq_boundary = layout.y_forward + layout.sequence_extent_up
    bottom_seq_boundary = layout.y_reverse - layout.sequence_extent_down
    top_box = layout.feature_boxes["lexa"]
    bottom_box = layout.feature_boxes["cpxr"]

    top_seq_to_kmer = top_box[1] - top_seq_boundary
    bottom_seq_to_kmer = bottom_seq_boundary - bottom_box[3]
    assert top_seq_to_kmer == pytest.approx(bottom_seq_to_kmer)


def test_unknown_motif_logo_style_key_is_rejected() -> None:
    with pytest.raises(SchemaError, match="Unknown style.motif_logo key"):
        resolve_style(
            preset=None,
            overrides={"motif_logo": {"unexpected_key": True}},
        )
