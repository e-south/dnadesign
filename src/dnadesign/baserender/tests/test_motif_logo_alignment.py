"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/baserender/tests/test_motif_logo_alignment.py

Regression tests for motif-logo alignment and overlap lane behavior.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

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


def test_unknown_motif_logo_style_key_is_rejected() -> None:
    with pytest.raises(SchemaError, match="Unknown style.motif_logo key"):
        resolve_style(
            preset=None,
            overrides={"motif_logo": {"unexpected_key": True}},
        )
