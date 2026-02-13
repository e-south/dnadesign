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
from dnadesign.baserender.src.core.record import Display, Effect, Feature
from dnadesign.baserender.src.render.effects.motif_logo import compute_motif_logo_geometry
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
    style = resolve_style(preset=None, overrides={"motif_logo": {"layout": "stack"}})
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
    style = resolve_style(preset=None, overrides={"motif_logo": {"layout": "stack"}})
    layout = compute_layout(record, style)

    first = compute_motif_logo_geometry(record=record, effect_index=0, layout=layout, style=style)
    second = compute_motif_logo_geometry(record=record, effect_index=1, layout=layout, style=style)
    assert first.lane != second.lane


def test_unknown_motif_logo_style_key_is_rejected() -> None:
    with pytest.raises(SchemaError, match="Unknown style.motif_logo key"):
        resolve_style(
            preset=None,
            overrides={"motif_logo": {"unexpected_key": True}},
        )
