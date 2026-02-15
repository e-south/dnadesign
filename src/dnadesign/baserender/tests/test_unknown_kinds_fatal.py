"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/baserender/tests/test_unknown_kinds_fatal.py

Tests that unknown feature/effect kinds fail fast during rendering.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import pytest

from dnadesign.baserender.src.config import resolve_style
from dnadesign.baserender.src.core import Record, RenderingError, Span
from dnadesign.baserender.src.core.record import Display, Effect, Feature
from dnadesign.baserender.src.render import Palette, render_record


def _style_palette():
    style = resolve_style(preset=None, overrides={})
    return style, Palette(style.palette)


def test_unknown_feature_kind_is_fatal() -> None:
    style, palette = _style_palette()
    record = Record(
        id="r1",
        alphabet="DNA",
        sequence="ACGTACGT",
        features=(
            Feature(
                id="f1",
                kind="typo_feature_kind",
                span=Span(start=0, end=4, strand="fwd"),
                label="ACGT",
                tags=("tf:x",),
                attrs={},
                render={},
            ),
        ),
        effects=(),
        display=Display(),
        meta={},
    )

    with pytest.raises(RenderingError, match="Unknown feature kind"):
        render_record(record, renderer_name="sequence_rows", style=style, palette=palette)


def test_unknown_effect_kind_is_fatal() -> None:
    style, palette = _style_palette()
    record = Record(
        id="r1",
        alphabet="DNA",
        sequence="ACGTACGT",
        features=(
            Feature(
                id="f1",
                kind="kmer",
                span=Span(start=0, end=4, strand="fwd"),
                label="ACGT",
                tags=("tf:x",),
                attrs={},
                render={},
            ),
        ),
        effects=(Effect(kind="typo_effect_kind", target={"feature_id": "f1"}, params={}, render={}),),
        display=Display(),
        meta={},
    )

    with pytest.raises(RenderingError, match="Unknown effect kind"):
        render_record(record, renderer_name="sequence_rows", style=style, palette=palette)


def test_span_link_with_collapsed_geometry_is_fatal() -> None:
    style, palette = _style_palette()
    record = Record(
        id="r1",
        alphabet="DNA",
        sequence="ACGTACGTAC",
        features=(
            Feature(
                id="f1",
                kind="kmer",
                span=Span(start=0, end=4, strand="fwd"),
                label="ACGT",
                tags=("tf:x",),
                attrs={},
                render={},
            ),
            Feature(
                id="f2",
                kind="kmer",
                span=Span(start=5, end=9, strand="fwd"),
                label="CGTA",
                tags=("tf:y",),
                attrs={},
                render={},
            ),
        ),
        effects=(
            Effect(
                kind="span_link",
                target={"from_feature_id": "f1", "to_feature_id": "f2"},
                params={"inner_margin_bp": 1000.0},
                render={},
            ),
        ),
        display=Display(),
        meta={},
    )

    with pytest.raises(RenderingError, match="collapsed geometry"):
        render_record(record, renderer_name="sequence_rows", style=style, palette=palette)


def test_span_link_unknown_target_keys_are_fatal() -> None:
    style, palette = _style_palette()
    record = Record(
        id="r1",
        alphabet="DNA",
        sequence="ACGTACGTAC",
        features=(
            Feature(
                id="f1",
                kind="kmer",
                span=Span(start=0, end=4, strand="fwd"),
                label="ACGT",
                tags=("tf:x",),
                attrs={},
                render={},
            ),
            Feature(
                id="f2",
                kind="kmer",
                span=Span(start=6, end=10, strand="fwd"),
                label="GTAC",
                tags=("tf:y",),
                attrs={},
                render={},
            ),
        ),
        effects=(
            Effect(
                kind="span_link",
                target={
                    "from_feature_id": "f1",
                    "to_feature_id": "f2",
                    "unexpected_target": "typo",
                },
                params={},
                render={},
            ),
        ),
        display=Display(),
        meta={},
    )

    with pytest.raises(RenderingError, match="Unknown keys in span_link.target"):
        render_record(record, renderer_name="sequence_rows", style=style, palette=palette)


def test_span_link_unknown_params_keys_are_fatal() -> None:
    style, palette = _style_palette()
    record = Record(
        id="r1",
        alphabet="DNA",
        sequence="ACGTACGTAC",
        features=(
            Feature(
                id="f1",
                kind="kmer",
                span=Span(start=0, end=4, strand="fwd"),
                label="ACGT",
                tags=("tf:x",),
                attrs={},
                render={},
            ),
            Feature(
                id="f2",
                kind="kmer",
                span=Span(start=6, end=10, strand="fwd"),
                label="GTAC",
                tags=("tf:y",),
                attrs={},
                render={},
            ),
        ),
        effects=(
            Effect(
                kind="span_link",
                target={"from_feature_id": "f1", "to_feature_id": "f2"},
                params={"label": "x", "unexpected_param": "typo"},
                render={},
            ),
        ),
        display=Display(),
        meta={},
    )

    with pytest.raises(RenderingError, match="Unknown keys in span_link.params"):
        render_record(record, renderer_name="sequence_rows", style=style, palette=palette)


def test_motif_logo_unknown_params_keys_are_fatal() -> None:
    style, palette = _style_palette()
    record = Record(
        id="r1",
        alphabet="DNA",
        sequence="ACGTACGT",
        features=(
            Feature(
                id="k1",
                kind="kmer",
                span=Span(start=0, end=4, strand="fwd"),
                label="ACGT",
                tags=("tf:x",),
                attrs={},
                render={},
            ),
        ),
        effects=(
            Effect(
                kind="motif_logo",
                target={"feature_id": "k1"},
                params={
                    "matrix": [
                        [0.9, 0.03, 0.04, 0.03],
                        [0.03, 0.9, 0.04, 0.03],
                        [0.03, 0.04, 0.9, 0.03],
                        [0.03, 0.04, 0.03, 0.9],
                    ],
                    "unexpected_param": True,
                },
                render={},
            ),
        ),
        display=Display(),
        meta={},
    )

    with pytest.raises(RenderingError, match="Unknown keys in motif_logo.params"):
        render_record(record, renderer_name="sequence_rows", style=style, palette=palette)
