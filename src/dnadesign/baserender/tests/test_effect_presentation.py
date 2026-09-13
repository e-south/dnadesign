"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/tests/test_effect_presentation.py

Check per-effect color controls before rendering.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from dataclasses import replace

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pytest

from dnadesign.baserender import Effect, Feature, Palette, Record, Span, Style, initialize_runtime, render_record
from dnadesign.baserender.src.core import RenderingError


def _record(params):
    return Record(
        id="bracket",
        alphabet="DNA",
        sequence="A" * 40,
        features=(
            Feature(id="left", kind="kmer", span=Span(0, 8, "fwd"), label="A" * 8),
            Feature(id="right", kind="kmer", span=Span(30, 38, "fwd"), label="A" * 8),
        ),
        effects=(
            Effect(
                kind="span_link",
                target={"from_feature_id": "left", "to_feature_id": "right"},
                params={"label": "22 bp", **params},
            ),
        ),
    )


def test_span_link_uses_its_explicit_line_and_label_colors():
    initialize_runtime()
    figure = render_record(
        _record({"color": "#D2D2D2", "label_color": "#969696"}),
        renderer_name="sequence_rows",
        style=Style(show_pair_rungs=False, show_coordinate_ticks=False, connectors=False),
        palette=Palette({"kmer": "#FF0000"}),
    )
    try:
        bracket = next(text for text in figure.axes[0].texts if text.get_text() == "22 bp")
        assert colors.to_hex(bracket.get_color()) == "#969696"
        assert all(colors.to_hex(line.get_color()) == "#d2d2d2" for line in figure.axes[0].lines)
    finally:
        plt.close(figure)


@pytest.mark.parametrize("key", ["color", "label_color"])
def test_span_link_rejects_invalid_colors_before_allocating_a_figure(key):
    initialize_runtime()
    before = plt.get_fignums()
    with pytest.raises(RenderingError, match=key):
        render_record(
            _record({key: "bad-color"}),
            renderer_name="sequence_rows",
            style=Style(show_pair_rungs=False, show_coordinate_ticks=False, connectors=False),
            palette=Palette({"kmer": "#FF0000"}),
        )
    assert plt.get_fignums() == before


def _illustration(tint):
    return replace(
        _record({}),
        effects=(
            Effect(
                kind="anchored_illustration",
                target={
                    "bindings": [
                        {"anchor_id": "upstream", "feature_id": "left", "start": 0, "end": 8},
                        {"anchor_id": "downstream", "feature_id": "right", "start": 30, "end": 38},
                    ]
                },
                params={"asset_id": "rnap_sigma70", "image_tint": tint},
            ),
        ),
    )


def test_anchored_illustration_tint_preserves_shape_and_alpha():
    initialize_runtime()
    figure = render_record(
        _illustration("#D2D2D2"),
        renderer_name="sequence_rows",
        style=Style(show_pair_rungs=False, show_coordinate_ticks=False, connectors=False),
        palette=Palette({"kmer": "#FF0000"}),
    )
    try:
        pixels = np.asarray(figure.axes[0].images[0].get_array())
        assert np.allclose(pixels[:, :, :3], 210 / 255)
        assert np.any(pixels[:, :, 3] > 0)
        assert np.any(pixels[:, :, 3] == 0)
    finally:
        plt.close(figure)


def test_anchored_illustration_rejects_invalid_tint_before_allocating_a_figure():
    initialize_runtime()
    before = plt.get_fignums()
    with pytest.raises(RenderingError, match="image_tint"):
        render_record(
            _illustration("bad-color"),
            renderer_name="sequence_rows",
            style=Style(show_pair_rungs=False, show_coordinate_ticks=False, connectors=False),
            palette=Palette({"kmer": "#FF0000"}),
        )
    assert plt.get_fignums() == before


@pytest.mark.parametrize(
    ("params", "binding_presentation", "expected"),
    [
        ({}, {}, ["#dde2e7", "#dde2e7"]),
        ({"fill_color": "#FF0000"}, {}, ["#ff0000", "#ff0000"]),
        ({"fill_color": "#FF0000"}, {"fill_color": "#D2D2D2"}, ["#ff0000", "#d2d2d2"]),
    ],
)
def test_anchored_illustration_binding_color_overrides_only_its_footprint(params, binding_presentation, expected):
    initialize_runtime()
    record = _illustration("#D2D2D2")
    effect = record.effects[0]
    bindings = [dict(binding) for binding in effect.target["bindings"]]
    bindings[1].update(binding_presentation)
    record = replace(
        record,
        effects=(replace(effect, target={"bindings": bindings}, params={**effect.params, **params}),),
    )
    figure = render_record(
        record,
        renderer_name="sequence_rows",
        style=Style(show_pair_rungs=False, show_coordinate_ticks=False, connectors=False),
        palette=Palette({"kmer": "#FF0000"}),
    )
    try:
        footprints = [
            patch
            for patch in figure.axes[0].patches
            if (patch.get_gid() or "").startswith("anchored_illustration_footprint:")
        ]
        assert [colors.to_hex(patch.get_facecolor()) for patch in footprints] == expected
        assert all(patch.get_alpha() == 0.42 for patch in footprints)
    finally:
        plt.close(figure)


@pytest.mark.parametrize("color", [False, None, "bad-color", 123, ["#D2D2D2"]])
def test_anchored_illustration_rejects_invalid_binding_color_before_allocating_a_figure(color):
    initialize_runtime()
    record = _illustration("#D2D2D2")
    effect = record.effects[0]
    bindings = [dict(binding) for binding in effect.target["bindings"]]
    bindings[1]["fill_color"] = color
    record = replace(record, effects=(replace(effect, target={"bindings": bindings}),))
    before = plt.get_fignums()
    with pytest.raises(RenderingError, match=r"target.bindings\[1\].fill_color"):
        render_record(
            record,
            renderer_name="sequence_rows",
            style=Style(show_pair_rungs=False, show_coordinate_ticks=False, connectors=False),
            palette=Palette({"kmer": "#FF0000"}),
        )
    assert plt.get_fignums() == before
