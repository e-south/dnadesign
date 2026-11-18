"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/tests/test_vector_placement.py

Native illustration, caption, and referenced-font placement contracts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from xml.etree import ElementTree as ET

import matplotlib.pyplot as plt
import pytest

from dnadesign.baserender import (
    Effect,
    Feature,
    IllustrationCaption,
    LayoutError,
    RenderingError,
    SequenceLayout,
    Span,
    SvgArtwork,
    render_record_figure,
    render_record_svg,
)

from .test_vector_output import STYLE, example


def _edge_record(edge):
    source = example()
    starts = (0, 8) if edge == "left" else (46, 54)
    features = tuple(
        Feature(name, "kmer", Span(start, start + 6, "fwd"), source.sequence[start : start + 6])
        for name, start in zip(("left", "right"), starts)
    )
    effect = Effect(
        "anchored_illustration",
        {
            "bindings": [
                {"anchor_id": anchor, "feature_id": feature.id, "start": feature.span.start, "end": feature.span.end}
                for anchor, feature in zip(("upstream", "downstream"), features)
            ]
        },
        {"asset_id": "rnap_sigma70", "width_px": 900},
    )
    return replace(source, features=features, effects=(effect,), meta={"fixed_content_top_extent_px": 750})


@pytest.mark.parametrize("edge", ["left", "right"])
def test_native_artwork_cannot_silently_move_to_fit(edge):
    before = plt.get_fignums()
    with pytest.raises(LayoutError, match="rnap_sigma70.*bounds.*canvas"):
        render_record_svg(_edge_record(edge), style_overrides=STYLE)
    assert plt.get_fignums() == before


def test_existing_raster_route_retains_its_declared_clamping_behavior():
    figure = render_record_figure(_edge_record("left"), style_overrides=STYLE)
    plt.close(figure)


def test_explicit_canvas_and_origin_can_resolve_artwork_overflow():
    source = replace(_edge_record("left"), meta={})
    result = render_record_svg(
        source,
        style_overrides=STYLE,
        layout=SequenceLayout(
            base_pitch_pt=10, canvas_size_pt=(1000, 800), sequence_origin_pt=(300, 500), top_extent_pt=300
        ),
    )
    assert result.diagnostics.sequence_origin_pt == pytest.approx((300, 500))
    bounds = result.diagnostics.illustrations[0].bounds_pt
    assert 0 <= bounds[0] < bounds[2] <= result.width_pt
    assert 0 <= bounds[1] < bounds[3] <= result.height_pt


@pytest.mark.parametrize("offset", [(10000, 0), (0, -10000), (0, 10000)])
def test_caption_outside_canvas_fails_before_allocation(offset):
    before = plt.get_fignums()
    with pytest.raises(LayoutError, match="rnap_sigma70.*caption.*bounds"):
        render_record_svg(
            example(),
            style_overrides=STYLE,
            captions={"rnap_sigma70": IllustrationCaption("RNAP", offset_px=offset)},
        )
    assert plt.get_fignums() == before


def test_hidden_caption_does_not_claim_visible_bounds():
    result = render_record_svg(
        example(),
        style_overrides=STYLE,
        captions={"rnap_sigma70": IllustrationCaption("RNAP", offset_px=(10000, 0), visible=False)},
    )
    placement = result.diagnostics.illustrations[0]
    assert placement.caption_bounds_pt is None
    assert placement.caption_origin_pt is None


def test_caption_diagnostics_use_the_same_baseline_as_serialization():
    result = render_record_svg(
        example(),
        style_overrides=STYLE,
        captions={"rnap_sigma70": IllustrationCaption("$RNAP$", offset_px=(4, -3))},
    )
    root = ET.fromstring(result.svg)
    text = next(node for node in root.iter() if node.get("data-baserender-caption"))
    placement = result.diagnostics.illustrations[0]
    assert text.text == "$RNAP$"
    assert (float(text.get("x")), float(text.get("y"))) == pytest.approx(placement.caption_origin_pt, rel=1e-5)
    x0, y0, x1, y1 = placement.caption_bounds_pt
    assert 0 <= x0 < x1 <= result.width_pt
    assert 0 <= y0 < y1 <= result.height_pt


@pytest.mark.parametrize("text", ["RNAP\nnext line", "RNAP\tlabel", "RNAP\x01"])
def test_caption_contract_rejects_unsupported_control_characters(text):
    with pytest.raises(RenderingError, match="caption.text.*single-line"):
        render_record_svg(example(), style_overrides=STYLE, captions={"rnap_sigma70": IllustrationCaption(text)})


def test_missing_caption_glyph_cannot_be_substituted():
    before = plt.get_fignums()
    with pytest.raises(RenderingError, match="does not cover.*U\\+10FFFF"):
        render_record_svg(
            example(),
            style_overrides=STYLE,
            captions={"rnap_sigma70": IllustrationCaption("RNAP \U0010ffff")},
        )
    assert plt.get_fignums() == before


def test_reference_policy_reports_the_fonts_that_define_geometry():
    result = render_record_svg(example(), style_overrides=STYLE)
    assert result.diagnostics.text_font_policy == "reference"
    assert {"sequence", "label", "caption:rnap_sigma70"} <= result.diagnostics.fonts.keys()
    for font in result.diagnostics.fonts.values():
        assert font.requested_family and font.resolved_family
        assert Path(font.path).is_file()
        assert sha256(Path(font.path).read_bytes()).hexdigest() == font.sha256


def test_very_large_caption_units_fail_with_context():
    with pytest.raises(RenderingError, match="caption.size_pt.*finite"):
        render_record_svg(
            example(),
            style_overrides=STYLE,
            captions={"rnap_sigma70": IllustrationCaption("RNAP", size_pt=10**1000)},
        )


def test_very_large_illustration_units_fail_with_context():
    source = example()
    effect = replace(source.effects[0], params={**source.effects[0].params, "width_px": 10**1000})
    with pytest.raises(RenderingError, match="width_px.*finite"):
        render_record_svg(replace(source, effects=(effect,)), style_overrides=STYLE)


@pytest.mark.parametrize("size", [(1e308, 1e-308), (1e-308, 1e308)])
def test_svg_frame_cannot_collapse_or_overflow_during_scaling(size):
    artwork = SvgArtwork(Path("unused.svg"), "0" * 64, ("body",), (0, 0, *size), {"center": (0.5, 0.5)})
    with pytest.raises(RenderingError, match="view_box.*aspect ratio"):
        artwork.validate()
