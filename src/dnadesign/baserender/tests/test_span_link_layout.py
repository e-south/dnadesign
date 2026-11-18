"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/tests/test_span_link_layout.py

Shared spacer geometry and native preserve-type admission tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

import re
from dataclasses import replace
from types import MappingProxyType
from xml.etree import ElementTree as ET

import matplotlib.pyplot as plt
import pytest

from dnadesign.baserender import (
    Effect,
    Feature,
    LayoutError,
    Record,
    RenderingError,
    SequenceLayout,
    Span,
    render_record_figure,
    render_record_svg,
)


def _record(*, gap=18, shrink=None):
    sequence = "ACGT" * 15
    start = 10 + gap
    params = {"label": f"{gap} bp"}
    if shrink is not None:
        params["shrink_label_to_fit"] = shrink
    return Record(
        id="spacer",
        alphabet="DNA",
        sequence=sequence,
        features=(
            Feature("left", "kmer", Span(4, 10, "fwd"), sequence[4:10]),
            Feature("right", "kmer", Span(start, start + 6, "fwd"), sequence[start : start + 6]),
        ),
        effects=(Effect("span_link", {"from_feature_id": "left", "to_feature_id": "right"}, params),),
    )


def test_native_span_keeps_fractional_requested_font_size():
    result = render_record_svg(_record(), style_overrides={"legend": False, "font_size_span_link_label": 12.5})
    text = next(
        node for node in ET.fromstring(result.svg).iter() if node.tag.endswith("}text") and node.text == "18 bp"
    )
    assert re.search(r"font-size:\s*12\.5px", text.get("style", ""))
    placement = result.diagnostics.span_links[0]
    assert placement.effect_index == 0
    assert placement.label == "18 bp"
    assert placement.font_size_pt == 12.5
    assert placement.endpoints_pt[0][1] == placement.endpoints_pt[1][1]
    x0, y0, x1, y1 = placement.label_bounds_pt
    assert 0 <= x0 < x1 <= result.width_pt
    assert 0 <= y0 < y1 <= result.height_pt


def test_native_short_span_fails_instead_of_reducing_type():
    before = plt.get_fignums()
    source = _record(gap=6)
    with pytest.raises(LayoutError, match="span_link.*label.*requires"):
        render_record_svg(source, style_overrides={"legend": False, "font_size_span_link_label": 48})
    assert "shrink_label_to_fit" not in source.effects[0].params
    assert plt.get_fignums() == before


def test_native_explicit_shrinking_is_an_actionable_contract_conflict():
    with pytest.raises(RenderingError, match="native.*shrink_label_to_fit"):
        render_record_svg(_record(shrink=True), style_overrides={"legend": False})


@pytest.mark.parametrize("explicit_spans", [False, True])
@pytest.mark.parametrize("lane", ["top", "bottom"])
def test_span_tracks_expand_natural_layout_but_must_fit_an_explicit_canvas(explicit_spans, lane):
    source = _record()
    target = (
        {"from_span": {"start": 4, "end": 10}, "to_span": {"start": 28, "end": 34}}
        if explicit_spans
        else source.effects[0].target
    )
    effect = replace(source.effects[0], target=target, params={"label": "", "lane": lane}, render={"track": 4})
    source = replace(source, effects=(effect,))
    ordinary = render_record_svg(_record(), style_overrides={"legend": False})
    expanded = render_record_svg(source, style_overrides={"legend": False})
    assert expanded.height_pt > ordinary.height_pt
    assert all(0 <= point[1] <= expanded.height_pt for point in expanded.diagnostics.span_links[0].endpoints_pt)
    figure = render_record_figure(source, style_overrides={"legend": False})
    try:
        assert figure.get_figheight() * 72 == pytest.approx(expanded.height_pt)
    finally:
        plt.close(figure)
    before = plt.get_fignums()
    with pytest.raises(LayoutError, match="layout canvas"):
        render_record_svg(
            source,
            style_overrides={"legend": False},
            layout=SequenceLayout(canvas_size_pt=(ordinary.width_pt, ordinary.height_pt)),
        )
    assert plt.get_fignums() == before


def test_existing_raster_span_can_still_fit_text():
    figure = render_record_figure(_record(gap=6), style_overrides={"legend": False, "font_size_span_link_label": 48})
    try:
        text = next(text for text in figure.axes[0].texts if text.get_text() == "6 bp")
        assert 6 <= text.get_fontsize() < 48
    finally:
        plt.close(figure)


def test_span_targets_accept_read_only_mappings():
    source = _record()
    effect = Effect(
        "span_link",
        {"from_span": MappingProxyType({"start": 1, "end": 6}), "to_span": {"start": 28, "end": 34}},
        {"label": "distance"},
    )
    assert render_record_svg(replace(source, effects=(effect,)), style_overrides={"legend": False}).svg


@pytest.mark.parametrize("start,end", [(-1, 6), (0, 61), (True, 6), (0.5, 6)])
def test_span_targets_reject_invalid_coordinates_before_drawing(start, end):
    source = _record()
    effect = Effect("span_link", {"from_span": {"start": start, "end": end}, "to_span": {"start": 28, "end": 34}})
    before = plt.get_fignums()
    with pytest.raises(RenderingError, match="span_link.from_span"):
        render_record_svg(replace(source, effects=(effect,)), style_overrides={"legend": False})
    assert plt.get_fignums() == before


@pytest.mark.parametrize("lane", ["top", "bottom"])
def test_planned_span_occupancy_contains_the_drawn_label(lane):
    from dnadesign.baserender import compute_layout, resolve_style
    from dnadesign.baserender.src.render.effects.span_link import compute_span_link_geometry

    source = _record()
    effect = replace(source.effects[0], params={**source.effects[0].params, "lane": lane})
    source = replace(source, effects=(effect,))
    style = resolve_style(preset=None, overrides={"legend": False, "font_size_span_link_label": 12.5})
    layout = compute_layout(source, style)
    geometry = compute_span_link_geometry(effect, layout, style, dict(layout.feature_boxes))
    figure = render_record_figure(source, style_overrides={"legend": False, "font_size_span_link_label": 12.5})
    try:
        figure.canvas.draw()
        text = next(text for text in figure.axes[0].texts if text.get_text() == "18 bp")
        bounds = text.get_window_extent(figure.canvas.get_renderer()).transformed(figure.axes[0].transData.inverted())
        x0, y0, x1, y1 = geometry.label_bounds
        assert x0 - 1 <= bounds.x0 < bounds.x1 <= x1 + 1
        assert y0 - 1 <= bounds.y0 < bounds.y1 <= y1 + 1
        assert text.get_fontsize() == geometry.font_size_pt
    finally:
        plt.close(figure)
