"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/tests/test_bottom_legend_layout.py

Measured footer typography, wrapping and fail-before-drawing contracts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from dataclasses import replace
from xml.etree import ElementTree as ET

import matplotlib.pyplot as plt
import pytest

from dnadesign.baserender import (
    Display,
    Feature,
    LayoutError,
    Record,
    RenderingError,
    SchemaError,
    SequenceLayout,
    Span,
    render_record_figure,
    render_record_svg,
    resolve_style,
)


def _record(labels=("Binding sites",)):
    sequence = "ACGT" * 4
    return Record(
        "legend",
        "DNA",
        sequence,
        features=tuple(
            Feature(f"f{i}", "kmer", Span(i * 4, i * 4 + 4, "fwd"), sequence[i * 4 : i * 4 + 4], tags=(f"tag{i}",))
            for i in range(len(labels))
        ),
        display=Display(tag_labels={f"tag{i}": label for i, label in enumerate(labels)}),
    )


def test_native_bottom_legend_preserves_fractional_type_and_reports_placement():
    result = render_record_svg(
        _record(), style_overrides={"legend_font_size": 12.5, "legend_height_px": 60, "font_label": "DejaVu Sans"}
    )
    node = next(
        node for node in ET.fromstring(result.svg).iter() if node.tag.endswith("}text") and node.text == "Binding sites"
    )
    assert "font-size: 12.5px" in node.get("style", "")
    assert "font-family: 'DejaVu Sans';" in node.get("style", "")
    entry = result.diagnostics.bottom_legend[0]
    assert entry.tag == "tag0"
    assert entry.label == "Binding sites"
    assert entry.font_size_pt == 12.5
    x0, y0, x1, y1 = entry.bounds_pt
    assert 0 <= x0 < x1 <= result.width_pt
    assert 0 <= y0 < y1 <= result.height_pt


def test_native_legend_keeps_declared_small_patch_dimensions():
    result = render_record_svg(
        _record(),
        style_overrides={"legend_patch_w": 4, "legend_patch_h": 3, "legend_gap_patch_text": 0, "dpi": 180},
    )
    x0, y0, x1, y1 = result.diagnostics.bottom_legend[0].patch_bounds_pt
    assert x1 - x0 == pytest.approx(4 * 72 / 180)
    assert y1 - y0 == pytest.approx(3 * 72 / 180)


def test_default_presentation_footer_contains_its_declared_type():
    result = render_record_svg(_record())
    assert result.diagnostics.bottom_legend[0].font_size_pt == 11


def test_native_legend_rejects_long_label_instead_of_shrinking(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("legend admission allocated a figure")

    monkeypatch.setattr(plt, "figure", forbidden)
    with pytest.raises(LayoutError, match="bottom legend.*24.*increase"):
        render_record_svg(_record(("Background regulatory binding sites",)), style_overrides={"legend_font_size": 24})


@pytest.mark.parametrize("native", [False, True])
def test_preserved_legend_rejects_insufficient_footer_before_drawing(native, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("legend admission allocated a figure")

    monkeypatch.setattr(plt, "figure", forbidden)
    style = {"legend_font_size": 12, "legend_height_px": 5}
    with pytest.raises(LayoutError, match="bottom legend.*footer"):
        if native:
            render_record_svg(_record(), style_overrides=style)
        else:
            render_record_figure(_record(), style_overrides=style, renderer_options={"layout": SequenceLayout()})


def test_ordinary_legend_retains_supported_font_fitting():
    figure = render_record_figure(
        _record(("Background regulatory binding sites",)), style_overrides={"legend_font_size": 24}
    )
    try:
        text = next(text for text in figure.axes[0].texts if text.get_text() == "Background regulatory binding sites")
        assert 6 <= text.get_fontsize() < 24
    finally:
        plt.close(figure)


@pytest.mark.parametrize("native", [False, True])
def test_all_legend_tags_admit_palette_colors_before_drawing(native, monkeypatch):
    source = _record()
    source = replace(source, features=(replace(source.features[0], tags=("tag0", "tag1")),))

    def forbidden(*args, **kwargs):
        pytest.fail("legend admission allocated a figure")

    monkeypatch.setattr(plt, "figure", forbidden)
    renderer = render_record_svg if native else render_record_figure
    with pytest.raises(RenderingError, match="bottom legend.*palette"):
        renderer(source, style_overrides={"palette": {"tag1": "invalid"}})


@pytest.mark.parametrize("dpi", [72, 180])
def test_wrapped_footer_uses_physical_row_height_and_keeps_type(dpi):
    labels = ("Background binding sites", "Regulator binding sites", "Combined binding sites")
    figure = render_record_figure(
        _record(labels),
        style_overrides={"legend_font_size": 24, "legend_height_px": 160 * dpi / 72, "dpi": dpi},
        renderer_options={"layout": SequenceLayout(canvas_size_pt=(400, 900))},
    )
    try:
        figure.canvas.draw()
        axis = figure.axes[0]
        texts = [text for text in axis.texts if text.get_text() in labels]
        assert len(texts) == len(labels)
        assert {text.get_fontsize() for text in texts} == {24}
        assert len({round(text.get_position()[1], 3) for text in texts}) > 1
        bounds = [text.get_window_extent(figure.canvas.get_renderer()) for text in texts]
        assert all(axis.bbox.contains(box.x0, box.y0) and axis.bbox.contains(box.x1, box.y1) for box in bounds)
        assert all(not first.overlaps(second) for i, first in enumerate(bounds) for second in bounds[i + 1 :])
    finally:
        plt.close(figure)


@pytest.mark.parametrize(
    "name,value",
    [
        ("legend_font_size", True),
        ("legend_font_size", 5),
        ("legend_font_size", float("inf")),
        ("legend_font_size", float("nan")),
        ("legend_height_px", -1),
        ("legend_height_px", 0),
        ("legend_patch_w", float("inf")),
        ("legend_patch_h", True),
        ("legend_gap_x", -1),
        ("legend_pad_px", float("nan")),
        ("legend_content_gap_px", float("inf")),
        ("legend_gap_patch_text", False),
    ],
)
def test_legend_style_rejects_invalid_units_at_shared_admission(name, value):
    with pytest.raises(SchemaError, match=name):
        resolve_style(preset=None, overrides={name: value})
