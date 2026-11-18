"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/tests/test_text_measurement.py

Public text measurements share literal font geometry with native caption output.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

import json
import subprocess
import sys

import pytest

import dnadesign.baserender as br


def test_text_measurement_import_remains_lazy():
    code = (
        "from dnadesign.baserender import measure_text, TextMeasurement\nimport sys\n"
        "assert 'matplotlib' not in sys.modules\nassert 'numpy' not in sys.modules"
    )
    result = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_public_measurement_preserves_type_size_and_reports_font_identity():
    first = br.measure_text("ACGT", family="Arial", size_pt=12)
    second = br.measure_text("ACGT", family="Arial", size_pt=18)
    assert isinstance(first, br.TextMeasurement)
    assert first.size_pt == 12 and second.size_pt == 18
    assert first.ink_height_pt > 0 and first.ink_width_pt > 0
    assert second.ink_bounds_pt == pytest.approx(tuple(value * 1.5 for value in first.ink_bounds_pt))
    assert second.advance_pt == pytest.approx(first.advance_pt * 1.5)
    assert first.font.resolved_family == "Arial" and len(first.font.sha256) == 64
    mapping = first.to_mapping(portable=True)
    assert "path" not in mapping["font"] and json.dumps(mapping)


@pytest.mark.parametrize("text", ["", " ", "   "])
def test_blank_text_has_finite_empty_ink_and_a_measured_advance(text):
    result = br.measure_text(text, family="Arial", size_pt=12)
    assert result.ink_bounds_pt == (0, 0, 0, 0)
    assert result.advance_pt >= 0
    assert bool(result.advance_pt) == bool(text)


@pytest.mark.parametrize("size", [0, -1, float("nan"), float("inf"), True])
def test_invalid_measurement_size_fails_before_font_loading(size, monkeypatch):
    from dnadesign.baserender.src.render import fonts

    monkeypatch.setattr(fonts, "resolve_font", lambda *args: pytest.fail("invalid size loaded a font"))
    with pytest.raises(br.RenderingError, match="size_pt"):
        br.measure_text("ACGT", family="Arial", size_pt=size)


@pytest.mark.parametrize("text", [None, 42, "line\nbreak", "tab\there", "a\u2028b", "\x00"])
def test_invalid_literal_text_fails_before_font_loading(text, monkeypatch):
    from dnadesign.baserender.src.render import fonts

    monkeypatch.setattr(fonts, "resolve_font", lambda *args: pytest.fail("invalid text loaded a font"))
    with pytest.raises(br.RenderingError, match="text"):
        br.measure_text(text, family="Arial", size_pt=12)


def test_missing_font_and_glyph_fail_explicitly():
    with pytest.raises(br.RenderingError, match="unavailable"):
        br.measure_text("ACGT", family="missing-baserender-test-font", size_pt=12)
    with pytest.raises(br.RenderingError, match="U\\+10FFFF"):
        br.measure_text("\U0010ffff", family="Arial", size_pt=12)


@pytest.mark.parametrize("text", ["ACGT", " " * 20])
def test_finite_size_that_overflows_text_geometry_fails_explicitly(text):
    with pytest.raises(br.RenderingError, match="text measurement.*finite"):
        br.measure_text(text, family="Arial", size_pt=1e308)


def test_measurement_agrees_with_serialized_literal_caption_without_allocating_its_own_figure():
    import matplotlib.pyplot as plt

    from .test_vector_output import STYLE, example

    before = plt.get_fignums()
    text = "$ A  C$"
    measured = br.measure_text(text, family="Arial", size_pt=18)
    assert plt.get_fignums() == before
    result = br.render_record_svg(
        example(),
        style_overrides=STYLE,
        captions={"rnap_sigma70": br.IllustrationCaption(text, family="Arial", size_pt=18)},
    )
    x0, y0, x1, y1 = result.diagnostics.illustrations[0].caption_bounds_pt
    assert x1 - x0 == pytest.approx(measured.ink_width_pt)
    assert y1 - y0 == pytest.approx(measured.ink_height_pt)
    origin_x, origin_y = result.diagnostics.illustrations[0].caption_origin_pt
    left, bottom, right, top = measured.ink_bounds_pt
    assert (x0, y0, x1, y1) == pytest.approx(
        (
            origin_x - measured.advance_pt / 2 + left,
            origin_y - top,
            origin_x - measured.advance_pt / 2 + right,
            origin_y - bottom,
        )
    )
