"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/tests/test_native_raster.py

Final PNG conversion preserves admitted SVG geometry and fails before unsafe I/O.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

import io
import math
import shutil
import subprocess
from dataclasses import replace
from functools import lru_cache
from types import MappingProxyType
from xml.etree import ElementTree as ET

import pytest
from PIL import Image

from dnadesign.baserender import Effect, Feature, Record, RenderingError, SequenceLayout, Span, render_record_svg


@lru_cache(maxsize=2)
def _svg(length=60):
    sequence = ("ACGT" * 25)[:length]
    record = Record(
        "raster",
        "DNA",
        sequence,
        features=(
            Feature("left", "kmer", Span(4, 10, "fwd"), sequence[4:10]),
            Feature("right", "kmer", Span(28, 34, "fwd"), sequence[28:34]),
        ),
        effects=(
            Effect(
                "anchored_illustration",
                {
                    "bindings": [
                        {"anchor_id": "upstream", "feature_id": "left", "start": 4, "end": 10},
                        {"anchor_id": "downstream", "feature_id": "right", "start": 28, "end": 34},
                    ]
                },
                {"asset_id": "rnap_sigma70", "width_px": 300},
            ),
        ),
    )
    return render_record_svg(
        record,
        style_overrides={"legend": False},
        layout=SequenceLayout(
            canvas_size_pt=(1200, 400),
            sequence_origin_pt=(60, 280),
            top_extent_pt=104,
        ),
    )


def _rasterize(source, **kwargs):
    from dnadesign.baserender import rasterize_svg

    return rasterize_svg(source, **kwargs)


@pytest.mark.skipif(shutil.which("inkscape") is None, reason="native PNG integration requires Inkscape")
@pytest.mark.parametrize("length", [60, 100])
@pytest.mark.parametrize("dpi", [72, 144])
def test_native_png_preserves_existing_geometry_and_transparency(length, dpi):
    source = _svg(length)
    result = _rasterize(source, dpi=dpi)
    assert result.width_px == 1200 * dpi / 72
    assert result.height_px == 400 * dpi / 72
    assert result.dpi == dpi
    assert result.diagnostics is source.diagnostics
    assert result.rasterizer.version.startswith("Inkscape ")
    assert len(result.source_svg_sha256) == 64
    image = Image.open(io.BytesIO(result.png))
    assert image.format == "PNG" and image.mode == "RGBA"
    assert image.size == (result.width_px, result.height_px)
    assert image.getchannel("A").getextrema() == (0, 255)
    x0, y0, x1, y1 = source.diagnostics.illustrations[0].bounds_pt
    region = image.crop(tuple(round(v * dpi / 72) for v in (x0, y0, x1, y1)))
    assert region.getchannel("A").getbbox() is not None


@pytest.mark.parametrize("dpi", [True, 0, -1, math.inf, math.nan, "144"])
def test_invalid_resolution_fails_before_external_execution(dpi, monkeypatch):
    source = _svg()

    def forbidden(*args, **kwargs):
        pytest.fail("invalid raster settings reached external execution")

    monkeypatch.setattr(subprocess, "run", forbidden)
    with pytest.raises(RenderingError, match="dpi"):
        _rasterize(source, dpi=dpi)


def test_pixel_budget_fails_before_external_execution(monkeypatch):
    source = _svg()

    def forbidden(*args, **kwargs):
        pytest.fail("oversized raster reached external execution")

    monkeypatch.setattr(subprocess, "run", forbidden)
    with pytest.raises(RenderingError, match="pixel budget"):
        _rasterize(source, dpi=180, max_pixels=100)


def test_changed_font_identity_fails_before_external_execution(monkeypatch):
    source = _svg()
    fonts = dict(source.diagnostics.fonts)
    fonts["sequence"] = replace(fonts["sequence"], sha256="0" * 64)
    source = replace(source, diagnostics=replace(source.diagnostics, fonts=MappingProxyType(fonts)))

    def forbidden(*args, **kwargs):
        pytest.fail("changed font reached external execution")

    monkeypatch.setattr(subprocess, "run", forbidden)
    with pytest.raises(RenderingError, match="font.*changed"):
        _rasterize(source)


@pytest.mark.parametrize("fonts", [None, {}, {"sequence": "Arial"}])
def test_incomplete_font_evidence_fails_before_external_execution(fonts, monkeypatch):
    source = replace(_svg(), diagnostics=replace(_svg().diagnostics, fonts=fonts))

    def forbidden(*args, **kwargs):
        pytest.fail("incomplete font identities reached external execution")

    monkeypatch.setattr(subprocess, "run", forbidden)
    with pytest.raises(RenderingError, match="font identities"):
        _rasterize(source)


@pytest.mark.parametrize(
    "kind,attrs",
    [
        ("image", {"href": "file:///private/example.png"}),
        ("script", {}),
        ("use", {"href": "https://example.invalid/graphic.svg#x"}),
    ],
)
def test_tampered_svg_cannot_load_external_or_active_content(kind, attrs, monkeypatch):
    source = _svg()
    root = ET.fromstring(source.svg)
    ET.SubElement(root, f"{{http://www.w3.org/2000/svg}}{kind}", attrs)
    source = replace(source, svg=ET.tostring(root, encoding="unicode"))

    def forbidden(*args, **kwargs):
        pytest.fail("unsafe SVG reached external execution")

    monkeypatch.setattr(subprocess, "run", forbidden)
    with pytest.raises(RenderingError, match="SVG"):
        _rasterize(source)


def test_missing_rasterizer_is_an_actionable_dependency_error():
    with pytest.raises(RenderingError, match="Inkscape.*unavailable"):
        _rasterize(_svg(), inkscape="missing-baserender-inkscape-command")


@pytest.mark.parametrize("failure", ["exit", "timeout", "not-png", "wrong-size"])
def test_converter_failures_do_not_return_an_artifact(failure, monkeypatch):
    source = _svg()

    def run(args, **kwargs):
        if "--version" in args:
            return subprocess.CompletedProcess(args, 0, b"Inkscape 1.4.3\n", b"")
        if failure == "timeout":
            raise subprocess.TimeoutExpired(args, kwargs["timeout"])
        if failure == "exit":
            return subprocess.CompletedProcess(args, 2, b"", b"intentional export failure")
        payload = b"not a PNG"
        if failure == "wrong-size":
            output = io.BytesIO()
            Image.new("RGBA", (8, 8)).save(output, format="PNG")
            payload = output.getvalue()
        return subprocess.CompletedProcess(args, 0, payload, b"")

    monkeypatch.setattr(shutil, "which", lambda _value: __file__)
    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(RenderingError, match="Inkscape|PNG|dimensions"):
        _rasterize(source)
