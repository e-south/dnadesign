"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/tests/test_vector_output.py

Public native SVG illustration and independent-caption contract tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from dataclasses import replace
from hashlib import sha256
from xml.etree import ElementTree as ET

import matplotlib.pyplot as plt
import pytest

from dnadesign.baserender import (
    Effect,
    Feature,
    IllustrationCaption,
    Record,
    RenderingError,
    Span,
    SvgArtwork,
    SvgIllustration,
    render_record_svg,
)

SVG = "{http://www.w3.org/2000/svg}"
STYLE = {"legend": False, "font_size_seq": 14, "show_pair_rungs": False}


def example(length=60):
    return Record(
        id=f"vector-{length}",
        alphabet="DNA",
        sequence=("ACGT" * 25)[:length],
        features=(
            Feature("left", "kmer", Span(4, 10, "fwd"), "ACGTAC"),
            Feature("right", "kmer", Span(28, 34, "fwd"), "ACGTAC"),
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
        meta={"fixed_content_top_extent_px": 260},
    )


@pytest.mark.parametrize("length", [60, 100])
def test_public_svg_preserves_vector_and_editable_caption(length):
    before = plt.get_fignums()
    result = render_record_svg(example(length), style_overrides=STYLE)
    root = ET.fromstring(result.svg)
    assert not root.findall(f".//{SVG}image")
    body = root.find(f".//{SVG}g[@id='br-vector-0']/{SVG}svg")
    assert len(body.findall(f".//{SVG}path")) == 12
    assert any(node.text == "RNAP" for node in root.findall(f".//{SVG}text"))
    assert result.width_pt > 0 and result.height_pt > 0
    assert result.diagnostics.base_pitch_pt > 0
    assert result.diagnostics.illustrations[0].source_sha256
    assert plt.get_fignums() == before


def test_caption_style_and_visibility_do_not_scale_or_move_body():
    first = render_record_svg(example(), style_overrides=STYLE)
    second = render_record_svg(
        example(),
        style_overrides=STYLE,
        captions={
            "rnap_sigma70": IllustrationCaption(
                "RNAP (σ⁷⁰)",
                family="DejaVu Sans Mono",
                size_pt=18,
                offset_px=(8, -3),
                anchor_fraction=(0.83827, 0.55688),
            )
        },
    )
    assert first.diagnostics.illustrations[0].bounds_pt == second.diagnostics.illustrations[0].bounds_pt
    assert first.diagnostics.illustrations[0].anchors_pt == second.diagnostics.illustrations[0].anchors_pt
    caption = ET.fromstring(second.svg).find(f".//{SVG}text[@data-baserender-caption]")
    assert caption.text == "RNAP (σ⁷⁰)"
    assert caption.get("font-size") == "18"
    assert caption.get("font-family") == "DejaVu Sans Mono"
    first_caption = ET.fromstring(first.svg).find(f".//{SVG}text[@data-baserender-caption]")
    assert float(caption.get("x")) - float(first_caption.get("x")) == pytest.approx(3.2, abs=1e-3)
    assert float(caption.get("y")) - float(first_caption.get("y")) == pytest.approx(-1.2, abs=1e-3)
    hidden = render_record_svg(
        example(),
        style_overrides=STYLE,
        captions={"rnap_sigma70": IllustrationCaption("RNAP", visible=False)},
    )
    assert ET.fromstring(hidden.svg).find(f".//{SVG}text[@data-baserender-caption]") is None


def artwork(tmp_path, body=None):
    path = tmp_path / "art.svg"
    path.write_text(
        body
        or """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 40">
      <defs><linearGradient id="g"><stop stop-color="red"/><stop offset="1" stop-color="blue"/></linearGradient>
      <clipPath id="c"><rect width="100" height="40"/></clipPath><image id="unused" href="old.png"/></defs>
      <g transform="translate(2 1)"><g id="body" clip-path="url(#c)">
      <path d="M0 0H90V30H0Z" fill="url(#g)"/></g></g></svg>"""
    )
    return SvgArtwork(
        path,
        sha256(path.read_bytes()).hexdigest(),
        ("body",),
        (0, 0, 100, 40),
        {"upstream": (0.25, 0.9), "downstream": (0.5, 0.9)},
    )


def test_pinned_caller_selection_preserves_transforms_defs_and_unique_instance_ids(tmp_path):
    art = artwork(tmp_path)
    record = example()
    record = replace(record, effects=record.effects * 2)
    result = render_record_svg(record, style_overrides=STYLE, illustrations={"rnap_sigma70": SvgIllustration(art)})
    root = ET.fromstring(result.svg)
    ids = [node.get("id") for node in root.iter() if node.get("id")]
    assert len(ids) == len(set(ids))
    assert len(root.findall(f".//{SVG}linearGradient")) == 2
    assert len(root.findall(f".//{SVG}clipPath")) >= 2
    assert root.find(f'.//{SVG}g[@transform="translate(2 1)"]') is not None
    assert not root.findall(f".//{SVG}image")


@pytest.mark.parametrize("problem", ["hash", "missing", "duplicate", "external", "active", "raster"])
def test_bad_vector_sources_fail_before_figure_allocation(tmp_path, problem):
    art = artwork(tmp_path)
    if problem == "hash":
        art = replace(art, sha256="0" * 64)
    elif problem == "missing":
        art = replace(art, element_ids=("absent",))
    elif problem == "duplicate":
        art = artwork(tmp_path, '<svg xmlns="http://www.w3.org/2000/svg"><g id="body"/><g id="body"/></svg>')
    elif problem == "external":
        art = artwork(
            tmp_path, '<svg xmlns="http://www.w3.org/2000/svg"><g id="body"><use href="other.svg#x"/></g></svg>'
        )
    elif problem == "active":
        art = artwork(tmp_path, '<svg xmlns="http://www.w3.org/2000/svg"><g id="body" onclick="alert(1)"/></svg>')
    else:
        art = artwork(
            tmp_path,
            '<svg xmlns="http://www.w3.org/2000/svg"><g id="body"><image href="data:image/png;base64,AAAA"/></g></svg>',
        )
    before = plt.get_fignums()
    with pytest.raises(RenderingError):
        render_record_svg(example(), style_overrides=STYLE, illustrations={"rnap_sigma70": SvgIllustration(art)})
    assert plt.get_fignums() == before


def test_missing_font_and_unknown_asset_key_fail_before_allocating_figure():
    before = plt.get_fignums()
    with pytest.raises(RenderingError, match="font"):
        render_record_svg(
            example(),
            style_overrides=STYLE,
            captions={"rnap_sigma70": IllustrationCaption("RNAP", family="NonexistentFontXYZ")},
        )
    with pytest.raises(RenderingError, match="unused"):
        render_record_svg(example(), captions={"unused": IllustrationCaption("wrong")})
    assert plt.get_fignums() == before


def test_native_output_never_decodes_a_raster_asset(monkeypatch):
    import matplotlib.image

    def reject_pixels(*args, **kwargs):
        raise AssertionError("native SVG must not load pixels")

    monkeypatch.setattr(matplotlib.image, "imread", reject_pixels)
    result = render_record_svg(example(), style_overrides=STYLE)
    root = ET.fromstring(result.svg)
    patch = root.find(f".//{SVG}g[@id='patch_1']/{SVG}path")
    assert "fill: none" in patch.get("style")
    nodes = list(root.iter())
    slot = next(i for i, node in enumerate(nodes) if node.get("id") == "br-vector-0")
    footprint = next(
        i for i, node in enumerate(nodes) if (node.get("id") or "").startswith("anchored_illustration_footprint:")
    )
    assert footprint < slot


@pytest.mark.parametrize(
    "content",
    [
        '<g id="body" style="fill:URL(https://example.invalid/x)"/>',
        '<g id="body" xml:base="https://example.invalid/"><use href="#x"/></g><path id="x"/>',
        '<g id="body"><use href="#x"/></g><g id="x"><use href="#x"/></g>',
        '<g id="body"><path fill="url(#absent)"/></g>',
        '<g id="body">' + "<g>" * 130 + "<path/>" + "</g>" * 130 + "</g>",
    ],
)
def test_unsupported_references_and_depth_fail_before_allocation(tmp_path, content):
    source = artwork(tmp_path, '<svg xmlns="http://www.w3.org/2000/svg">' + content + "</svg>")
    before = plt.get_fignums()
    with pytest.raises(RenderingError):
        render_record_svg(example(), style_overrides=STYLE, illustrations={"rnap_sigma70": SvgIllustration(source)})
    assert plt.get_fignums() == before


@pytest.mark.parametrize("size", [0, -1, True, float("nan"), float("inf")])
def test_bad_caption_units_fail_before_allocation(size):
    before = plt.get_fignums()
    with pytest.raises(RenderingError, match="size_pt"):
        render_record_svg(
            example(), style_overrides=STYLE, captions={"rnap_sigma70": IllustrationCaption("RNAP", size_pt=size)}
        )
    assert plt.get_fignums() == before


def test_raster_figure_rejects_vector_source_before_allocation(tmp_path):
    from dnadesign.baserender import render_record_figure

    record = example()
    effect = replace(record.effects[0], params={**record.effects[0].params, "svg_asset": artwork(tmp_path)})
    before = plt.get_fignums()
    with pytest.raises(RenderingError, match="render_record_svg"):
        render_record_figure(replace(record, effects=(effect,)), style_overrides=STYLE)
    assert plt.get_fignums() == before


def test_svg_is_repeatable_with_stable_pitch_and_strand_anchors():
    first = render_record_svg(example(60), style_overrides=STYLE)
    again = render_record_svg(example(60), style_overrides=STYLE)
    longer = render_record_svg(example(100), style_overrides=STYLE)
    assert first.svg == again.svg
    assert first.diagnostics.base_pitch_pt == longer.diagnostics.base_pitch_pt
    assert first.diagnostics.strand_anchors_pt == longer.diagnostics.strand_anchors_pt


def test_caller_owned_vector_does_not_require_a_packaged_asset_id(tmp_path):
    record = example()
    effect = replace(record.effects[0], params={"asset_id": "custom-vector", "width_px": 300})
    result = render_record_svg(
        replace(record, effects=(effect,)),
        style_overrides=STYLE,
        illustrations={"custom-vector": SvgIllustration(artwork(tmp_path))},
    )
    assert result.diagnostics.illustrations[0].asset_id == "custom-vector"


def test_stylesheet_sources_are_rejected_rather_than_losing_styles(tmp_path):
    source = artwork(
        tmp_path,
        '<svg xmlns="http://www.w3.org/2000/svg"><style>path{fill:red}</style>'
        '<g id="body"><path d="M0 0H10V10Z"/></g></svg>',
    )
    with pytest.raises(RenderingError, match="stylesheets"):
        render_record_svg(example(), style_overrides=STYLE, illustrations={"rnap_sigma70": SvgIllustration(source)})


def test_legacy_corrupt_pixels_still_fail_before_figure_allocation(tmp_path, monkeypatch):
    import json

    from dnadesign.baserender import render_record_figure
    from dnadesign.baserender.src.render.effects import anchored_illustration as loader

    payload = json.loads((loader._ASSET_ROOT / "rnap_sigma70.json").read_text())
    payload.update(asset_id="corrupt-pixels", image="broken.png")
    (tmp_path / "corrupt-pixels.json").write_text(json.dumps(payload))
    (tmp_path / "broken.png").write_bytes(b"not a PNG")
    monkeypatch.setattr(loader, "_ASSET_ROOT", tmp_path)
    record = example()
    effect = replace(record.effects[0], params={**record.effects[0].params, "asset_id": "corrupt-pixels"})
    before = plt.get_fignums()
    try:
        with pytest.raises(RenderingError, match="image"):
            render_record_figure(replace(record, effects=(effect,)), style_overrides=STYLE)
        assert plt.get_fignums() == before
    finally:
        loader._load_asset.cache_clear()
        loader._load_pixels.cache_clear()
        for number in set(plt.get_fignums()) - set(before):
            plt.close(number)


def test_missing_illustration_binding_cannot_silently_drop_its_connector():
    record = example()
    bindings = [dict(item) for item in record.effects[0].target["bindings"]]
    bindings[0]["feature_id"] = "absent-feature"
    effect = replace(record.effects[0], target={"bindings": bindings})
    before = plt.get_fignums()
    with pytest.raises(RenderingError, match="absent-feature"):
        render_record_svg(replace(record, effects=(effect,)), style_overrides=STYLE)
    assert plt.get_fignums() == before
