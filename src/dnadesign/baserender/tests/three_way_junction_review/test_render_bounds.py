"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/tests/three_way_junction_review/test_render_bounds.py

Resource and selection bounds for Junction review rendering.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import io
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
import pytest
from matplotlib.collections import LineCollection

import dnadesign.baserender as baserender
from dnadesign.baserender.src.outputs.names import _safe_stem
from dnadesign.baserender.src.render import junction_three_way_assembly as assembly_renderer
from dnadesign.baserender.src.render.junction_review.primitives import draw_base_run
from dnadesign.baserender.src.render.sequence_preview import bounded_sequence_preview, bounded_svg_gid

from .fixtures import (
    _adapt_payload,
    _payload,
    _payload_with_large_display_scalars,
    _payload_with_long_junction_sequences,
    _payload_with_many_junctions,
    _rename_target_geometry,
)


def _base_text(axis, prefix: str) -> str:
    marker = f"{prefix}:base:"
    artists = [artist for artist in axis.texts if (artist.get_gid() or "").startswith(marker)]
    artists.sort(key=lambda artist: int((artist.get_gid() or "").split(marker, 1)[1].split(":", 1)[0]))
    return "".join(artist.get_text() for artist in artists)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (" Target A ", "Target_A"),
        ("../target", "target"),
        ("target ! value", "target_value"),
        ("a..b", "a..b"),
        ("___", "record"),
        ("a---", "a"),
        ("a/b", "a_b"),
        ("αtargetβ", "target"),
        ("foo.bar-1", "foo.bar-1"),
    ],
)
def test_bounded_review_image_stems_preserve_ordinary_sanitization(raw: str, expected: str) -> None:
    assert _safe_stem(raw) == expected


def test_review_image_stems_bound_long_ids_with_a_deterministic_digest() -> None:
    first = _safe_stem("target-" + ("x" * 10_000) + "-a")
    second = _safe_stem("target-" + ("x" * 10_000) + "-b")

    assert len(first.encode()) <= 120
    assert first.startswith("target-")
    assert first != second
    assert first == _safe_stem("target-" + ("x" * 10_000) + "-a")


def test_bounded_sequence_preview_has_a_fixed_sequence_text_budget() -> None:
    sequence = "ACGT" * 1_000
    preview = bounded_sequence_preview(sequence)

    assert preview.length_nt == 4_000
    assert preview.preview == "ACGTAC…GTACGT"
    assert preview.sha256_prefix == hashlib.sha256(sequence.encode()).hexdigest()[:12]
    assert sequence not in preview.label("bind")


def test_base_gids_hash_an_unbounded_identifier_once() -> None:
    figure, axis = plt.subplots()
    try:
        long_prefix = f"junction:{'x' * 10_000}:top"
        artists = draw_base_run(
            axis,
            "ACGT" * 8,
            start_x=0,
            start_y=0,
            delta_x=1,
            delta_y=0,
            gid_prefix=long_prefix,
            fontsize=5,
        )
        gids = tuple(str(artist.get_gid()) for artist in artists)
        prefixes = {gid.split(":base:", 1)[0] for gid in gids}

        assert prefixes == {bounded_svg_gid(long_prefix)}
        assert all(len(gid) < 80 for gid in gids)
        assert all(long_prefix not in gid for gid in gids)
    finally:
        plt.close(figure)


@pytest.mark.parametrize("unsafe_id", ["target-" + ("x" * 100_000), "target-\x00invalid"])
def test_exported_review_svgs_bound_and_escape_all_identifier_derived_gids(unsafe_id: str) -> None:
    payload = _payload()
    _rename_target_geometry(payload, target_id=unsafe_id)
    junction_id = payload["geometry"]["junctions"][0]["junction_id"]  # type: ignore[index]
    renderers = (
        ("junction_annealed_fragments", None),
        ("junction_three_way_assembly", {"view": "junction_detail", "junction_ids": [junction_id]}),
    )

    for renderer, options in renderers:
        figure = baserender.render(_adapt_payload(payload), renderer=renderer, options=options)
        try:
            buffer = io.BytesIO()
            figure.savefig(buffer, format="svg", metadata={"Date": None})
            svg = buffer.getvalue()
            ET.fromstring(svg)
            assert unsafe_id.encode("utf-8") not in svg
            assert len(svg) < 150_000
        finally:
            plt.close(figure)


def test_overview_bounds_identifiers_and_omits_search_scalars() -> None:
    payload = _payload_with_large_display_scalars()
    figure = baserender.render(_adapt_payload(payload), renderer="junction_three_way_assembly")
    try:
        text = "\n".join(item.get_text() for item in figure.axes[0].texts)
        assert "10007 chars" in text
        assert "SHA-256[:12]" in text
        assert payload["target"]["target_id"] not in text  # type: ignore[index]
        assert "toehold paths" not in text
        assert "barcode candidates" not in text
    finally:
        plt.close(figure)


def test_detail_preserves_every_junction_base_and_pairing_edge() -> None:
    payload = _payload_with_long_junction_sequences()
    junction = payload["geometry"]["junctions"][0]  # type: ignore[index]
    figure = baserender.render(
        _adapt_payload(payload),
        renderer="junction_three_way_assembly",
        options={"view": "junction_detail", "junction_ids": [junction["junction_id"]]},
    )
    try:
        axis = figure.axes[0]
        prefix = f"junction:{junction['junction_id']}"
        assert _base_text(axis, f"{prefix}:barcode-b") == junction["barcode"]
        assert _base_text(axis, f"{prefix}:barcode-b-star") == junction["barcode_complement"]
        assert _base_text(axis, f"{prefix}:toehold-top") == junction["toehold"]
        collections = [item for item in axis.collections if isinstance(item, LineCollection)]
        context_bases = sum(len(_base_text(axis, f"{prefix}:{role}")) for role in ("left-top", "right-top"))
        expected = len(junction["barcode"]) + len(junction["toehold"]) + context_bases
        assert sum(len(item.get_segments()) for item in collections) == expected
    finally:
        plt.close(figure)


def test_detail_rejects_excessive_base_glyphs_before_figure_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _payload_with_long_junction_sequences(toehold_length=120, barcode_length=150)
    junction = payload["geometry"]["junctions"][0]  # type: ignore[index]

    def fail_if_allocated(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("oversized detail workload must reject before figure allocation")

    monkeypatch.setattr(assembly_renderer.plt, "subplots", fail_if_allocated)
    with pytest.raises(baserender.SchemaError, match="requires 584 base glyphs; the per-junction limit is 512"):
        baserender.render(
            _adapt_payload(payload),
            renderer="junction_three_way_assembly",
            options={"view": "junction_detail", "junction_ids": [junction["junction_id"]]},
        )


def test_annealed_map_requires_fragment_selection_before_large_allocation() -> None:
    payload = _payload_with_many_junctions(junction_count=20)
    record = _adapt_payload(payload)

    with pytest.raises(baserender.SchemaError, match="requires render.options.fragment_ids"):
        baserender.render(record, renderer="junction_annealed_fragments")

    fragment_ids = [fragment["fragment_id"] for fragment in payload["geometry"]["fragments"][:2]]  # type: ignore[index]
    figure = baserender.render(
        record,
        renderer="junction_annealed_fragments",
        options={"fragment_ids": fragment_ids},
    )
    try:
        text = "\n".join(item.get_text() for item in figure.axes[0].texts)
        assert "2 selected fragment pairs" in text
        assert "F01" in text and "F02" in text
        assert "F03" not in text
    finally:
        plt.close(figure)


def test_overview_rejects_excessive_fragments_before_figure_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = _adapt_payload(_payload_with_many_junctions(junction_count=256))

    def fail_if_allocated(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("oversized overview must reject before figure allocation")

    monkeypatch.setattr(assembly_renderer.plt, "subplots", fail_if_allocated)
    with pytest.raises(baserender.SchemaError, match="contains 257 fragments; the overview limit is 256"):
        baserender.render(record, renderer="junction_three_way_assembly")


def test_renderer_options_reject_unknown_keys() -> None:
    with pytest.raises(baserender.RenderingError, match="received unknown options"):
        baserender.render(
            _adapt_payload(_payload()),
            renderer="junction_three_way_assembly",
            options={"surprise": True},
        )


def test_renderer_options_reject_non_string_keys() -> None:
    with pytest.raises(baserender.RenderingError, match="option keys must be strings"):
        baserender.render(
            _adapt_payload(_payload()),
            renderer="junction_three_way_assembly",
            options={1: "overview"},  # type: ignore[dict-item]
        )


@pytest.mark.parametrize(
    ("style_overrides", "message"),
    [
        ({"dpi": 10**12}, "style.dpi exceeds the renderer limit"),
        ({"figure_scale": 10**12}, "style.figure_scale exceeds the renderer limit"),
        ({"dpi": 300, "figure_scale": 2.0}, "canvas exceeds the 64 MiB RGBA allocation limit"),
    ],
)
def test_unsafe_canvas_styles_reject_before_figure_allocation(
    monkeypatch: pytest.MonkeyPatch,
    style_overrides: dict[str, object],
    message: str,
) -> None:
    def fail_if_allocated(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("unsafe styles must reject before figure allocation")

    monkeypatch.setattr(assembly_renderer.plt, "subplots", fail_if_allocated)
    with pytest.raises(baserender.SchemaError, match=message):
        baserender.render(
            _adapt_payload(_payload()),
            renderer="junction_three_way_assembly",
            style={"overrides": style_overrides},
        )
