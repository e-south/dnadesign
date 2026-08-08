"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/tests/three_way_junction_review/test_render_bounds.py

Resource and layout bounds for Junction review rendering.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib

import matplotlib.pyplot as plt
import pytest

import dnadesign.baserender as baserender
from dnadesign.baserender.src.config import Style
from dnadesign.baserender.src.outputs.names import _safe_stem
from dnadesign.baserender.src.render import three_way_junction_review as review_renderer_module
from dnadesign.baserender.src.render.sequence_preview import bounded_sequence_preview

from .fixtures import (
    _adapt_payload,
    _payload,
    _payload_with_large_display_scalars,
    _payload_with_long_junction_sequences,
    _payload_with_long_recovery_primers,
    _payload_with_many_junctions,
    _rename_target_geometry,
)


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


def test_bounded_review_image_stems_discard_unbounded_edge_runs() -> None:
    assert _safe_stem("x" + ("." * 10_000)) == "x"
    assert _safe_stem("!" * 10_000) == "record"


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
    assert len(preview.label("bind")) < 64


def test_review_renderer_bounds_long_primer_sequences_with_explicit_preview_metadata() -> None:
    payload = _payload_with_long_recovery_primers()
    record = _adapt_payload(payload)

    figure = baserender.render(record, renderer="three_way_junction_review")
    try:
        figure.canvas.draw()
        recovery_axis = figure.axes[2]
        renderer = figure.canvas.get_renderer()
        axis_box = recovery_axis.get_window_extent(renderer=renderer)
        recovery_text = "\n".join(item.get_text() for item in recovery_axis.texts)

        assert "96 nt" in recovery_text
        assert "100 nt" in recovery_text
        assert "SHA-256[:12]" in recovery_text
        assert "preview" in recovery_text
        assert payload["recovery"]["forward"]["binding_sequence_5to3"] not in recovery_text
        for artist in recovery_axis.texts:
            artist_box = artist.get_window_extent(renderer=renderer)
            assert artist_box.x0 >= axis_box.x0
            assert artist_box.x1 <= axis_box.x1
    finally:
        plt.close(figure)


def test_review_renderer_bounds_long_junction_sequences_with_explicit_preview_metadata() -> None:
    payload = _payload_with_long_junction_sequences()
    record = _adapt_payload(payload)

    figure = baserender.render(record, renderer="three_way_junction_review")
    try:
        figure.canvas.draw()
        junction_axis = figure.axes[1]
        renderer = figure.canvas.get_renderer()
        axis_box = junction_axis.get_window_extent(renderer=renderer)
        junction_text = "\n".join(item.get_text() for item in junction_axis.texts)

        assert "T = toehold" in junction_text
        assert "B = matched barcode" in junction_text
        assert "100 nt" in junction_text
        assert "digest = SHA-256[:12]" in junction_text
        assert payload["geometry"]["junctions"][0]["toehold"] not in junction_text
        assert payload["geometry"]["junctions"][0]["barcode"] not in junction_text
        for artist in junction_axis.texts:
            artist_box = artist.get_window_extent(renderer=renderer)
            assert artist_box.x0 >= axis_box.x0
            assert artist_box.x1 <= axis_box.x1
    finally:
        plt.close(figure)


def test_review_renderer_bounds_long_ids_and_integer_metrics() -> None:
    payload = _payload_with_large_display_scalars()
    record = _adapt_payload(payload)

    figure = baserender.render(record, renderer="three_way_junction_review")
    try:
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        text = "\n".join(item.get_text() for axis in figure.axes for item in axis.texts)

        assert "10007 chars" in text
        assert "10009 chars" in text
        assert "10001 digits" in text
        assert "SHA-256[:12]" in text
        assert payload["target"]["target_id"] not in text
        assert ("1" + ("0" * 10_000)) not in text
        for axis in (figure.axes[0], figure.axes[3]):
            axis_box = axis.get_window_extent(renderer=renderer)
            for artist in axis.texts:
                artist_box = artist.get_window_extent(renderer=renderer)
                assert artist_box.x0 >= axis_box.x0
                assert artist_box.x1 <= axis_box.x1
    finally:
        plt.close(figure)


def test_review_renderer_bounds_producer_valid_wide_identifiers() -> None:
    payload = _payload()
    identifier = "W" * 32
    payload["target"]["target_id"] = identifier
    payload["target"]["assembly_group_id"] = identifier
    payload["search"]["assembly_group_id"] = identifier
    payload["checks"][0]["subject"]["id"] = identifier
    payload["checks"][1]["subject"]["id"] = identifier
    record = _adapt_payload(payload)

    figure = baserender.render(record, renderer="three_way_junction_review")
    try:
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        text = "\n".join(item.get_text() for axis in figure.axes for item in axis.texts)
        assert "32 chars" in text
        assert identifier not in text
        for axis in (figure.axes[0], figure.axes[3]):
            axis_box = axis.get_window_extent(renderer=renderer)
            assert all(item.get_window_extent(renderer=renderer).x1 <= axis_box.x1 for item in axis.texts)
    finally:
        plt.close(figure)


def test_review_renderer_distinguishes_target_junctions_from_assembly_group_loci() -> None:
    first = _payload()
    second = _payload_with_long_recovery_primers()
    _rename_target_geometry(second, target_id="target-02")
    for payload in (first, second):
        payload["recovery"]["mode"] = "target_specific"
        payload["search"]["locus_count"] = 2
        payload["search"]["barcode_candidates_generated"] = 10
    record = baserender.adapt_records(
        [first, second],
        adapter_kind="three_way_junction_review_v1",
    )[0]

    figure = baserender.render(record, renderer="three_way_junction_review")
    try:
        junction_text = "\n".join(item.get_text() for item in figure.axes[1].texts)
        search_text = "\n".join(item.get_text() for item in figure.axes[3].texts)

        assert "1 target junction · every target junction shown" in junction_text
        assert "assembly-group loci  2" in search_text
    finally:
        plt.close(figure)


def test_review_renderer_escapes_control_characters_in_public_adapter_identifiers() -> None:
    payload = _payload()
    identifier = "assembly\n\t" * 100
    payload["target"]["target_id"] = identifier
    payload["target"]["assembly_group_id"] = identifier
    payload["search"]["assembly_group_id"] = identifier
    payload["checks"][0]["subject"]["id"] = identifier
    payload["checks"][1]["subject"]["id"] = identifier
    record = _adapt_payload(payload)

    figure = baserender.render(record, renderer="three_way_junction_review")
    try:
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        for axis in (figure.axes[0], figure.axes[3]):
            axis_box = axis.get_window_extent(renderer=renderer)
            for item in axis.texts:
                assert "\n" not in item.get_text()
                assert "\t" not in item.get_text()
                assert item.get_window_extent(renderer=renderer).x1 <= axis_box.x1
    finally:
        plt.close(figure)


def test_review_renderer_bounds_geometry_artist_counts() -> None:
    record = _adapt_payload(_payload_with_many_junctions())

    figure = baserender.render(record, renderer="three_way_junction_review")
    try:
        geometry_axis, junction_axis = figure.axes[:2]
        geometry_text = "\n".join(item.get_text() for item in geometry_axis.texts)
        junction_text = "\n".join(item.get_text() for item in junction_axis.texts)

        assert "15 more fragment pairs" in geometry_text
        assert "bounded target-junction preview" in junction_text
        assert "every target junction shown" not in junction_text
        assert "14 more junctions" in junction_text
        assert len(geometry_axis.patches) <= 12
        assert len(geometry_axis.texts) <= 22
        assert len(junction_axis.lines) <= 6
        assert len(junction_axis.texts) <= 18
    finally:
        plt.close(figure)


@pytest.mark.parametrize(
    ("style_overrides", "message"),
    [
        ({"dpi": 10**12}, "three_way_junction_review style.dpi exceeds the renderer limit"),
        ({"figure_scale": 10**12}, "three_way_junction_review style.figure_scale exceeds the renderer limit"),
        (
            {"dpi": 300, "figure_scale": 2.0},
            "three_way_junction_review canvas exceeds the 64 MiB RGBA allocation limit",
        ),
    ],
)
def test_review_renderer_rejects_unsafe_canvas_styles_before_figure_allocation(
    monkeypatch: pytest.MonkeyPatch,
    style_overrides: dict[str, object],
    message: str,
) -> None:
    record = _adapt_payload(_payload())

    def fail_if_allocated(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("unsafe review styles must reject before figure allocation")

    monkeypatch.setattr(review_renderer_module.plt, "subplots", fail_if_allocated)

    with pytest.raises(baserender.SchemaError, match=f"^{message}$"):
        baserender.render(
            record,
            renderer="three_way_junction_review",
            style={"overrides": style_overrides},
        )


@pytest.mark.parametrize(
    ("style", "expected"),
    [
        (Style(), (15.2, 4.2)),
        (Style(dpi=300, figure_scale=1.6), (24.32, 6.72)),
    ],
)
def test_review_renderer_accepts_normal_documented_canvas_styles(
    style: Style,
    expected: tuple[float, float],
) -> None:
    assert review_renderer_module._review_figure_size(style) == pytest.approx(expected)
