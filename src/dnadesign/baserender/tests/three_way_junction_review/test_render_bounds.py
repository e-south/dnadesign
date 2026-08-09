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
from matplotlib.collections import LineCollection

import dnadesign.baserender as baserender
from dnadesign.baserender.src.config import Style
from dnadesign.baserender.src.outputs.names import _safe_stem
from dnadesign.baserender.src.render import three_way_junction_review as review_renderer_module
from dnadesign.baserender.src.render.junction_pairing_layout import sequence_chunks
from dnadesign.baserender.src.render.sequence_preview import bounded_sequence_preview

from .fixtures import (
    _adapt_payload,
    _payload,
    _payload_with_large_display_scalars,
    _payload_with_long_junction_sequences,
    _payload_with_long_recovery_primers,
    _payload_with_many_junctions,
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


def _rendered_text(payload: dict[str, object]) -> tuple[object, str]:
    figure = baserender.render(_adapt_payload(payload), renderer="three_way_junction_review")
    text = "\n".join(item.get_text() for item in figure.axes[0].texts)
    return figure, text


def _spaced(sequence: str) -> str:
    return " ".join(sequence)


def test_review_renderer_preserves_every_long_primer_base() -> None:
    payload = _payload_with_long_recovery_primers()
    figure, text = _rendered_text(payload)
    try:
        for direction in ("forward", "reverse"):
            order = payload["recovery"][direction]["order_sequence_5to3"]  # type: ignore[index]
            for chunk in sequence_chunks(order):
                assert _spaced(chunk.sequence) in text
        assert "SHA-256[:12]" not in text
        assert "preview" not in text
    finally:
        plt.close(figure)


def test_review_renderer_preserves_every_long_junction_base_and_pairing_edge() -> None:
    payload = _payload_with_long_junction_sequences()
    figure, text = _rendered_text(payload)
    try:
        junction = payload["geometry"]["junctions"][0]  # type: ignore[index]
        displayed = (
            junction["toehold"],
            junction["toehold_complement"][::-1],
            junction["barcode"],
            junction["barcode_complement"][::-1],
        )
        for sequence in displayed:
            for chunk in sequence_chunks(sequence):
                assert _spaced(chunk.sequence) in text
        pair_collections = [item for item in figure.axes[0].collections if isinstance(item, LineCollection)]
        expected_pair_count = len(junction["toehold"]) + len(junction["barcode"])
        assert sum(len(item.get_segments()) for item in pair_collections) >= expected_pair_count
    finally:
        plt.close(figure)


def test_review_renderer_bounds_identifiers_and_omits_unhelpful_search_scalars() -> None:
    payload = _payload_with_large_display_scalars()
    figure, text = _rendered_text(payload)
    try:
        assert "10007 chars" in text
        assert "SHA-256[:12]" in text
        assert payload["target"]["target_id"] not in text  # type: ignore[index]
        assert ("1" + ("0" * 10_000)) not in text
        assert "toehold paths" not in text
        assert "barcode candidates" not in text
    finally:
        plt.close(figure)


def test_review_renderer_escapes_control_characters_in_adapter_identifiers() -> None:
    payload = _payload()
    identifier = "assembly\n\t" * 100
    payload["target"]["target_id"] = identifier  # type: ignore[index]
    payload["target"]["assembly_group_id"] = identifier  # type: ignore[index]
    payload["search"]["assembly_group_id"] = identifier  # type: ignore[index]
    payload["checks"][0]["subject"]["id"] = identifier  # type: ignore[index]
    payload["checks"][1]["subject"]["id"] = identifier  # type: ignore[index]
    figure, _ = _rendered_text(payload)
    try:
        for item in figure.axes[0].texts:
            assert "\n" not in item.get_text()
            assert "\t" not in item.get_text()
    finally:
        plt.close(figure)


def test_review_renderer_keeps_artist_counts_linear_without_one_text_artist_per_base() -> None:
    payload = _payload_with_many_junctions()
    figure, text = _rendered_text(payload)
    try:
        axis = figure.axes[0]
        target_length = len(payload["target"]["sequence_5to3"])  # type: ignore[index]
        junction_count = len(payload["geometry"]["junctions"])  # type: ignore[index]

        assert f"J{junction_count:02d}" in text
        assert len(axis.texts) < target_length + (junction_count * 32)
        assert len(axis.patches) <= (junction_count * 2) + 1
        assert all(isinstance(item, LineCollection) for item in axis.collections)
    finally:
        plt.close(figure)


def test_large_review_uses_the_compact_annealed_fragment_stage() -> None:
    payload = _payload_with_many_junctions()
    figure, text = _rendered_text(payload)
    try:
        fragment_count = len(payload["strands"])
        assert f"Compact view for {fragment_count} fragments" in text
        assert "exact order strands, junction pairs, and recovered duplex remain below" in text
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


def test_review_renderer_sizes_the_canvas_from_exact_sequence_content() -> None:
    short = review_renderer_module._review_from_record(_adapt_payload(_payload()))
    long = review_renderer_module._review_from_record(_adapt_payload(_payload_with_many_junctions()))

    short_size = review_renderer_module._review_figure_size(Style(), short)
    long_size = review_renderer_module._review_figure_size(Style(), long)

    assert short_size[0] == pytest.approx(15.2)
    assert short_size[1] >= 6.4
    assert long_size[0] == pytest.approx(15.2)
    assert long_size[1] > short_size[1]
