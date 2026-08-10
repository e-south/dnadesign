"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/tests/three_way_junction_review/test_assembly_process.py

Molecular truth and scaling tests for Junction review figures.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

import matplotlib.pyplot as plt
import pytest

import dnadesign.baserender as baserender
from dnadesign.baserender.src.config import ImagesOutputCfg
from dnadesign.baserender.src.outputs.images import write_images
from dnadesign.baserender.src.render import junction_three_way_assembly as assembly_renderer
from dnadesign.baserender.src.render.junction_review.primitives import (
    draw_molecular_path,
    draw_segmented_strand,
)
from dnadesign.junction import parse_request, plan
from dnadesign.junction.presentation import review_contracts
from dnadesign.junction.tests.scenarios.factories import scale_request_mapping

from .fixtures import _adapt_payload, _payload


@cache
def _real_records(*, target_count: int, target_length: int):
    request = parse_request(
        scale_request_mapping(
            target_count=target_count,
            target_length=target_length,
            topology="shared",
            nominal_fragment_oligo_length=132,
            search_range=2,
            barcode_generation_attempts=250_000,
        )
    )
    reviews = review_contracts(plan(request))
    return baserender.adapt_records(
        [review.model_dump(mode="json") for review in reviews],
        adapter_kind="three_way_junction_review_v1",
    )


def _artists_with_prefix(axis, prefix: str):
    return tuple(artist for artist in axis.texts if (artist.get_gid() or "").startswith(prefix))


def _ordered_base_text(axis, prefix: str) -> str:
    bases: list[tuple[int, float, str]] = []
    for artist in axis.collections:
        gid = artist.get_gid() or ""
        if not gid.startswith(prefix) or ":glyph:" not in gid:
            continue
        window = int(gid.split(":window:", 1)[1].split(":", 1)[0])
        base = gid.rsplit(":glyph:", 1)[1]
        bases.extend((window, float(offset[0]), base) for offset in artist.get_offsets())
    return "".join(base for _window, _x, base in sorted(bases))


def test_molecular_primitives_use_square_termini_and_rounded_bends() -> None:
    figure, axis = plt.subplots()
    try:
        draw_segmented_strand(
            axis,
            start_x=0.0,
            center_y=0.0,
            base_step=1.0,
            length=4,
            segments=((0, 2, "#eeeeee"), (2, 4, "#dddddd")),
            height=0.2,
            gid_prefix="test-strand",
        )
        draw_molecular_path(axis, (0.0, 1.0, 1.0), (1.0, 1.0, 2.0), color="#dddddd", gid="test-path")

        body = next(patch for patch in axis.patches if patch.get_gid() == "test-strand:body")
        outline = next(patch for patch in axis.patches if patch.get_gid() == "test-strand:outline")
        line = next(item for item in axis.lines if item.get_gid() == "test-path")
        assert body.__class__.__name__ == "Rectangle"
        assert outline.__class__.__name__ == "Rectangle"
        assert line.get_solid_capstyle() == "butt"
        assert line.get_solid_joinstyle() == "round"
    finally:
        plt.close(figure)


def test_fragment_and_detail_typography_is_centered_legible_and_collision_free() -> None:
    record = _adapt_payload(_payload())
    annealed = baserender.render(record, renderer="junction_annealed_fragments")
    detail = baserender.render(
        record,
        renderer="junction_three_way_assembly",
        options={"view": "junction_detail", "junction_ids": ["junction-01"]},
    )
    try:
        annealed_axis = annealed.axes[0]
        title = next(text for text in annealed_axis.texts if "expected to anneal" in text.get_text())
        bases = _artists_with_prefix(annealed_axis, "junction-annealed:")
        assert title.get_position()[0] == pytest.approx(0.5)
        assert title.get_ha() == "center"
        assert min(text.get_fontsize() for text in bases if ":base:" in (text.get_gid() or "")) >= 9.0

        annealed.canvas.draw()
        top_bases = sorted(
            _artists_with_prefix(annealed_axis, "junction-annealed:fragment-01:top:base:"),
            key=lambda artist: artist.get_position()[0],
        )
        boxes = [artist.get_window_extent(annealed.canvas.get_renderer()) for artist in top_bases]
        assert all(left.x1 <= right.x0 + 0.5 for left, right in zip(boxes, boxes[1:], strict=False))

        figure_titles = [text for text in detail.texts if "selected three-way junction" in text.get_text()]
        assert len(figure_titles) == 1
        assert figure_titles[0].get_position()[0] == pytest.approx(0.5)
        assert figure_titles[0].get_ha() == "center"
        detail_bases = [
            text
            for text in detail.axes[0].texts
            if (text.get_gid() or "").startswith("junction:junction-01:") and ":base:" in (text.get_gid() or "")
        ]
        assert min(text.get_fontsize() for text in detail_bases) >= 9.0
    finally:
        plt.close(annealed)
        plt.close(detail)


def test_assembly_view_shows_orders_three_way_state_and_exact_recovered_duplex() -> None:
    [record] = _real_records(target_count=1, target_length=1_000)
    review = record.meta["three_way_junction_review"]
    assert len(review["geometry"]["junctions"]) == 10

    figure = baserender.render(
        record,
        renderer="junction_three_way_assembly",
        options={"view": "assembly"},
    )
    try:
        axis = figure.axes[0]
        assert axis.get_gid() == "junction-three-way-assembly:assembly"
        text = "\n".join(item.get_text() for item in axis.texts)
        assert "The oligos remain separate before annealing" in text
        assert "The plan specifies an annealed pre-ligation state" in text
        assert "The expected PCR product is a recovered duplex" in text
        assert "does not establish annealing, ligation, amplification, or yield" in text
        expected_top = review["recovery"]["extended_top_sequence_5to3"]
        assert _ordered_base_text(axis, "junction-three-way-assembly:product:top:window:") == expected_top
        expected_bottom = expected_top.translate(str.maketrans("ACGT", "TGCA"))
        assert _ordered_base_text(axis, "junction-three-way-assembly:product:bottom:window:") == expected_bottom
        product_pairs = [
            collection
            for collection in axis.collections
            if (collection.get_gid() or "").startswith("junction-three-way-assembly:product:window:")
            and (collection.get_gid() or "").endswith(":pairs")
        ]
        assert sum(len(collection.get_segments()) for collection in product_pairs) == len(expected_top)
        assert not _artists_with_prefix(axis, "junction-three-way-assembly:product:top:window:")
        product_glyphs = [
            collection
            for collection in axis.collections
            if (collection.get_gid() or "").startswith("junction-three-way-assembly:product:top:window:")
            and ":glyph:" in (collection.get_gid() or "")
        ]
        assert product_glyphs
        assert all(
            collection.get_transform().__class__.__name__ == "IdentityTransform" for collection in product_glyphs
        )
        assert _artists_with_prefix(axis, "junction-three-way-assembly:orders:")
        assert _artists_with_prefix(axis, "junction-three-way-assembly:three-way:")

        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        by_gid = {artist.get_gid(): artist for artist in axis.texts if artist.get_gid()}
        order_title = by_gid["junction-three-way-assembly:orders:title"].get_window_extent(renderer)
        order_labels = [
            artist.get_window_extent(renderer)
            for gid, artist in by_gid.items()
            if gid.startswith("junction-three-way-assembly:orders:") and gid.endswith(":label")
        ]
        orientation = by_gid["junction-three-way-assembly:orders:orientation"].get_window_extent(renderer)
        annealing = by_gid["junction-three-way-assembly:transition:annealing"].get_window_extent(renderer)
        three_way_title = by_gid["junction-three-way-assembly:three-way:title"].get_window_extent(renderer)
        assert all(not order_title.overlaps(label) for label in order_labels)
        assert not orientation.overlaps(annealing)
        assert not annealing.overlaps(three_way_title)
    finally:
        plt.close(figure)


def test_default_assembly_view_is_not_a_parallel_overview_contract() -> None:
    record = _adapt_payload(_payload())
    figure = baserender.render(record, renderer="junction_three_way_assembly")
    try:
        assert figure.axes[0].get_gid() == "junction-three-way-assembly:assembly"
    finally:
        plt.close(figure)

    with pytest.raises(baserender.SchemaError, match="must be 'assembly' or 'junction_detail'"):
        baserender.render(
            record,
            renderer="junction_three_way_assembly",
            options={"view": "overview"},
        )


def test_oversized_recovered_duplex_rejects_before_figure_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    [record] = _real_records(target_count=1, target_length=10_000)

    def fail_if_allocated(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("oversized molecular view must reject before figure allocation")

    monkeypatch.setattr(assembly_renderer.plt, "subplots", fail_if_allocated)
    with pytest.raises(baserender.SchemaError, match="recovered-duplex limit"):
        baserender.render(
            record,
            renderer="junction_three_way_assembly",
            options={"view": "assembly"},
        )


def test_one_pot_multi_target_review_writes_one_named_figure_per_target(tmp_path: Path) -> None:
    records = _real_records(target_count=3, target_length=360)
    assert [len(record.meta["three_way_junction_review"]["geometry"]["junctions"]) for record in records] == [
        3,
        3,
        3,
    ]
    style = baserender.resolve_style(preset=None, overrides=None)
    output = ImagesOutputCfg(kind="images", dir=tmp_path / "images", path=None, fmt="svg")

    result = write_images(
        records,
        output=output,
        renderer_name="junction_three_way_assembly",
        style=style,
        palette=baserender.Palette(style.palette),
        renderer_options={"view": "assembly"},
    )

    paths = sorted(result.glob("*.svg"))
    assert [path.stem for path in paths] == [record.id for record in records]
    for path, record in zip(paths, records, strict=True):
        payload = path.read_text(encoding="utf-8")
        assert record.id in payload
        assert all(other_id not in payload for other_id in other_target_ids(record.id, records))


def other_target_ids(current: str, records) -> set[str]:
    return {record.id for record in records if record.id != current}
