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
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
from matplotlib.colors import to_rgba

import dnadesign.baserender as baserender
from dnadesign.baserender.src.config import ImagesOutputCfg
from dnadesign.baserender.src.outputs.images import write_images
from dnadesign.baserender.src.render import junction_three_way_assembly as assembly_renderer
from dnadesign.baserender.src.render.junction_review.assembly_geometry import (
    MOLECULE_TO_TRANSITION,
    PRODUCT_BASE_WIDTH_INCHES,
    STAGE_TITLE_TO_MOLECULE,
    TRANSITION_TO_STAGE_TITLE,
)
from dnadesign.baserender.src.render.junction_review.foundation import PRIMER_BINDING_SITE, review_from_record
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
        window = int(gid.split(":window:", 1)[1].split(":", 1)[0]) if ":window:" in gid else 0
        base = gid.rsplit(":glyph:", 1)[1]
        bases.extend((window, float(offset[0]), base) for offset in artist.get_offsets())
    return "".join(base for _window, _x, base in sorted(bases))


def _compact_base_text(axis, prefix: str, *, coordinate: int = 0, reverse: bool = False) -> str:
    bases: list[tuple[float, str]] = []
    for artist in axis.collections:
        gid = artist.get_gid() or ""
        if not gid.startswith(prefix) or ":glyph:" not in gid:
            continue
        base = gid.rsplit(":glyph:", 1)[1]
        bases.extend((float(offset[coordinate]), base) for offset in artist.get_offsets())
    return "".join(base for _position, base in sorted(bases, reverse=reverse))


def _collection_x_center(axis, prefix: str) -> float:
    positions = [
        float(offset[0])
        for collection in axis.collections
        if (collection.get_gid() or "").startswith(prefix) and ":glyph:" in (collection.get_gid() or "")
        for offset in collection.get_offsets()
    ]
    return (min(positions) + max(positions)) / 2


def _assert_texts_do_not_overlap(figure, texts) -> None:
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    boxes = tuple((artist, artist.get_window_extent(renderer)) for artist in texts if artist.get_text().strip())
    collisions = [
        (left.get_text(), right.get_text())
        for (left, left_box), (right, right_box) in combinations(boxes, 2)
        if left_box.overlaps(right_box)
    ]
    assert collisions == []


def _assert_annotations_clear_bases(figure, annotations, bases) -> None:
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    annotation_boxes = tuple(
        (artist, artist.get_window_extent(renderer)) for artist in annotations if artist.get_text().strip()
    )
    base_boxes = tuple((artist, artist.get_window_extent(renderer)) for artist in bases)
    collisions = [
        (annotation.get_text(), base.get_text())
        for annotation, annotation_box in annotation_boxes
        for base, base_box in base_boxes
        if annotation_box.overlaps(base_box)
    ]
    assert collisions == []


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
        title = next(text for text in annealed_axis.texts if "expected annealing" in text.get_text())
        bases = _artists_with_prefix(annealed_axis, "junction-annealed:")
        assert title.get_position()[0] == pytest.approx(0.5)
        assert title.get_ha() == "center"
        assert min(text.get_fontsize() for text in bases if ":base:" in (text.get_gid() or "")) >= 9.0
        annealed_text = "\n".join(text.get_text() for text in annealed_axis.texts)
        assert "forward primer-binding site" in annealed_text
        assert "reverse primer-binding site" in annealed_text
        assert any(patch.get_facecolor() == to_rgba(PRIMER_BINDING_SITE) for patch in annealed_axis.patches)
        assert "target contributes" not in annealed_text
        assert "Vertical guides" not in annealed_text
        assert "Expected sequence geometry" not in annealed_text

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
        assert figure_titles[0].get_fontweight() == "semibold"
        assert len(detail.texts) == 1
        local_title = next(text for text in detail.axes[0].texts if "joins F" in text.get_text())
        assert local_title.get_fontweight() == "normal"
        detail_text = "\n".join(text.get_text() for text in (*detail.texts, *detail.axes[0].texts))
        assert "barcode duplex" in detail_text
        assert "Expected sequence geometry" not in detail_text
        detail_bases = [
            text
            for text in detail.axes[0].texts
            if (text.get_gid() or "").startswith("junction:junction-01:") and ":base:" in (text.get_gid() or "")
        ]
        assert min(text.get_fontsize() for text in detail_bases) >= 9.0
        annealed_annotations = [artist for artist in annealed_axis.texts if ":base:" not in (artist.get_gid() or "")]
        detail_annotations = [artist for artist in detail.axes[0].texts if ":base:" not in (artist.get_gid() or "")]
        _assert_texts_do_not_overlap(annealed, annealed_annotations)
        _assert_texts_do_not_overlap(detail, (*detail.texts, *detail_annotations))
        _assert_annotations_clear_bases(
            annealed,
            annealed_annotations,
            [artist for artist in annealed_axis.texts if ":base:" in (artist.get_gid() or "")],
        )
        _assert_annotations_clear_bases(detail, detail_annotations, detail_bases)
    finally:
        plt.close(annealed)
        plt.close(detail)


def test_assembly_view_shows_orders_three_way_state_and_exact_pcr_duplex() -> None:
    [record] = _real_records(target_count=1, target_length=1_000)
    review = record.meta["three_way_junction_review"]
    assert len(review["geometry"]["junctions"]) == 10
    layout = assembly_renderer._assembly_layout(review_from_record(record))

    figure = baserender.render(
        record,
        renderer="junction_three_way_assembly",
        options={"view": "assembly"},
    )
    try:
        axis = figure.axes[0]
        assert axis.get_gid() == "junction-three-way-assembly:assembly"
        text = "\n".join(item.get_text() for item in axis.texts)
        assert "Oligo plan for target-0000" in text
        assert "Input target sequence" in text
        assert "The oligos remain separate before annealing" not in text
        assert "Fragment oligos encode the target" in text
        assert "Annealing forms pre-ligation junctions" in text
        assert "modeled" not in text
        assert "PCR yields the expected linear duplex" in text
        assert "forward primer-binding site" in text
        assert "reverse primer-binding site" in text
        assert any(patch.get_facecolor() == to_rgba(PRIMER_BINDING_SITE) for patch in axis.patches)
        assert "anneal" not in {
            artist.get_text()
            for artist in axis.texts
            if (artist.get_gid() or "").startswith("junction-three-way-assembly:transition:")
        }
        assert "fragment oligos span" not in text
        assert "Gray marks target sequence" not in text
        assert "Expected sequence geometry" not in text
        for fragment, strands in zip(review["geometry"]["fragments"], review["strands"], strict=True):
            prefix = f"junction-three-way-assembly:orders:{fragment['fragment_id']}"
            assert _compact_base_text(axis, f"{prefix}:top:") == strands["barcode_bearing_sequence_5to3"]
            assert _compact_base_text(axis, f"{prefix}:bottom:") == strands["complement_sequence_5to3"][::-1]
            termini = [
                artist
                for artist in axis.texts
                if (artist.get_gid() or "").startswith(prefix) and ":terminus:" in (artist.get_gid() or "")
            ]
            assert sorted(artist.get_text() for artist in termini) == ["3′", "3′", "5′", "5′"]
        expected_target = review["target"]["sequence_5to3"]
        assert _ordered_base_text(axis, "junction-three-way-assembly:input") == expected_target
        expected_target_complement = expected_target.translate(str.maketrans("ACGT", "TGCA"))
        assert _ordered_base_text(axis, "junction-three-way-assembly:three-way:top") == expected_target
        assert _ordered_base_text(axis, "junction-three-way-assembly:three-way:bottom") == expected_target_complement
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
        preligation_target_pairs = [
            collection
            for collection in axis.collections
            if (collection.get_gid() or "") == "junction-three-way-assembly:three-way:target-pairs"
        ]
        assert sum(len(collection.get_segments()) for collection in preligation_target_pairs) == len(expected_target)
        top_gaps = [
            line
            for line in axis.lines
            if (line.get_gid() or "").startswith("junction-three-way-assembly:three-way:")
            and (line.get_gid() or "").endswith(":top-gap")
        ]
        assert len(top_gaps) == len(review["geometry"]["junctions"])
        order_label_y = {
            artist.get_position()[1]
            for artist in axis.texts
            if (artist.get_gid() or "").startswith("junction-three-way-assembly:orders:")
            and (artist.get_gid() or "").endswith(":label")
        }
        assert len(order_label_y) == 1
        preligation_top_y = {
            float(offset[1])
            for collection in axis.collections
            if (collection.get_gid() or "").startswith("junction-three-way-assembly:three-way:top:glyph:")
            for offset in collection.get_offsets()
        }
        assert len(preligation_top_y) == 1
        barcode_pairs = [
            collection
            for collection in axis.collections
            if (collection.get_gid() or "").startswith("junction-three-way-assembly:three-way:")
            and (collection.get_gid() or "").endswith(":barcode-pairs")
        ]
        assert sum(len(collection.get_segments()) for collection in barcode_pairs) == sum(
            len(junction["barcode"]) for junction in review["geometry"]["junctions"]
        )
        for junction in review["geometry"]["junctions"]:
            prefix = f"junction-three-way-assembly:three-way:{junction['junction_id']}"
            assert _compact_base_text(axis, f"{prefix}:barcode:", coordinate=1) == junction["barcode"]
            assert (
                _compact_base_text(axis, f"{prefix}:barcode-complement:", coordinate=1, reverse=True)
                == junction["barcode_complement"]
            )
            termini = [
                artist
                for artist in axis.texts
                if (artist.get_gid() or "").startswith(prefix) and ":terminus:" in (artist.get_gid() or "")
            ]
            assert sorted(artist.get_text() for artist in termini) == ["3′", "3′", "5′", "5′"]
            stem_top = max(
                max(line.get_ydata())
                for line in axis.lines
                if (line.get_gid() or "").startswith(prefix) and (line.get_gid() or "").endswith(":path")
            )
            for role in ("top-left", "top-right"):
                terminus = next(artist for artist in termini if (artist.get_gid() or "").endswith(f":terminus:{role}"))
                assert terminus.get_position()[1] - stem_top == pytest.approx(0.025)

        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        by_gid = {artist.get_gid(): artist for artist in axis.texts if artist.get_gid()}
        annealing = by_gid["junction-three-way-assembly:transition:annealing"].get_window_extent(renderer)
        fragmentation = by_gid["junction-three-way-assembly:transition:fragmentation"].get_window_extent(renderer)
        input_title = by_gid["junction-three-way-assembly:input:title"].get_window_extent(renderer)
        order_title = by_gid["junction-three-way-assembly:orders:title"].get_window_extent(renderer)
        three_way_title = by_gid["junction-three-way-assembly:three-way:title"].get_window_extent(renderer)
        junction_labels = [
            artist
            for gid, artist in by_gid.items()
            if gid.startswith("junction-three-way-assembly:three-way:") and gid.endswith(":label")
        ]
        assert not input_title.overlaps(fragmentation)
        assert not fragmentation.overlaps(order_title)
        assert not order_title.overlaps(annealing)
        assert not annealing.overlaps(three_way_title)
        assert all(not three_way_title.overlaps(label.get_window_extent(renderer)) for label in junction_labels)
        assert all(label.get_ha() == "left" for label in junction_labels)
        assert all(label.get_va() == "center" for label in junction_labels)
        for junction in review["geometry"]["junctions"]:
            label = by_gid[f"junction-three-way-assembly:three-way:{junction['junction_id']}:label"]
            junction_x = layout.target_left + junction["toehold_span"]["end"] * layout.target_base_step
            assert label.get_position()[0] > junction_x
        product_coordinates = [
            artist.get_window_extent(renderer)
            for gid, artist in by_gid.items()
            if gid.startswith("junction-three-way-assembly:product:window:") and gid.endswith(":coordinate")
        ]
        product_termini = [
            artist.get_window_extent(renderer)
            for gid, artist in by_gid.items()
            if gid.startswith("junction-three-way-assembly:product:terminus:")
        ]
        assert len(product_coordinates) == 3
        assert all(
            not coordinate.overlaps(terminus) for coordinate in product_coordinates for terminus in product_termini
        )
        assert _collection_x_center(axis, "junction-three-way-assembly:orders:") == pytest.approx(0.5)
        assert _collection_x_center(axis, "junction-three-way-assembly:three-way:top") == pytest.approx(0.5)
        assert _collection_x_center(axis, "junction-three-way-assembly:product:top:window:0") == pytest.approx(0.5)
        assert by_gid["junction-three-way-assembly:title"].get_fontweight() == "semibold"
        for gid in (
            "junction-three-way-assembly:input:title",
            "junction-three-way-assembly:orders:title",
            "junction-three-way-assembly:three-way:title",
            "junction-three-way-assembly:product:title",
        ):
            assert by_gid[gid].get_fontweight() == "normal"
            assert by_gid[gid].get_fontsize() == pytest.approx(17.0)
        annotation_prefixes = (
            "junction-three-way-assembly:orders:",
            "junction-three-way-assembly:three-way:",
            "junction-three-way-assembly:product:",
        )
        molecular_annotations = [
            artist
            for gid, artist in by_gid.items()
            if gid.startswith(annotation_prefixes)
            and any(token in gid for token in (":label", ":terminus:", ":coordinate"))
        ]
        assert molecular_annotations
        assert min(artist.get_fontsize() for artist in molecular_annotations) >= 13.0
        _assert_texts_do_not_overlap(figure, axis.texts)
    finally:
        plt.close(figure)


@pytest.mark.parametrize("figure_scale", [1.0, 1.5])
def test_assembly_figure_scale_changes_canvas_without_changing_molecular_coordinates(
    figure_scale: float,
) -> None:
    record = _adapt_payload(_payload())
    coordinate_height = assembly_renderer._assembly_layout(review_from_record(record)).height

    figure = baserender.render(
        record,
        renderer="junction_three_way_assembly",
        style={"figure_scale": figure_scale},
        options={"view": "assembly"},
    )
    try:
        assert tuple(figure.axes[0].get_ylim()) == pytest.approx((0.0, coordinate_height))
        assert tuple(figure.get_size_inches()) == pytest.approx(
            (
                assembly_renderer._assembly_layout(review_from_record(record)).width * figure_scale,
                coordinate_height * figure_scale,
            )
        )
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


def test_short_continuous_stages_share_the_pcr_nucleotide_scale_and_stage_rhythm() -> None:
    review = review_from_record(_adapt_payload(_payload()))
    layout = assembly_renderer._assembly_layout(review)

    assert layout.width * layout.target_base_step == pytest.approx(PRODUCT_BASE_WIDTH_INCHES)
    assert layout.target_fontsize == pytest.approx(11.0)
    assert layout.input_title_y - layout.input_first_y == pytest.approx(STAGE_TITLE_TO_MOLECULE)
    assert layout.fragmentation_transition_y - layout.orders_title_y == pytest.approx(TRANSITION_TO_STAGE_TITLE)
    assert layout.orders_title_y - layout.orders_first_y == pytest.approx(STAGE_TITLE_TO_MOLECULE)
    assert layout.orders_first_y - 0.22 - layout.annealing_transition_y == pytest.approx(MOLECULE_TO_TRANSITION)
    assert layout.annealing_transition_y - layout.preligation_title_y == pytest.approx(TRANSITION_TO_STAGE_TITLE)
    assert layout.recovery_transition_y - layout.product_title_y == pytest.approx(TRANSITION_TO_STAGE_TITLE)
    assert layout.product_title_y - layout.product_first_y == pytest.approx(STAGE_TITLE_TO_MOLECULE)


def test_expected_pcr_duplex_stays_on_one_row_when_it_fits() -> None:
    record = _adapt_payload(_payload())
    review = review_from_record(record)
    layout = assembly_renderer._assembly_layout(review)
    assert layout.product_bases_per_row == len(review.recovery.extended_top_sequence_5to3)

    figure = baserender.render(record, renderer="junction_three_way_assembly")
    try:
        axis = figure.axes[0]
        assert not [
            artist
            for artist in axis.texts
            if (artist.get_gid() or "").startswith("junction-three-way-assembly:product:window:")
            and (artist.get_gid() or "").endswith(":coordinate")
        ]
        pairs = [
            collection
            for collection in axis.collections
            if (collection.get_gid() or "").startswith("junction-three-way-assembly:product:window:")
            and (collection.get_gid() or "").endswith(":pairs")
        ]
        assert len(pairs) == 1
    finally:
        plt.close(figure)


def test_oversized_expected_pcr_duplex_rejects_before_figure_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    [record] = _real_records(target_count=1, target_length=10_000)

    def fail_if_allocated(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("oversized molecular view must reject before figure allocation")

    monkeypatch.setattr(assembly_renderer.plt, "subplots", fail_if_allocated)
    with pytest.raises(baserender.SchemaError, match="expected-PCR-duplex limit"):
        baserender.render(
            record,
            renderer="junction_three_way_assembly",
            options={"view": "assembly"},
        )


def test_gene_scale_detail_defaults_to_all_ten_junctions_in_one_grid() -> None:
    [record] = _real_records(target_count=1, target_length=1_000)

    figure = baserender.render(
        record,
        renderer="junction_three_way_assembly",
        options={"view": "junction_detail"},
    )
    try:
        assert figure.texts[0].get_text() == "All 10 three-way junctions show the expected local annealing geometry"
        assert len(figure.axes) == 12
        used_axes = tuple(axis for axis in figure.axes if axis.texts)
        assert len(used_axes) == 10
        first_row_centers = tuple(axis.get_position().x0 + axis.get_position().width / 2 for axis in figure.axes[:3])
        last_center = used_axes[-1].get_position().x0 + used_axes[-1].get_position().width / 2
        assert last_center == pytest.approx(first_row_centers[1])
        for index, axis in enumerate(used_axes, start=1):
            texts = tuple(artist.get_text() for artist in axis.texts)
            assert f"b{index}* · 22 nt barcode duplex" in texts
            annotations = tuple(artist for artist in axis.texts if ":base:" not in (artist.get_gid() or ""))
            bases = tuple(artist for artist in axis.texts if ":base:" in (artist.get_gid() or ""))
            _assert_texts_do_not_overlap(figure, annotations)
            _assert_annotations_clear_bases(figure, annotations, bases)
    finally:
        plt.close(figure)


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
        figure = baserender.render(
            record,
            renderer="junction_three_way_assembly",
            options={"view": "assembly"},
        )
        try:
            _assert_texts_do_not_overlap(figure, figure.axes[0].texts)
        finally:
            plt.close(figure)


def other_target_ids(current: str, records) -> set[str]:
    return {record.id for record in records if record.id != current}
