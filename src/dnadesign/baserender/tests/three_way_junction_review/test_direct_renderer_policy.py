"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/tests/three_way_junction_review/test_direct_renderer_policy.py

Direct-render enforcement of adapter-owned renderer compatibility.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import pytest

import dnadesign.baserender as baserender
from dnadesign.baserender.src.config import ImagesOutputCfg, VideoOutputCfg
from dnadesign.baserender.src.core import Record, Span
from dnadesign.baserender.src.core.record import Display, Effect, Feature
from dnadesign.baserender.src.outputs.images import write_images
from dnadesign.baserender.src.outputs.video import write_video
from dnadesign.baserender.src.render import renderer as renderer_core
from dnadesign.baserender.src.render.sequence_rows import SequenceRowsRenderer
from dnadesign.baserender.src.render.topology_cartoon import TopologyCartoonRenderer

from .fixtures import _payload


def _review_record() -> Record:
    return baserender.adapt_records([_payload()], adapter_kind="three_way_junction_review_v1")[0]


def _topology_record() -> Record:
    return baserender.adapt_record(
        {
            "contract_kind": "yiu_topology_cartoon_v1",
            "state_id": "topology-valid",
            "topology_kind": "circular_duplex",
            "sequence": "ACGT",
            "segments": [{"segment_id": "payload", "state_start": 0, "state_end": 4}],
            "annotations": [],
            "cuts": [],
            "junctions": [],
            "fragments": [],
            "display": {"title": "Valid topology"},
            "meta": {},
        },
        adapter_kind="yiu_topology_cartoon_v1",
        alphabet="DNA",
    )


def _collapsed_span_link_record() -> Record:
    return Record(
        id="collapsed-span-link",
        alphabet="DNA",
        sequence="ACGTACGTAC",
        features=(
            Feature(
                id="left",
                kind="kmer",
                span=Span(start=0, end=4, strand="fwd"),
                label="ACGT",
                tags=("left",),
                attrs={},
                render={},
            ),
            Feature(
                id="right",
                kind="kmer",
                span=Span(start=6, end=10, strand="fwd"),
                label="GTAC",
                tags=("right",),
                attrs={},
                render={},
            ),
        ),
        effects=(
            Effect(
                kind="span_link",
                target={"from_feature_id": "left", "to_feature_id": "right"},
                params={"inner_margin_bp": 1000},
                render={},
            ),
        ),
        display=Display(),
        meta={},
    )


def _invoke_batch_surface(surface: str, records: list[Record], tmp_path: Path, *, renderer_name: str) -> None:
    if surface == "single":
        baserender.render(records[0], renderer=renderer_name)
        return
    if surface == "grid":
        baserender.render(records, renderer=renderer_name)
        return

    style = baserender.resolve_style(preset=None, overrides=None)
    palette = baserender.Palette(style.palette)
    if surface == "image_directory":
        write_images(
            records,
            output=ImagesOutputCfg(kind="images", dir=tmp_path / "images", path=None, fmt="svg"),
            renderer_name=renderer_name,
            style=style,
            palette=palette,
        )
        return
    if surface == "image_single":
        write_images(
            records,
            output=ImagesOutputCfg(
                kind="images",
                dir=None,
                path=tmp_path / "single-image" / "grid.svg",
                fmt="svg",
            ),
            renderer_name=renderer_name,
            style=style,
            palette=palette,
        )
        return
    if surface == "video":
        write_video(
            records,
            output=VideoOutputCfg(
                kind="video",
                path=tmp_path / "video" / "review.mp4",
                fmt="mp4",
                fps=1,
                frames_per_record=1,
                pauses={},
                width_px=100,
                height_px=100,
                aspect_ratio=None,
                total_duration=None,
            ),
            renderer_name=renderer_name,
            style=style,
            palette=palette,
        )
        return
    raise AssertionError(f"unknown test surface: {surface}")


def _direct_render(record: Record, _tmp_path: Path) -> None:
    baserender.render(record)


def _grid_render(record: Record, _tmp_path: Path) -> None:
    baserender.render([record], renderer="sequence_rows")


def _image_writer(record: Record, tmp_path: Path) -> None:
    style = baserender.resolve_style(preset=None, overrides=None)
    write_images(
        [record],
        output=ImagesOutputCfg(kind="images", dir=tmp_path / "images", path=None, fmt="svg"),
        renderer_name="sequence_rows",
        style=style,
        palette=baserender.Palette(style.palette),
    )


@pytest.mark.parametrize("render_surface", [_direct_render, _grid_render, _image_writer])
def test_direct_surfaces_reject_adapter_renderer_mismatch_before_render_or_figure_allocation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    render_surface: Callable[[Record, Path], None],
) -> None:
    record = _review_record()

    def _unexpected_call(*_args: object, **_kwargs: object) -> None:
        pytest.fail("incompatible renderer reached rendering or figure allocation")

    monkeypatch.setattr(renderer_core, "get_renderer", _unexpected_call)
    monkeypatch.setattr(SequenceRowsRenderer, "render", _unexpected_call)
    monkeypatch.setattr(plt, "subplots", _unexpected_call)

    with pytest.raises(
        baserender.RenderingError,
        match=("record.meta.adapter 'three_way_junction_review_v1' is not compatible with renderer 'sequence_rows'"),
    ):
        render_surface(record, tmp_path)
    assert not (tmp_path / "images").exists()


def test_direct_surface_accepts_adapter_supported_renderer() -> None:
    record = _review_record()

    figure = baserender.render(record, renderer="junction_three_way_assembly")
    try:
        assert len(figure.axes) == 1
        assert figure.axes[0].get_gid() == "junction-three-way-assembly:overview"
    finally:
        plt.close(figure)


def test_direct_surface_keeps_unowned_records_renderer_agnostic() -> None:
    record = Record(id="unowned", alphabet="DNA", sequence="ACGT")

    figure = baserender.render(record)
    try:
        assert figure.axes
    finally:
        plt.close(figure)


def test_grid_rejects_a_late_renderer_mismatch_before_render_or_figure_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = [
        Record(id="unowned", alphabet="DNA", sequence="ACGT"),
        Record(
            id="owned",
            alphabet="DNA",
            sequence="ACGT",
            meta={"adapter": "sequence_evidence_map_v1"},
        ),
    ]

    def _unexpected_call(*_args: object, **_kwargs: object) -> None:
        pytest.fail("heterogeneous grid reached rendering or figure allocation")

    monkeypatch.setattr("dnadesign.baserender.src.public.api.render_record_figure", _unexpected_call)
    monkeypatch.setattr(plt, "subplots", _unexpected_call)

    with pytest.raises(
        baserender.RenderingError,
        match="record.meta.adapter 'sequence_evidence_map_v1' is not compatible with renderer 'sequence_rows'",
    ):
        baserender.render(records, renderer="sequence_rows")


def test_grid_rejects_a_late_invalid_record_before_render_or_figure_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = [
        Record(id="valid", alphabet="DNA", sequence="ACGT"),
        Record(id="invalid", alphabet="DNA", sequence="ACGX"),
    ]

    def _unexpected_call(*_args: object, **_kwargs: object) -> None:
        pytest.fail("invalid heterogeneous grid reached rendering or figure allocation")

    monkeypatch.setattr("dnadesign.baserender.src.public.api.render_record_figure", _unexpected_call)
    monkeypatch.setattr(plt, "subplots", _unexpected_call)

    with pytest.raises(baserender.RenderingError):
        baserender.render(records)


@pytest.mark.parametrize("surface", ["grid", "image_directory", "image_single", "video"])
def test_batch_surfaces_reject_late_renderer_specific_invalidity_before_allocation_or_output_mutation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    surface: str,
) -> None:
    records = [
        _topology_record(),
        Record(id="renderer-invalid", alphabet="DNA", sequence="ACGT"),
    ]

    def _unexpected_call(*_args: object, **_kwargs: object) -> None:
        pytest.fail("renderer-invalid batch reached rendering or figure allocation")

    monkeypatch.setattr(TopologyCartoonRenderer, "render", _unexpected_call)
    monkeypatch.setattr(plt, "subplots", _unexpected_call)

    with pytest.raises(
        baserender.RenderingError,
        match="topology_cartoon requires record.meta.topology_cartoon",
    ):
        _invoke_batch_surface(surface, records, tmp_path, renderer_name="topology_cartoon")

    assert not (tmp_path / "images").exists()
    assert not (tmp_path / "single-image").exists()
    assert not (tmp_path / "video").exists()


@pytest.mark.parametrize("surface", ["single", "grid", "image_directory", "image_single", "video"])
def test_render_surfaces_wrap_malformed_record_meta_before_policy_or_output_mutation(
    tmp_path: Path,
    surface: str,
) -> None:
    record = Record(id="invalid-meta", alphabet="DNA", sequence="ACGT", meta=[])  # type: ignore[arg-type]

    with pytest.raises(baserender.RenderingError, match="record.meta must be a mapping/dict"):
        _invoke_batch_surface(surface, [record], tmp_path, renderer_name="sequence_rows")

    assert not (tmp_path / "images").exists()
    assert not (tmp_path / "single-image").exists()
    assert not (tmp_path / "video").exists()


@pytest.mark.parametrize("surface", ["single", "grid", "image_directory", "image_single", "video"])
@pytest.mark.parametrize(
    ("invalid_record", "message"),
    [
        (_collapsed_span_link_record(), "span_link collapsed geometry"),
        (
            Record(
                id="invalid-highlight-index",
                alphabet="DNA",
                sequence="ACGT",
                meta={"base_highlights": {"primary": ["not-an-index"]}},
            ),
            "record.meta.base_highlights.primary must contain integer indices",
        ),
        (
            Record(
                id="invalid-highlight-color",
                alphabet="DNA",
                sequence="ACGT",
                meta={
                    "base_highlights": {"primary": [0]},
                    "base_highlight_color": {"primary": "not-a-color"},
                },
            ),
            "record.meta.base_highlight_color.primary must be a valid color",
        ),
        (
            Record(
                id="invalid-span-edge-alpha",
                alphabet="DNA",
                sequence="ACGT",
                meta={"span_edge_markers": [{"start": 0, "end": 1, "alpha": 2.0, "color": "#000"}]},
            ),
            r"record.meta.span_edge_markers alpha must be finite and in \[0, 1\]",
        ),
        (
            Record(
                id="invalid-indexed-highlight-key",
                alphabet="DNA",
                sequence="ACGT",
                meta={"base_highlight_colors": {"primary": {"not-an-index": "#000"}}},
            ),
            "record.meta.base_highlight_colors.primary keys must be integer indices",
        ),
    ],
)
def test_sequence_rows_rejects_late_draw_failures_before_allocation_or_output_mutation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    surface: str,
    invalid_record: Record,
    message: str,
) -> None:
    records = [Record(id="valid", alphabet="DNA", sequence="ACGT"), invalid_record]
    if surface == "single":
        records = [invalid_record]

    def _unexpected_call(*_args: object, **_kwargs: object) -> None:
        pytest.fail("invalid sequence_rows evidence reached figure allocation")

    monkeypatch.setattr(plt, "figure", _unexpected_call)
    monkeypatch.setattr(plt, "subplots", _unexpected_call)

    with pytest.raises(baserender.RenderingError, match=message):
        _invoke_batch_surface(surface, records, tmp_path, renderer_name="sequence_rows")

    assert not (tmp_path / "images").exists()
    assert not (tmp_path / "single-image").exists()
    assert not (tmp_path / "video").exists()


def test_video_writer_rejects_adapter_renderer_mismatch_before_output_mutation(tmp_path: Path) -> None:
    record = Record(
        id="owned",
        alphabet="DNA",
        sequence="ACGT",
        meta={"adapter": "sequence_evidence_map_v1"},
    )
    style = baserender.resolve_style(preset=None, overrides=None)
    output = VideoOutputCfg(
        kind="video",
        path=tmp_path / "video" / "review.mp4",
        fmt="mp4",
        fps=1,
        frames_per_record=1,
        pauses={},
        width_px=100,
        height_px=100,
        aspect_ratio=None,
        total_duration=None,
    )

    with pytest.raises(
        baserender.RenderingError,
        match="record.meta.adapter 'sequence_evidence_map_v1' is not compatible with renderer 'sequence_rows'",
    ):
        write_video(
            [record],
            output=output,
            renderer_name="sequence_rows",
            style=style,
            palette=baserender.Palette(style.palette),
        )

    assert not output.path.parent.exists()


def test_direct_writers_reject_a_late_invalid_record_before_output_mutation(tmp_path: Path) -> None:
    records = [
        Record(id="valid", alphabet="DNA", sequence="ACGT"),
        Record(id="invalid", alphabet="DNA", sequence="ACGX"),
    ]
    style = baserender.resolve_style(preset=None, overrides=None)
    image_output = ImagesOutputCfg(kind="images", dir=tmp_path / "images", path=None, fmt="svg")
    single_image_output = ImagesOutputCfg(
        kind="images",
        dir=None,
        path=tmp_path / "single-image" / "grid.svg",
        fmt="svg",
    )
    video_output = VideoOutputCfg(
        kind="video",
        path=tmp_path / "video" / "review.mp4",
        fmt="mp4",
        fps=1,
        frames_per_record=1,
        pauses={},
        width_px=100,
        height_px=100,
        aspect_ratio=None,
        total_duration=None,
    )

    with pytest.raises(baserender.RenderingError):
        write_images(
            records,
            output=image_output,
            renderer_name="sequence_rows",
            style=style,
            palette=baserender.Palette(style.palette),
        )
    with pytest.raises(baserender.RenderingError):
        write_video(
            records,
            output=video_output,
            renderer_name="sequence_rows",
            style=style,
            palette=baserender.Palette(style.palette),
        )
    with pytest.raises(baserender.RenderingError):
        write_images(
            records,
            output=single_image_output,
            renderer_name="sequence_rows",
            style=style,
            palette=baserender.Palette(style.palette),
        )

    assert not image_output.dir.exists()
    assert not single_image_output.path.parent.exists()
    assert not video_output.path.parent.exists()


@pytest.mark.parametrize("surface", ["single", "grid", "image_directory", "image_single", "video"])
def test_render_surfaces_reject_an_unknown_renderer_before_output_mutation(
    tmp_path: Path,
    surface: str,
) -> None:
    with pytest.raises(baserender.RenderingError, match="Unknown renderer: missing"):
        _invoke_batch_surface(
            surface,
            [_review_record()],
            tmp_path,
            renderer_name="missing",
        )

    assert not (tmp_path / "images").exists()
    assert not (tmp_path / "single-image").exists()
    assert not (tmp_path / "video").exists()
