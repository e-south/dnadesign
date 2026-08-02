"""Direct-render enforcement of adapter-owned renderer compatibility."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import pytest

import dnadesign.baserender as baserender
from dnadesign.baserender.src.config import ImagesOutputCfg, VideoOutputCfg
from dnadesign.baserender.src.core import Record, RenderingError, SchemaError
from dnadesign.baserender.src.outputs.images import write_images
from dnadesign.baserender.src.outputs.video import write_video
from dnadesign.baserender.src.render import renderer as renderer_core
from dnadesign.baserender.src.render.sequence_rows import SequenceRowsRenderer

from .fixtures import _payload


def _review_record() -> Record:
    return baserender.adapt_records([_payload()], adapter_kind="three_way_junction_review_v1")[0]


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
        (RenderingError, SchemaError),
        match=("record.meta.adapter 'three_way_junction_review_v1' is not compatible with renderer 'sequence_rows'"),
    ):
        render_surface(record, tmp_path)
    assert not (tmp_path / "images").exists()


def test_direct_surface_accepts_adapter_supported_renderer() -> None:
    record = _review_record()

    figure = baserender.render(record, renderer="three_way_junction_review")
    try:
        assert len(figure.axes) == 4
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
        SchemaError,
        match="record.meta.adapter 'sequence_evidence_map_v1' is not compatible with renderer 'sequence_rows'",
    ):
        baserender.render(records, renderer="sequence_rows")


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
        SchemaError,
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
