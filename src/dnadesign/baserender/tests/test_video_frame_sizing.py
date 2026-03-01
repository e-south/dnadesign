"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/baserender/tests/test_video_frame_sizing.py

Video writer sizing tests for mixed rendered frame dimensions.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from dnadesign.baserender.src.config import Style, VideoOutputCfg
from dnadesign.baserender.src.core import Record
from dnadesign.baserender.src.outputs import (
    _apply_fixed_content_radius,
    _letterbox_rgba,
    _target_frame_size,
    _trim_white_border_rgba,
    _union_centered_content_bounds,
    write_video,
)
from dnadesign.baserender.src.render import Palette


class _FakeFFMpegWriter:
    class _SavingContext:
        def __init__(self, path: str):
            self._path = Path(path)

        def __enter__(self):
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_bytes(b"fake-mp4")
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def __init__(self, *args, **kwargs):
        del args, kwargs
        self.frames = 0

    def saving(self, fig, path: str, dpi: int):
        del fig, dpi
        return self._SavingContext(path)

    def grab_frame(self):
        self.frames += 1


def test_write_video_handles_later_frames_larger_than_first(monkeypatch, tmp_path: Path) -> None:
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt

    def _fake_render_record(record: Record, *, renderer_name: str, style: Style, palette: Palette):
        del renderer_name, style, palette
        if record.id == "small":
            fig, ax = plt.subplots(figsize=(2.0, 1.0), dpi=100)
        else:
            fig, ax = plt.subplots(figsize=(2.0, 1.4), dpi=100)
        ax.set_axis_off()
        return fig

    monkeypatch.setattr(animation.writers, "is_available", lambda name: True)
    monkeypatch.setattr(animation, "FFMpegWriter", _FakeFFMpegWriter)
    monkeypatch.setattr("dnadesign.baserender.src.outputs.render_record", _fake_render_record)

    records = [
        Record(id="small", alphabet="DNA", sequence="ACGT"),
        Record(id="large", alphabet="DNA", sequence="ACGT"),
    ]
    output = VideoOutputCfg(
        kind="video",
        path=tmp_path / "mixed-size.mp4",
        fmt="mp4",
        fps=8,
        frames_per_record=1,
        pauses={},
        width_px=None,
        height_px=None,
        aspect_ratio=None,
        total_duration=2.0,
        title_text="Best-so-far trajectory · Chain 1 · objective_scalar",
        title_font_size=18,
        title_align="center",
    )

    out_path = write_video(
        records,
        output=output,
        renderer_name="sequence_rows",
        style=Style(dpi=100),
        palette=Palette(),
    )

    assert out_path.exists()


def test_letterbox_centers_content_vertically() -> None:
    arr = np.zeros((2, 2, 4), dtype=np.uint8)
    arr[:, :, :] = 17

    framed = _letterbox_rgba(arr, width=4, height=6)

    assert framed.shape == (6, 4, 4)
    assert np.all(framed[2:4, 1:3, :] == 17)
    assert np.all(framed[:2, :, :] == 255)
    assert np.all(framed[4:, :, :] == 255)


def test_letterbox_downscales_oversized_frames_when_target_is_explicit() -> None:
    arr = np.zeros((20, 40, 4), dtype=np.uint8)
    arr[:, :, :] = 17

    framed = _letterbox_rgba(arr, width=20, height=10)

    assert framed.shape == (10, 20, 4)
    assert np.any(framed[:, :, :3] < 248)


def test_target_frame_size_keeps_explicit_height_even_when_smaller_than_natural(tmp_path: Path) -> None:
    output = VideoOutputCfg(
        kind="video",
        path=tmp_path / "explicit-height.mp4",
        fmt="mp4",
        fps=8,
        frames_per_record=1,
        pauses={},
        width_px=None,
        height_px=200,
        aspect_ratio=None,
        total_duration=2.0,
        title_text=None,
        title_font_size=None,
        title_align="center",
    )

    width, height = _target_frame_size(natural_w=1000, natural_h=800, output=output)

    assert width == 1000
    assert height == 200


def test_trim_white_border_rgba_crops_white_edges() -> None:
    arr = np.ones((10, 12, 4), dtype=np.uint8) * 255
    arr[3:7, 4:9, :] = 42

    trimmed = _trim_white_border_rgba(arr, threshold=248, pad_px=1)

    assert trimmed.shape == (6, 7, 4)
    assert np.any(trimmed[:, :, :3] < 248)


def test_union_centered_content_bounds_is_stable_across_varying_frame_sizes() -> None:
    frame_shapes = [(120, 80), (100, 70)]
    content_bounds = [(10, 8, 90, 60), (6, 10, 70, 58)]

    bounds = _union_centered_content_bounds(
        frame_shapes=frame_shapes,
        content_bounds=content_bounds,
        canvas_width=120,
        canvas_height=80,
        pad_px=2,
    )

    assert bounds == (8, 6, 92, 65)


def test_apply_fixed_content_radius_sets_consistent_meta_for_sequence_rows() -> None:
    records = [
        Record(id="r1", alphabet="DNA", sequence="ACGTACGT"),
        Record(id="r2", alphabet="DNA", sequence="ACGTACGTACGT"),
    ]

    prepared = _apply_fixed_content_radius(records, renderer_name="sequence_rows", style=Style())

    assert [record.id for record in prepared] == ["r1", "r2"]
    radii = [float(record.meta["fixed_content_radius_px"]) for record in prepared]
    assert all(radius > 0.0 for radius in radii)
    assert radii[0] == radii[1]
