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
import pytest

from dnadesign.baserender.src.config import Style, VideoOutputCfg
from dnadesign.baserender.src.core import Display, Feature, Record, SchemaError, Span
from dnadesign.baserender.src.outputs import (
    _apply_fixed_content_radius,
    _letterbox_rgba,
    _scale_rgba_to_width,
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
    monkeypatch.setattr("dnadesign.baserender.src.outputs.video.render_record", _fake_render_record)

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


def test_scale_rgba_to_width_can_upscale_cropped_content_for_fill_width_video() -> None:
    arr = np.zeros((10, 20, 4), dtype=np.uint8)
    arr[:, :, 3] = 255

    scaled = np.asarray(_scale_rgba_to_width(arr, width=40))

    assert scaled.shape[:2] == (20, 40)


def test_write_video_accepts_title_without_dynamic_subtitle(monkeypatch, tmp_path: Path) -> None:
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt

    def _fake_render_record(record: Record, *, renderer_name: str, style: Style, palette: Palette):
        del record, renderer_name, style, palette
        fig, ax = plt.subplots(figsize=(2.0, 1.0), dpi=100)
        ax.plot([0, 1], [0, 1], color="black")
        ax.set_axis_off()
        return fig

    monkeypatch.setattr(animation.writers, "is_available", lambda name: True)
    monkeypatch.setattr(animation, "FFMpegWriter", _FakeFFMpegWriter)
    monkeypatch.setattr("dnadesign.baserender.src.outputs.video.render_record", _fake_render_record)

    output = VideoOutputCfg(
        kind="video",
        path=tmp_path / "title-only.mp4",
        fmt="mp4",
        fps=2,
        frames_per_record=1,
        pauses={},
        width_px=640,
        height_px=None,
        aspect_ratio=16 / 9,
        total_duration=None,
        title_text="Title without subtitle",
        title_font_size=18,
        title_align="center",
    )

    out_path = write_video(
        [Record(id="r1", alphabet="DNA", sequence="ACGT", display=Display())],
        output=output,
        renderer_name="sequence_rows",
        style=Style(dpi=100, uniform_display_font_size=True),
        palette=Palette(),
    )

    assert out_path.exists()


def test_write_video_fill_width_without_header_top_aligns_sequence_rows(monkeypatch, tmp_path: Path) -> None:
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt

    class _FrameCaptureWriter:
        class _SavingContext:
            def __init__(self, writer: "_FrameCaptureWriter", path: str):
                self._writer = writer
                self._path = Path(path)

            def __enter__(self):
                self._path.parent.mkdir(parents=True, exist_ok=True)
                self._path.write_bytes(b"fake-mp4")
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        def __init__(self, *args, **kwargs):
            del args, kwargs
            self._fig = None
            self.frame_top_margins: list[int] = []

        def saving(self, fig, path: str, dpi: int):
            del dpi
            self._fig = fig
            return self._SavingContext(self, path)

        def grab_frame(self):
            if self._fig is None:
                raise AssertionError("Expected figure to be bound before frame capture.")
            image = self._fig.axes[0].images[0]
            arr = np.asarray(image.get_array())
            non_white = np.any(arr[:, :, :3] < 245, axis=2)
            ys, _xs = np.where(non_white)
            self.frame_top_margins.append(int(ys.min()))

    writer_box: dict[str, _FrameCaptureWriter] = {}

    def _writer_factory(*args, **kwargs):
        writer = _FrameCaptureWriter(*args, **kwargs)
        writer_box["writer"] = writer
        return writer

    def _fake_render_record(record: Record, *, renderer_name: str, style: Style, palette: Palette):
        del renderer_name, style, palette
        fig, ax = plt.subplots(figsize=(3.0, 1.0), dpi=100)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_axis_off()
        ax.add_patch(plt.Rectangle((0.1, 0.88), 0.8, 0.04, color="black"))
        y0 = 0.76 if record.id == "top" else 0.06
        ax.add_patch(plt.Rectangle((0.1, y0), 0.8, 0.18, color="black"))
        return fig

    monkeypatch.setattr(animation.writers, "is_available", lambda name: True)
    monkeypatch.setattr(animation, "FFMpegWriter", _writer_factory)
    monkeypatch.setattr("dnadesign.baserender.src.outputs.video.render_record", _fake_render_record)

    records = [
        Record(id="top", alphabet="DNA", sequence="ACGT"),
        Record(id="bottom", alphabet="DNA", sequence="TGCA"),
    ]
    output = VideoOutputCfg(
        kind="video",
        path=tmp_path / "fill-width-no-header.mp4",
        fmt="mp4",
        fps=8,
        frames_per_record=1,
        pauses={},
        width_px=None,
        height_px=None,
        aspect_ratio=None,
        total_duration=None,
        content_fit="fill_width",
        title_text=None,
        title_font_size=None,
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
    assert writer_box["writer"].frame_top_margins
    assert max(writer_box["writer"].frame_top_margins) <= 24


def test_write_video_fill_width_keeps_embedded_legend_level_stable(monkeypatch, tmp_path: Path) -> None:
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt

    class _LegendCaptureWriter:
        class _SavingContext:
            def __init__(self, writer: "_LegendCaptureWriter", path: str):
                self._writer = writer
                self._path = Path(path)

            def __enter__(self):
                self._path.parent.mkdir(parents=True, exist_ok=True)
                self._path.write_bytes(b"fake-mp4")
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        def __init__(self, *args, **kwargs):
            del args, kwargs
            self._fig = None
            self.legend_y0: list[int] = []

        def saving(self, fig, path: str, dpi: int):
            del dpi
            self._fig = fig
            return self._SavingContext(self, path)

        def grab_frame(self):
            if self._fig is None:
                raise AssertionError("Expected figure to be bound before frame capture.")
            image = self._fig.axes[0].images[0]
            arr = np.asarray(image.get_array())[:, :, :3]
            green = (arr[:, :, 1] > 100) & (arr[:, :, 0] < 80) & (arr[:, :, 2] < 80)
            ys, _xs = np.where(green)
            if ys.size == 0:
                raise AssertionError("Expected green legend marker in captured frame.")
            self.legend_y0.append(int(ys.min()))

    writer_box: dict[str, _LegendCaptureWriter] = {}

    def _writer_factory(*args, **kwargs):
        writer = _LegendCaptureWriter(*args, **kwargs)
        writer_box["writer"] = writer
        return writer

    def _fake_render_record(record: Record, *, renderer_name: str, style: Style, palette: Palette):
        del renderer_name, style, palette
        fig, ax = plt.subplots(figsize=(4.0, 2.0), dpi=100)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_axis_off()
        # Varying top content should not move the embedded legend band.
        content_y = 0.76 if record.id == "high" else 0.48
        ax.add_patch(plt.Rectangle((0.1, content_y), 0.18, 0.08, color="black"))
        ax.add_patch(plt.Rectangle((0.42, 0.06), 0.16, 0.06, color="#008000"))
        return fig

    monkeypatch.setattr(animation.writers, "is_available", lambda name: True)
    monkeypatch.setattr(animation, "FFMpegWriter", _writer_factory)
    monkeypatch.setattr("dnadesign.baserender.src.outputs.video.render_record", _fake_render_record)

    records = [
        Record(id="high", alphabet="DNA", sequence="ACGT"),
        Record(id="low", alphabet="DNA", sequence="TGCA"),
    ]
    output = VideoOutputCfg(
        kind="video",
        path=tmp_path / "legend-stable.mp4",
        fmt="mp4",
        fps=8,
        frames_per_record=1,
        pauses={},
        width_px=None,
        height_px=None,
        aspect_ratio=None,
        total_duration=None,
        content_fit="fill_width",
        title_text=None,
        title_font_size=None,
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
    assert len(set(writer_box["writer"].legend_y0)) == 1


def test_write_video_fill_width_per_frame_preserves_content_aspect(monkeypatch, tmp_path: Path) -> None:
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt

    class _AspectCaptureWriter:
        class _SavingContext:
            def __init__(self, writer: "_AspectCaptureWriter", path: str):
                self._writer = writer
                self._path = Path(path)

            def __enter__(self):
                self._path.parent.mkdir(parents=True, exist_ok=True)
                self._path.write_bytes(b"fake-mp4")
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        def __init__(self, *args, **kwargs):
            del args, kwargs
            self._fig = None
            self.content_shapes: list[tuple[int, int]] = []

        def saving(self, fig, path: str, dpi: int):
            del dpi
            self._fig = fig
            return self._SavingContext(self, path)

        def grab_frame(self):
            if self._fig is None:
                raise AssertionError("Expected figure to be bound before frame capture.")
            image = self._fig.axes[0].images[0]
            arr = np.asarray(image.get_array())
            non_white = np.any(arr[:, :, :3] < 245, axis=2)
            ys, xs = np.where(non_white)
            self.content_shapes.append((int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)))

    writer_box: dict[str, _AspectCaptureWriter] = {}

    def _writer_factory(*args, **kwargs):
        writer = _AspectCaptureWriter(*args, **kwargs)
        writer_box["writer"] = writer
        return writer

    def _fake_render_record(record: Record, *, renderer_name: str, style: Style, palette: Palette):
        del renderer_name, style, palette
        fig_width = 4.0 if record.id == "wide" else 2.0
        fig, ax = plt.subplots(figsize=(fig_width, 1.0), dpi=100)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_axis_off()
        ax.add_patch(plt.Rectangle((0.0, 0.0), 1.0, 1.0, color="black"))
        return fig

    monkeypatch.setattr(animation.writers, "is_available", lambda name: True)
    monkeypatch.setattr(animation, "FFMpegWriter", _writer_factory)
    monkeypatch.setattr("dnadesign.baserender.src.outputs.video.render_record", _fake_render_record)

    records = [
        Record(id="wide", alphabet="DNA", sequence="ACGT"),
        Record(id="narrow", alphabet="DNA", sequence="TGCA"),
    ]
    output = VideoOutputCfg(
        kind="video",
        path=tmp_path / "fill-width-per-frame.mp4",
        fmt="mp4",
        fps=8,
        frames_per_record=1,
        pauses={},
        width_px=240,
        height_px=140,
        aspect_ratio=None,
        total_duration=None,
        content_fit="fill_width_per_frame",
        title_text=None,
        title_font_size=None,
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
    assert len(writer_box["writer"].content_shapes) == 2
    wide_shape, narrow_shape = writer_box["writer"].content_shapes
    assert abs(wide_shape[0] - narrow_shape[0]) <= 8
    assert abs((wide_shape[0] / wide_shape[1]) - 4.0) < 0.15
    assert abs((narrow_shape[0] / narrow_shape[1]) - 2.0) < 0.15


def test_write_video_footer_legend_font_is_stable_for_variable_sequence_lengths(monkeypatch, tmp_path: Path) -> None:
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt

    class _LegendFontCaptureWriter:
        class _SavingContext:
            def __init__(self, writer: "_LegendFontCaptureWriter", path: str):
                self._writer = writer
                self._path = Path(path)

            def __enter__(self):
                self._path.parent.mkdir(parents=True, exist_ok=True)
                self._path.write_bytes(b"fake-mp4")
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        def __init__(self, *args, **kwargs):
            del args, kwargs
            self._fig = None
            self.legend_font_sizes: list[tuple[float, ...]] = []
            self.legend_patch_sizes: list[tuple[tuple[float, float], ...]] = []

        def saving(self, fig, path: str, dpi: int):
            del dpi
            self._fig = fig
            return self._SavingContext(self, path)

        def grab_frame(self):
            if self._fig is None:
                raise AssertionError("Expected figure to be bound before frame capture.")
            self.legend_font_sizes.append(
                tuple(
                    float(text.get_fontsize())
                    for text in self._fig.texts
                    if str(text.get_gid()).startswith("video_footer_legend_label:")
                )
            )
            self.legend_patch_sizes.append(
                tuple(
                    (float(patch.get_width()), float(patch.get_height()))
                    for patch in self._fig.artists
                    if str(patch.get_gid()).startswith("video_footer_legend_patch:")
                )
            )

    writer_box: dict[str, _LegendFontCaptureWriter] = {}

    def _writer_factory(*args, **kwargs):
        writer = _LegendFontCaptureWriter(*args, **kwargs)
        writer_box["writer"] = writer
        return writer

    def _fake_render_record(record: Record, *, renderer_name: str, style: Style, palette: Palette):
        del renderer_name, style, palette
        fig_width = 8.0 if record.id == "long" else 2.0
        fig, ax = plt.subplots(figsize=(fig_width, 1.0), dpi=100)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_axis_off()
        ax.add_patch(plt.Rectangle((0.0, 0.2), 1.0, 0.6, color="black"))
        return fig

    monkeypatch.setattr(animation.writers, "is_available", lambda name: True)
    monkeypatch.setattr(animation, "FFMpegWriter", _writer_factory)
    monkeypatch.setattr("dnadesign.baserender.src.outputs.video.render_record", _fake_render_record)

    feature = Feature(id="tfbs", kind="kmer", span=Span(start=0, end=4), label="ACGT", tags=("tf:LexA",))
    records = [
        Record(
            id="short",
            alphabet="DNA",
            sequence="ACGT",
            features=(feature,),
            display=Display(tag_labels={"tf:LexA": "LexA sites"}),
        ),
        Record(
            id="long",
            alphabet="DNA",
            sequence="ACGT" * 16,
            features=(feature,),
            display=Display(tag_labels={"tf:LexA": "LexA sites"}),
        ),
    ]
    output = VideoOutputCfg(
        kind="video",
        path=tmp_path / "stable-footer-legend.mp4",
        fmt="mp4",
        fps=8,
        frames_per_record=1,
        pauses={},
        width_px=320,
        height_px=120,
        aspect_ratio=None,
        total_duration=None,
        content_fit="fill_width_per_frame",
        title_text=None,
        title_font_size=None,
        title_align="center",
    )

    out_path = write_video(
        records,
        output=output,
        renderer_name="sequence_rows",
        style=Style(dpi=100, legend=True, legend_mode="bottom", legend_font_size=14),
        palette=Palette({"tf:LexA": "#2D9B66"}),
    )

    assert out_path.exists()
    assert writer_box["writer"].legend_font_sizes == [(10.08,), (10.08,)]
    assert writer_box["writer"].legend_patch_sizes
    assert all(
        max(width * output.width_px, height * output.height_px) <= 14.0
        for frame_sizes in writer_box["writer"].legend_patch_sizes
        for width, height in frame_sizes
    )


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


def test_target_frame_size_rejects_conflicting_explicit_size_and_aspect_ratio(tmp_path: Path) -> None:
    output = VideoOutputCfg(
        kind="video",
        path=tmp_path / "conflicting-size.mp4",
        fmt="mp4",
        fps=8,
        frames_per_record=1,
        pauses={},
        width_px=100,
        height_px=55,
        aspect_ratio=2.0,
        total_duration=2.0,
        title_text=None,
        title_font_size=None,
        title_align="center",
    )

    with pytest.raises(SchemaError, match="aspect"):
        _target_frame_size(natural_w=80, natural_h=40, output=output)


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
    top_extents = [float(record.meta["fixed_content_top_extent_px"]) for record in prepared]
    bottom_extents = [float(record.meta["fixed_content_bottom_extent_px"]) for record in prepared]
    assert all(value > 0.0 for value in top_extents)
    assert all(value > 0.0 for value in bottom_extents)
    assert top_extents[0] == top_extents[1]
    assert bottom_extents[0] == bottom_extents[1]


def test_write_video_updates_subtitle_per_frame_from_record_display(monkeypatch, tmp_path: Path) -> None:
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt

    expected_subtitles = [
        "lexA=0.80 cpxR=0.71",
        "lexA=0.85 cpxR=0.74",
    ]

    class _SubtitleCaptureWriter:
        class _SavingContext:
            def __init__(self, writer: "_SubtitleCaptureWriter", path: str):
                self._writer = writer
                self._path = Path(path)

            def __enter__(self):
                self._path.parent.mkdir(parents=True, exist_ok=True)
                self._path.write_bytes(b"fake-mp4")
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        def __init__(self, *args, **kwargs):
            del args, kwargs
            self._fig = None
            self.seen_subtitles: list[str] = []
            self.seen_positions: list[tuple[float, float] | None] = []

        def saving(self, fig, path: str, dpi: int):
            del dpi
            self._fig = fig
            return self._SavingContext(self, path)

        def grab_frame(self):
            if self._fig is None:
                raise AssertionError("Expected figure to be bound before frame capture.")
            subtitle = ""
            subtitle_position: tuple[float, float] | None = None
            for text in self._fig.texts:
                text_value = str(text.get_text())
                if text_value in expected_subtitles:
                    subtitle = text_value
                    x_pos, y_pos = text.get_position()
                    subtitle_position = (float(x_pos), float(y_pos))
                    break
            self.seen_subtitles.append(subtitle)
            self.seen_positions.append(subtitle_position)

    writer_box: dict[str, _SubtitleCaptureWriter] = {}

    def _writer_factory(*args, **kwargs):
        writer = _SubtitleCaptureWriter(*args, **kwargs)
        writer_box["writer"] = writer
        return writer

    def _fake_render_record(record: Record, *, renderer_name: str, style: Style, palette: Palette):
        del record, renderer_name, style, palette
        fig, ax = plt.subplots(figsize=(2.4, 1.2), dpi=100)
        ax.set_axis_off()
        return fig

    monkeypatch.setattr(animation.writers, "is_available", lambda name: True)
    monkeypatch.setattr(animation, "FFMpegWriter", _writer_factory)
    monkeypatch.setattr("dnadesign.baserender.src.outputs.video.render_record", _fake_render_record)

    records = [
        Record(
            id="frame-1",
            alphabet="DNA",
            sequence="ACGT",
            display=Display(video_subtitle=expected_subtitles[0]),
        ),
        Record(
            id="frame-2",
            alphabet="DNA",
            sequence="TGCA",
            display=Display(video_subtitle=expected_subtitles[1]),
        ),
    ]
    output = VideoOutputCfg(
        kind="video",
        path=tmp_path / "dynamic-subtitle.mp4",
        fmt="mp4",
        fps=8,
        frames_per_record=1,
        pauses={},
        width_px=None,
        height_px=None,
        aspect_ratio=None,
        total_duration=None,
        title_text="Trajectory",
        title_font_size=12,
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
    writer = writer_box["writer"]
    assert writer.seen_subtitles == expected_subtitles
    assert all(position is not None for position in writer.seen_positions)
    first_position = writer.seen_positions[0]
    assert all(position == first_position for position in writer.seen_positions)


def test_write_video_long_title_text_is_fitted_within_figure_bounds(monkeypatch, tmp_path: Path) -> None:
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt

    title_text = (
        "Best-so-far motif placement improves over sweeps for "
        "project_tfs_lexa_cpxr_baer_rcda_lrp_fur_fnr_acrr_soxr_soxs "
        "chain_1 objective_scalar normalized-llr"
    )

    class _BoundsCaptureWriter:
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
            self._fig = None
            self.title_bounds: tuple[float, float, float, float] | None = None

        def saving(self, fig, path: str, dpi: int):
            del dpi
            self._fig = fig
            return self._SavingContext(path)

        def grab_frame(self):
            if self._fig is None:
                raise AssertionError("Expected figure to be bound before frame capture.")
            self._fig.canvas.draw()
            renderer = self._fig.canvas.get_renderer()
            if not self._fig.texts:
                raise AssertionError("Expected at least one figure text artist for video title.")
            title_artist = self._fig.texts[0]
            bbox = title_artist.get_window_extent(renderer=renderer).transformed(self._fig.transFigure.inverted())
            self.title_bounds = (float(bbox.x0), float(bbox.y0), float(bbox.x1), float(bbox.y1))

    writer_box: dict[str, _BoundsCaptureWriter] = {}

    def _writer_factory(*args, **kwargs):
        writer = _BoundsCaptureWriter(*args, **kwargs)
        writer_box["writer"] = writer
        return writer

    def _fake_render_record(record: Record, *, renderer_name: str, style: Style, palette: Palette):
        del record, renderer_name, style, palette
        fig, ax = plt.subplots(figsize=(2.4, 1.2), dpi=100)
        ax.set_axis_off()
        return fig

    monkeypatch.setattr(animation.writers, "is_available", lambda name: True)
    monkeypatch.setattr(animation, "FFMpegWriter", _writer_factory)
    monkeypatch.setattr("dnadesign.baserender.src.outputs.video.render_record", _fake_render_record)

    records = [Record(id="frame-1", alphabet="DNA", sequence="ACGT")]
    output = VideoOutputCfg(
        kind="video",
        path=tmp_path / "long-title-fit.mp4",
        fmt="mp4",
        fps=8,
        frames_per_record=1,
        pauses={},
        width_px=None,
        height_px=None,
        aspect_ratio=None,
        total_duration=None,
        title_text=title_text,
        title_font_size=16,
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
    writer = writer_box["writer"]
    assert writer.title_bounds is not None
    x0, y0, x1, y1 = writer.title_bounds
    assert x0 >= 0.0
    assert x1 <= 1.0
    assert y0 >= 0.0
    assert y1 <= 1.0


def test_write_video_keeps_title_font_larger_than_subtitle_font(monkeypatch, tmp_path: Path) -> None:
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt

    title_text = (
        "Best-so-far motif placement improves over sweeps for "
        "project_tfs_lexa_cpxr_baer_rcda_lrp_fur_fnr_acrr_soxr_soxs "
        "chain_1 objective_scalar normalized-llr"
    )
    subtitle_text = "lexA=0.99 · cpxR=0.95"

    class _FontCaptureWriter:
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
            self._fig = None
            self.title_font_size: float | None = None
            self.subtitle_font_size: float | None = None

        def saving(self, fig, path: str, dpi: int):
            del dpi
            self._fig = fig
            return self._SavingContext(path)

        def grab_frame(self):
            if self._fig is None:
                raise AssertionError("Expected figure to be bound before frame capture.")
            self._fig.canvas.draw()
            title_artist = None
            subtitle_artist = None
            for artist in self._fig.texts:
                color = str(artist.get_color()).lower()
                if color == "#374151":
                    title_artist = artist
                elif color == "#4b5563":
                    subtitle_artist = artist
            if title_artist is None or subtitle_artist is None:
                raise AssertionError("Expected both title and subtitle artists in video frame.")
            self.title_font_size = float(title_artist.get_fontsize())
            self.subtitle_font_size = float(subtitle_artist.get_fontsize())

    writer_box: dict[str, _FontCaptureWriter] = {}

    def _writer_factory(*args, **kwargs):
        writer = _FontCaptureWriter(*args, **kwargs)
        writer_box["writer"] = writer
        return writer

    def _fake_render_record(record: Record, *, renderer_name: str, style: Style, palette: Palette):
        del record, renderer_name, style, palette
        fig, ax = plt.subplots(figsize=(2.4, 1.2), dpi=100)
        ax.set_axis_off()
        return fig

    monkeypatch.setattr(animation.writers, "is_available", lambda name: True)
    monkeypatch.setattr(animation, "FFMpegWriter", _writer_factory)
    monkeypatch.setattr("dnadesign.baserender.src.outputs.video.render_record", _fake_render_record)

    records = [
        Record(
            id="frame-1",
            alphabet="DNA",
            sequence="ACGT",
            display=Display(video_subtitle=subtitle_text),
        )
    ]
    output = VideoOutputCfg(
        kind="video",
        path=tmp_path / "title-subtitle-font-hierarchy.mp4",
        fmt="mp4",
        fps=8,
        frames_per_record=1,
        pauses={},
        width_px=None,
        height_px=None,
        aspect_ratio=None,
        total_duration=None,
        title_text=title_text,
        title_font_size=16,
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
    writer = writer_box["writer"]
    assert writer.title_font_size is not None
    assert writer.subtitle_font_size is not None
    assert writer.title_font_size > writer.subtitle_font_size


def test_write_video_can_lock_title_and_subtitle_to_uniform_display_font_size(monkeypatch, tmp_path: Path) -> None:
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt

    title_text = "DenseGen Workspace | Sequence abcdefgh...wxyz | Plan ethanol ciprofloxacin"
    subtitle_text = "Sequence abcdefgh...wxyz | Plan ethanol ciprofloxacin"

    class _FontCaptureWriter:
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
            self._fig = None
            self.title_font_size: float | None = None
            self.subtitle_font_size: float | None = None

        def saving(self, fig, path: str, dpi: int):
            del dpi
            self._fig = fig
            return self._SavingContext(path)

        def grab_frame(self):
            if self._fig is None:
                raise AssertionError("Expected figure to be bound before frame capture.")
            self._fig.canvas.draw()
            title_artist = None
            subtitle_artist = None
            for artist in self._fig.texts:
                color = str(artist.get_color()).lower()
                if color == "#374151":
                    title_artist = artist
                elif color == "#4b5563":
                    subtitle_artist = artist
            if title_artist is None or subtitle_artist is None:
                raise AssertionError("Expected both title and subtitle artists in video frame.")
            self.title_font_size = float(title_artist.get_fontsize())
            self.subtitle_font_size = float(subtitle_artist.get_fontsize())

    writer_box: dict[str, _FontCaptureWriter] = {}

    def _writer_factory(*args, **kwargs):
        writer = _FontCaptureWriter(*args, **kwargs)
        writer_box["writer"] = writer
        return writer

    def _fake_render_record(record: Record, *, renderer_name: str, style: Style, palette: Palette):
        del record, renderer_name, style, palette
        fig, ax = plt.subplots(figsize=(2.8, 1.3), dpi=100)
        ax.set_axis_off()
        return fig

    monkeypatch.setattr(animation.writers, "is_available", lambda name: True)
    monkeypatch.setattr(animation, "FFMpegWriter", _writer_factory)
    monkeypatch.setattr("dnadesign.baserender.src.outputs.video.render_record", _fake_render_record)

    records = [
        Record(
            id="frame-1",
            alphabet="DNA",
            sequence="ACGT",
            display=Display(video_subtitle=subtitle_text),
        )
    ]
    output = VideoOutputCfg(
        kind="video",
        path=tmp_path / "title-subtitle-uniform-font.mp4",
        fmt="mp4",
        fps=8,
        frames_per_record=1,
        pauses={},
        width_px=None,
        height_px=None,
        aspect_ratio=None,
        total_duration=None,
        title_text=title_text,
        title_font_size=18,
        title_align="center",
    )

    out_path = write_video(
        records,
        output=output,
        renderer_name="sequence_rows",
        style=Style(
            dpi=100,
            font_size_seq=18,
            font_size_label=18,
            legend_font_size=18,
            uniform_display_font_size=True,
        ),
        palette=Palette(),
    )

    assert out_path.exists()
    writer = writer_box["writer"]
    assert writer.title_font_size == 18.0
    assert writer.subtitle_font_size == 18.0


def test_write_video_title_avoids_bbox_patch_to_prevent_subtitle_clip(monkeypatch, tmp_path: Path) -> None:
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt

    class _CaptureWriter:
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
            self._fig = None
            self.title_has_bbox_patch: bool | None = None

        def saving(self, fig, path: str, dpi: int):
            del dpi
            self._fig = fig
            return self._SavingContext(path)

        def grab_frame(self):
            if self._fig is None:
                raise AssertionError("Expected figure to be bound before frame capture.")
            self._fig.canvas.draw()
            title_artist = None
            subtitle_artist = None
            for artist in self._fig.texts:
                color = str(artist.get_color()).lower()
                if color == "#374151":
                    title_artist = artist
                elif color == "#4b5563":
                    subtitle_artist = artist
            if title_artist is None or subtitle_artist is None:
                raise AssertionError("Expected both title and subtitle artists in video frame.")
            self.title_has_bbox_patch = title_artist.get_bbox_patch() is not None

    writer_box: dict[str, _CaptureWriter] = {}

    def _writer_factory(*args, **kwargs):
        writer = _CaptureWriter(*args, **kwargs)
        writer_box["writer"] = writer
        return writer

    def _fake_render_record(record: Record, *, renderer_name: str, style: Style, palette: Palette):
        del record, renderer_name, style, palette
        fig, ax = plt.subplots(figsize=(2.8, 1.6), dpi=100)
        ax.set_axis_off()
        return fig

    monkeypatch.setattr(animation.writers, "is_available", lambda name: True)
    monkeypatch.setattr(animation, "FFMpegWriter", _writer_factory)
    monkeypatch.setattr("dnadesign.baserender.src.outputs.video.render_record", _fake_render_record)

    records = [
        Record(
            id="frame-1",
            alphabet="DNA",
            sequence="ACGT",
            display=Display(video_subtitle="lexA=0.85 cpxR=0.82"),
        )
    ]
    output = VideoOutputCfg(
        kind="video",
        path=tmp_path / "title-without-bbox.mp4",
        fmt="mp4",
        fps=8,
        frames_per_record=1,
        pauses={},
        width_px=None,
        height_px=None,
        aspect_ratio=None,
        total_duration=None,
        title_text="Trajectory",
        title_font_size=12,
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
    writer = writer_box["writer"]
    assert writer.title_has_bbox_patch is not None
    assert writer.title_has_bbox_patch is False


def test_write_video_anchors_title_block_to_content_envelope(monkeypatch, tmp_path: Path) -> None:
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt

    class _AnchorCaptureWriter:
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
            self._fig = None
            self.title_bounds: tuple[float, float, float, float] | None = None
            self.subtitle_bounds: tuple[float, float, float, float] | None = None

        def saving(self, fig, path: str, dpi: int):
            del dpi
            self._fig = fig
            return self._SavingContext(path)

        def grab_frame(self):
            if self._fig is None:
                raise AssertionError("Expected figure to be bound before frame capture.")
            self._fig.canvas.draw()
            renderer = self._fig.canvas.get_renderer()
            title_artist = None
            subtitle_artist = None
            for artist in self._fig.texts:
                color = str(artist.get_color()).lower()
                if color == "#374151":
                    title_artist = artist
                elif color == "#4b5563":
                    subtitle_artist = artist
            if title_artist is None or subtitle_artist is None:
                raise AssertionError("Expected both title and subtitle artists in video frame.")
            title_bbox = title_artist.get_window_extent(renderer=renderer).transformed(self._fig.transFigure.inverted())
            subtitle_bbox = subtitle_artist.get_window_extent(renderer=renderer).transformed(
                self._fig.transFigure.inverted()
            )
            self.title_bounds = (
                float(title_bbox.x0),
                float(title_bbox.y0),
                float(title_bbox.x1),
                float(title_bbox.y1),
            )
            self.subtitle_bounds = (
                float(subtitle_bbox.x0),
                float(subtitle_bbox.y0),
                float(subtitle_bbox.x1),
                float(subtitle_bbox.y1),
            )

    writer_box: dict[str, _AnchorCaptureWriter] = {}

    def _writer_factory(*args, **kwargs):
        writer = _AnchorCaptureWriter(*args, **kwargs)
        writer_box["writer"] = writer
        return writer

    def _fake_render_record(record: Record, *, renderer_name: str, style: Style, palette: Palette):
        del record, renderer_name, style, palette
        fig, ax = plt.subplots(figsize=(2.8, 1.6), dpi=100)
        ax.set_axis_off()
        return fig

    monkeypatch.setattr(animation.writers, "is_available", lambda name: True)
    monkeypatch.setattr(animation, "FFMpegWriter", _writer_factory)
    monkeypatch.setattr("dnadesign.baserender.src.outputs.video.render_record", _fake_render_record)
    monkeypatch.setattr(
        "dnadesign.baserender.src.outputs.video._sequence_rows_content_envelope_norms",
        lambda records, style: (0.68, 0.28),
    )

    records = [
        Record(
            id="frame-1",
            alphabet="DNA",
            sequence="ACGT",
            display=Display(video_subtitle="lexA=0.85 cpxR=0.82"),
        )
    ]
    output = VideoOutputCfg(
        kind="video",
        path=tmp_path / "title-anchor.mp4",
        fmt="mp4",
        fps=8,
        frames_per_record=1,
        pauses={},
        width_px=None,
        height_px=None,
        aspect_ratio=None,
        total_duration=None,
        title_text="Trajectory",
        title_font_size=12,
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
    writer = writer_box["writer"]
    assert writer.title_bounds is not None
    assert writer.subtitle_bounds is not None
    _, subtitle_y0, _, subtitle_y1 = writer.subtitle_bounds
    _, title_y0, _, title_y1 = writer.title_bounds
    assert subtitle_y0 >= 0.688
    assert subtitle_y1 <= 0.88
    assert title_y0 >= subtitle_y1
    assert title_y1 <= 0.985


def test_write_video_anchor_respects_high_content_envelope_without_low_cap(monkeypatch, tmp_path: Path) -> None:
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt

    class _AnchorCaptureWriter:
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
            self._fig = None
            self.subtitle_bounds: tuple[float, float, float, float] | None = None

        def saving(self, fig, path: str, dpi: int):
            del dpi
            self._fig = fig
            return self._SavingContext(path)

        def grab_frame(self):
            if self._fig is None:
                raise AssertionError("Expected figure to be bound before frame capture.")
            self._fig.canvas.draw()
            renderer = self._fig.canvas.get_renderer()
            subtitle_artist = None
            for artist in self._fig.texts:
                if str(artist.get_color()).lower() == "#4b5563":
                    subtitle_artist = artist
                    break
            if subtitle_artist is None:
                raise AssertionError("Expected subtitle artist in video frame.")
            subtitle_bbox = subtitle_artist.get_window_extent(renderer=renderer).transformed(
                self._fig.transFigure.inverted()
            )
            self.subtitle_bounds = (
                float(subtitle_bbox.x0),
                float(subtitle_bbox.y0),
                float(subtitle_bbox.x1),
                float(subtitle_bbox.y1),
            )

    writer_box: dict[str, _AnchorCaptureWriter] = {}

    def _writer_factory(*args, **kwargs):
        writer = _AnchorCaptureWriter(*args, **kwargs)
        writer_box["writer"] = writer
        return writer

    def _fake_render_record(record: Record, *, renderer_name: str, style: Style, palette: Palette):
        del record, renderer_name, style, palette
        fig, ax = plt.subplots(figsize=(2.8, 1.6), dpi=100)
        ax.set_axis_off()
        return fig

    monkeypatch.setattr(animation.writers, "is_available", lambda name: True)
    monkeypatch.setattr(animation, "FFMpegWriter", _writer_factory)
    monkeypatch.setattr("dnadesign.baserender.src.outputs.video.render_record", _fake_render_record)
    monkeypatch.setattr(
        "dnadesign.baserender.src.outputs.video._sequence_rows_content_envelope_norms",
        lambda records, style: (0.92, 0.26),
    )

    records = [
        Record(
            id="frame-1",
            alphabet="DNA",
            sequence="ACGT",
            display=Display(video_subtitle="lexA=0.85 cpxR=0.82"),
        )
    ]
    output = VideoOutputCfg(
        kind="video",
        path=tmp_path / "title-anchor-high-envelope.mp4",
        fmt="mp4",
        fps=8,
        frames_per_record=1,
        pauses={},
        width_px=None,
        height_px=None,
        aspect_ratio=None,
        total_duration=None,
        title_text="Trajectory",
        title_font_size=12,
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
    writer = writer_box["writer"]
    assert writer.subtitle_bounds is not None
    _, subtitle_y0, _, subtitle_y1 = writer.subtitle_bounds
    assert subtitle_y0 >= 0.82
    assert subtitle_y1 <= 0.99


def test_write_video_places_subtitle_above_rendered_content(monkeypatch, tmp_path: Path) -> None:
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    class _ContentBoundsWriter:
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
            self._fig = None
            self.subtitle_y0: float | None = None
            self.subtitle_y1: float | None = None
            self.title_y0: float | None = None
            self.title_y1: float | None = None
            self.content_top_norm: float | None = None

        def saving(self, fig, path: str, dpi: int):
            del dpi
            self._fig = fig
            return self._SavingContext(path)

        def grab_frame(self):
            if self._fig is None:
                raise AssertionError("Expected figure to be bound before frame capture.")
            self._fig.canvas.draw()
            renderer = self._fig.canvas.get_renderer()
            subtitle_artist = None
            title_artist = None
            for artist in self._fig.texts:
                color = str(artist.get_color()).lower()
                if color == "#4b5563":
                    subtitle_artist = artist
                elif color == "#374151":
                    title_artist = artist
            if title_artist is None:
                raise AssertionError("Expected title artist in video frame.")
            if subtitle_artist is None:
                raise AssertionError("Expected subtitle artist in video frame.")
            title_bbox = title_artist.get_window_extent(renderer=renderer).transformed(self._fig.transFigure.inverted())
            subtitle_bbox = subtitle_artist.get_window_extent(renderer=renderer).transformed(
                self._fig.transFigure.inverted()
            )
            self.subtitle_y0 = float(subtitle_bbox.y0)
            self.subtitle_y1 = float(subtitle_bbox.y1)
            self.title_y0 = float(title_bbox.y0)
            self.title_y1 = float(title_bbox.y1)

            image_artists = self._fig.axes[0].images
            if not image_artists:
                raise AssertionError("Expected rendered frame image artist.")
            arr = np.asarray(image_artists[0].get_array())
            rgb = arr[:, :, :3]
            ys, _xs = np.where(np.any(rgb < 252, axis=2))
            if ys.size == 0:
                raise AssertionError("Expected non-white rendered content in video frame.")
            top_px = int(ys.min())
            self.content_top_norm = 1.0 - (float(top_px) / float(arr.shape[0]))

    writer_box: dict[str, _ContentBoundsWriter] = {}

    def _writer_factory(*args, **kwargs):
        writer = _ContentBoundsWriter(*args, **kwargs)
        writer_box["writer"] = writer
        return writer

    def _fake_render_record(record: Record, *, renderer_name: str, style: Style, palette: Palette):
        del record, renderer_name, style, palette
        fig, ax = plt.subplots(figsize=(6.0, 1.8), dpi=100)
        ax.set_axis_off()
        ax.add_patch(Rectangle((0.0, 0.74), 1.0, 0.24, transform=ax.transAxes, color="#1f2937"))
        return fig

    monkeypatch.setattr(animation.writers, "is_available", lambda name: True)
    monkeypatch.setattr(animation, "FFMpegWriter", _writer_factory)
    monkeypatch.setattr("dnadesign.baserender.src.outputs.video.render_record", _fake_render_record)

    records = [
        Record(
            id="frame-1",
            alphabet="DNA",
            sequence="ACGT",
            display=Display(video_subtitle="lexA=0.85 cpxR=0.82"),
        )
    ]
    output = VideoOutputCfg(
        kind="video",
        path=tmp_path / "subtitle-above-content.mp4",
        fmt="mp4",
        fps=8,
        frames_per_record=1,
        pauses={},
        width_px=None,
        height_px=None,
        aspect_ratio=None,
        total_duration=None,
        title_text="Trajectory",
        title_font_size=12,
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
    writer = writer_box["writer"]
    assert writer.subtitle_y0 is not None
    assert writer.subtitle_y1 is not None
    assert writer.title_y0 is not None
    assert writer.title_y1 is not None
    assert writer.content_top_norm is not None
    assert writer.subtitle_y0 >= (writer.content_top_norm + 0.008)
    assert writer.subtitle_y0 <= (writer.content_top_norm + 0.03)
    assert writer.title_y0 >= (writer.subtitle_y1 + 0.005)
    assert writer.title_y0 <= (writer.subtitle_y1 + 0.03)
    assert writer.title_y1 <= 0.985


def test_write_video_stacks_header_block_above_high_content_without_collision(monkeypatch, tmp_path: Path) -> None:
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    class _HighContentBoundsWriter:
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
            self._fig = None
            self.subtitle_y0: float | None = None
            self.subtitle_y1: float | None = None
            self.title_y0: float | None = None
            self.content_top_norm: float | None = None

        def saving(self, fig, path: str, dpi: int):
            del dpi
            self._fig = fig
            return self._SavingContext(path)

        def grab_frame(self):
            if self._fig is None:
                raise AssertionError("Expected figure to be bound before frame capture.")
            self._fig.canvas.draw()
            renderer = self._fig.canvas.get_renderer()
            subtitle_artist = None
            title_artist = None
            for artist in self._fig.texts:
                color = str(artist.get_color()).lower()
                if color == "#4b5563":
                    subtitle_artist = artist
                elif color == "#374151":
                    title_artist = artist
            if subtitle_artist is None:
                raise AssertionError("Expected subtitle artist in video frame.")
            if title_artist is None:
                raise AssertionError("Expected title artist in video frame.")
            subtitle_bbox = subtitle_artist.get_window_extent(renderer=renderer).transformed(
                self._fig.transFigure.inverted()
            )
            title_bbox = title_artist.get_window_extent(renderer=renderer).transformed(self._fig.transFigure.inverted())
            self.subtitle_y0 = float(subtitle_bbox.y0)
            self.subtitle_y1 = float(subtitle_bbox.y1)
            self.title_y0 = float(title_bbox.y0)

            image_artists = self._fig.axes[0].images
            if not image_artists:
                raise AssertionError("Expected rendered frame image artist.")
            arr = np.asarray(image_artists[0].get_array())
            rgb = arr[:, :, :3]
            ys, _xs = np.where(np.any(rgb < 252, axis=2))
            if ys.size == 0:
                raise AssertionError("Expected non-white rendered content in video frame.")
            top_px = int(ys.min())
            self.content_top_norm = 1.0 - (float(top_px) / float(arr.shape[0]))

    writer_box: dict[str, _HighContentBoundsWriter] = {}

    def _writer_factory(*args, **kwargs):
        writer = _HighContentBoundsWriter(*args, **kwargs)
        writer_box["writer"] = writer
        return writer

    def _fake_render_record(record: Record, *, renderer_name: str, style: Style, palette: Palette):
        del record, renderer_name, style, palette
        fig, ax = plt.subplots(figsize=(12.0, 4.0), dpi=100)
        ax.set_axis_off()
        ax.add_patch(Rectangle((0.0, 0.55), 1.0, 0.42, transform=ax.transAxes, color="#1f2937"))
        return fig

    monkeypatch.setattr(animation.writers, "is_available", lambda name: True)
    monkeypatch.setattr(animation, "FFMpegWriter", _writer_factory)
    monkeypatch.setattr("dnadesign.baserender.src.outputs.video.render_record", _fake_render_record)

    records = [
        Record(
            id="frame-1",
            alphabet="DNA",
            sequence="ACGT",
            display=Display(video_subtitle="lexA=0.85 cpxR=0.82"),
        )
    ]
    output = VideoOutputCfg(
        kind="video",
        path=tmp_path / "header-above-high-content.mp4",
        fmt="mp4",
        fps=8,
        frames_per_record=1,
        pauses={},
        width_px=None,
        height_px=None,
        aspect_ratio=None,
        total_duration=None,
        title_text="Trajectory",
        title_font_size=12,
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
    writer = writer_box["writer"]
    assert writer.subtitle_y0 is not None
    assert writer.subtitle_y1 is not None
    assert writer.title_y0 is not None
    assert writer.content_top_norm is not None
    assert writer.subtitle_y0 >= (writer.content_top_norm + 0.008)
    assert writer.title_y0 >= (writer.subtitle_y1 + 0.005)
