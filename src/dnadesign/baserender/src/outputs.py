"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/baserender/src/outputs.py

Writers for sequence-rows image and video outputs.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path
from typing import Iterable

from .config import ImagesOutputCfg, Style, VideoOutputCfg
from .core import Record, SchemaError
from .render import Palette, render_record


def _safe_stem(raw: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", raw.strip())
    stem = stem.strip("._-")
    return stem or "record"


def _unique_stem(base: str, used: set[str]) -> str:
    if base not in used:
        used.add(base)
        return base
    i = 2
    while True:
        candidate = f"{base}_{i}"
        if candidate not in used:
            used.add(candidate)
            return candidate
        i += 1


def write_images(
    records: Iterable[Record],
    *,
    output: ImagesOutputCfg,
    renderer_name: str,
    style: Style,
    palette: Palette,
) -> Path:
    import matplotlib.pyplot as plt

    out_dir = output.dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    used: set[str] = set()
    count = 0
    for index, record in enumerate(records):
        stem = _safe_stem(record.id if record.id else f"record_{index}")
        name = _unique_stem(stem, used)
        out_path = out_dir / f"{name}.{output.fmt}"

        fig = render_record(record, renderer_name=renderer_name, style=style, palette=palette)
        fig.patch.set_facecolor("white")
        fig.patch.set_alpha(1.0)
        fig.savefig(
            out_path,
            format=output.fmt,
            bbox_inches=None,
            pad_inches=0.0,
            facecolor="white",
        )
        plt.close(fig)
        count += 1

    if count == 0:
        raise SchemaError("No records to render after adapter, transforms, and selection")
    return out_dir


def _even_ceil(value: int) -> int:
    return value if value % 2 == 0 else value + 1


def _target_frame_size(
    *,
    natural_w: int,
    natural_h: int,
    output: VideoOutputCfg,
) -> tuple[int, int]:
    width = natural_w
    height = natural_h
    explicit_size = False

    if output.width_px is not None:
        width = int(output.width_px)
        explicit_size = True
    if output.height_px is not None:
        height = int(output.height_px)
        explicit_size = True
    if output.aspect_ratio is not None:
        explicit_size = True
        ratio = float(output.aspect_ratio)
        if output.width_px is not None and output.height_px is None:
            height = int(round(width / ratio))
        elif output.height_px is not None and output.width_px is None:
            width = int(round(height * ratio))
        else:
            width_candidate = max(width, int(round(height * ratio)))
            height_candidate = int(round(width_candidate / ratio))
            width, height = width_candidate, height_candidate

    width = max(_even_ceil(width), 2)
    height = max(_even_ceil(height), 2)
    if not explicit_size:
        width = max(width, _even_ceil(natural_w))
        height = max(height, _even_ceil(natural_h))
    return width, height


def _scale_rgba_to_fit(arr, *, width: int, height: int):
    h, w = arr.shape[:2]
    if width >= w and height >= h:
        return arr
    scale = min(float(width) / float(w), float(height) / float(h))
    if scale <= 0:
        raise SchemaError("Target frame dimensions must be > 0")
    new_w = max(1, int(round(float(w) * scale)))
    new_h = max(1, int(round(float(h) * scale)))
    try:
        from PIL import Image
    except Exception as exc:  # pragma: no cover - pillow is required in runtime env
        raise SchemaError("Pillow is required to scale oversized video frames") from exc
    image = Image.fromarray(arr, mode="RGBA")
    resized = image.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)
    return resized


def _letterbox_rgba(arr, *, width: int, height: int):
    import numpy as np

    h, w = arr.shape[:2]
    if width < w or height < h:
        arr = np.asarray(_scale_rgba_to_fit(arr, width=width, height=height))
        h, w = arr.shape[:2]
    if width < w or height < h:
        raise SchemaError(f"Unable to fit rendered frame ({w}x{h}) into target ({width}x{height}).")
    out = np.ones((height, width, 4), dtype=arr.dtype) * 255
    y0 = (height - h) // 2
    x0 = (width - w) // 2
    out[y0 : y0 + h, x0 : x0 + w, :] = arr
    return out


def _trim_white_border_rgba(arr, *, threshold: int = 248, pad_px: int = 2):
    import numpy as np

    if arr.ndim != 3 or arr.shape[2] < 3:
        return arr
    rgb = arr[:, :, :3]
    non_white = np.any(rgb < int(threshold), axis=2)
    ys, xs = np.where(non_white)
    if ys.size == 0 or xs.size == 0:
        return arr
    h, w = arr.shape[:2]
    top = max(0, int(ys.min()) - int(pad_px))
    bottom = min(h - 1, int(ys.max()) + int(pad_px))
    left = max(0, int(xs.min()) - int(pad_px))
    right = min(w - 1, int(xs.max()) + int(pad_px))
    return arr[top : bottom + 1, left : right + 1, :]


def _content_bounds_rgba(arr, *, threshold: int = 252):
    import numpy as np

    if arr.ndim != 3 or arr.shape[2] < 3:
        return None
    rgb = arr[:, :, :3]
    non_white = np.any(rgb < int(threshold), axis=2)
    ys, xs = np.where(non_white)
    if ys.size == 0 or xs.size == 0:
        return None
    left = int(xs.min())
    top = int(ys.min())
    right = int(xs.max())
    bottom = int(ys.max())
    return (left, top, right, bottom)


def _union_centered_content_bounds(
    *,
    frame_shapes: list[tuple[int, int]],
    content_bounds: list[tuple[int, int, int, int] | None],
    canvas_width: int,
    canvas_height: int,
    pad_px: int = 2,
) -> tuple[int, int, int, int] | None:
    if len(frame_shapes) != len(content_bounds):
        raise SchemaError("frame_shapes/content_bounds length mismatch while computing video crop bounds")

    union_left: int | None = None
    union_top: int | None = None
    union_right: int | None = None
    union_bottom: int | None = None
    for (w, h), bounds in zip(frame_shapes, content_bounds):
        if bounds is None:
            continue
        left, top, right, bottom = bounds
        x_offset = (int(canvas_width) - int(w)) // 2
        y_offset = (int(canvas_height) - int(h)) // 2
        left_c = int(left) + x_offset
        top_c = int(top) + y_offset
        right_c = int(right) + x_offset
        bottom_c = int(bottom) + y_offset
        if union_left is None:
            union_left = left_c
            union_top = top_c
            union_right = right_c
            union_bottom = bottom_c
            continue
        union_left = min(union_left, left_c)
        union_top = min(union_top, top_c)
        union_right = max(union_right, right_c)
        union_bottom = max(union_bottom, bottom_c)

    if union_left is None or union_top is None or union_right is None or union_bottom is None:
        return None

    pad = max(0, int(pad_px))
    left = max(0, union_left - pad)
    top = max(0, union_top - pad)
    right = min(int(canvas_width) - 1, union_right + pad)
    bottom = min(int(canvas_height) - 1, union_bottom + pad)
    if right < left or bottom < top:
        return None
    return (left, top, right, bottom)


def _pause_frames(record_id: str, *, output: VideoOutputCfg) -> int:
    raw = output.pauses.get(record_id, 0.0)
    pause_seconds = float(raw)
    if pause_seconds < 0:
        raise SchemaError(f"outputs.video.pauses[{record_id!r}] must be >= 0")
    return int(round(pause_seconds * output.fps))


def _sequence_rows_content_radius_px(record: Record, *, style: Style) -> float:
    from .render.layout import compute_layout

    layout = compute_layout(record, style)
    show_two = bool(style.show_reverse_complement and record.alphabet == "DNA")
    centerline = (layout.y_forward + layout.y_reverse) / 2.0 if show_two else layout.y_forward
    top_extent = float(layout.content_top) - float(centerline)
    bottom_extent = float(centerline) - float(layout.content_bottom)
    return float(max(top_extent, bottom_extent))


def _apply_fixed_content_radius(
    records: list[Record],
    *,
    renderer_name: str,
    style: Style,
) -> list[Record]:
    if renderer_name != "sequence_rows":
        return list(records)
    if not records:
        return []
    fixed_radius = max(_sequence_rows_content_radius_px(record, style=style) for record in records)
    out: list[Record] = []
    for record in records:
        meta = dict(record.meta)
        meta["fixed_content_radius_px"] = float(fixed_radius)
        out.append(replace(record, meta=meta))
    return out


def write_video(
    records: Iterable[Record],
    *,
    output: VideoOutputCfg,
    renderer_name: str,
    style: Style,
    palette: Palette,
) -> Path:
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt
    import numpy as np

    materialized = list(records)
    if not materialized:
        raise SchemaError("No records to render after adapter, transforms, and selection")
    if output.fmt != "mp4":
        raise SchemaError("outputs.video.fmt must be 'mp4'")
    if not animation.writers.is_available("ffmpeg"):
        raise SchemaError("FFmpeg writer is not available to Matplotlib")

    out_path = output.path.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    prepared = _apply_fixed_content_radius(materialized, renderer_name=renderer_name, style=style)

    def _render_rgba(record: Record):
        panel = render_record(record, renderer_name=renderer_name, style=style, palette=palette)
        panel.canvas.draw()
        array = np.asarray(panel.canvas.buffer_rgba())
        plt.close(panel)
        return array

    natural_canvas_w = 0
    natural_canvas_h = 0
    frame_shapes: list[tuple[int, int]] = []
    content_bounds: list[tuple[int, int, int, int] | None] = []
    for rec in prepared:
        sample = _render_rgba(rec)
        h, w = sample.shape[:2]
        natural_canvas_w = max(natural_canvas_w, int(w))
        natural_canvas_h = max(natural_canvas_h, int(h))
        frame_shapes.append((int(w), int(h)))
        content_bounds.append(_content_bounds_rgba(sample))

    stable_bounds = _union_centered_content_bounds(
        frame_shapes=frame_shapes,
        content_bounds=content_bounds,
        canvas_width=natural_canvas_w,
        canvas_height=natural_canvas_h,
        pad_px=10,
    )
    if stable_bounds is None:
        natural_w = int(natural_canvas_w)
        natural_h = int(natural_canvas_h)
    else:
        left, top, right, bottom = stable_bounds
        natural_w = int((right - left) + 1)
        natural_h = int((bottom - top) + 1)

    frame_w, frame_h = _target_frame_size(natural_w=natural_w, natural_h=natural_h, output=output)

    frames_per_record = int(output.frames_per_record)
    if output.total_duration is not None:
        pause_total = sum(_pause_frames(rec.id, output=output) for rec in prepared)
        frame_budget = max(1, int(round(float(output.total_duration) * output.fps)) - pause_total)
        frames_per_record = max(1, frame_budget // len(prepared))

    fig, ax = plt.subplots(figsize=(frame_w / style.dpi, frame_h / style.dpi), dpi=style.dpi)
    ax.set_position([0, 0, 1, 1])
    ax.set_axis_off()
    image = ax.imshow(
        np.zeros((frame_h, frame_w, 4), dtype=np.uint8),
        interpolation="nearest",
        origin="upper",
    )
    ax.set_aspect("auto")

    if output.title_text is not None:
        align = str(output.title_align).strip().lower()
        if align == "left":
            x = 0.01
            ha = "left"
        elif align == "right":
            x = 0.99
            ha = "right"
        else:
            x = 0.5
            ha = "center"
        if output.title_font_size is not None:
            title_size = int(output.title_font_size)
        else:
            title_size = max(12, style.font_size_label + 4)
        fig.text(
            x,
            0.955,
            str(output.title_text),
            ha=ha,
            va="top",
            fontsize=title_size,
            family=style.font_label,
            color="#374151",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.93, "pad": 1.8},
            zorder=100,
        )

    writer = animation.FFMpegWriter(
        fps=output.fps,
        codec="libx264",
        extra_args=[
            "-vf",
            "scale=trunc(iw/2)*2:trunc(ih/2)*2,setsar=1",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-profile:v",
            "main",
        ],
    )

    try:
        with writer.saving(fig, str(out_path), dpi=style.dpi):
            for rec in prepared:
                arr = _render_rgba(rec)
                if (arr.shape[1], arr.shape[0]) != (natural_canvas_w, natural_canvas_h):
                    arr = _letterbox_rgba(arr, width=natural_canvas_w, height=natural_canvas_h)

                if stable_bounds is not None:
                    left, top, right, bottom = stable_bounds
                    arr = arr[top : bottom + 1, left : right + 1, :]

                if (arr.shape[1], arr.shape[0]) != (frame_w, frame_h):
                    arr = _letterbox_rgba(arr, width=frame_w, height=frame_h)
                image.set_data(arr)

                repeats = max(1, frames_per_record + _pause_frames(rec.id, output=output))
                for _ in range(repeats):
                    writer.grab_frame()
    finally:
        plt.close(fig)

    return out_path
