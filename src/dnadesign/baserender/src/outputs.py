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

    if output.width_px is not None:
        width = max(width, int(output.width_px))
    if output.height_px is not None:
        height = max(height, int(output.height_px))
    if output.aspect_ratio is not None:
        ratio = float(output.aspect_ratio)
        width_candidate = max(width, int(round(height * ratio)))
        height_candidate = int(round(width_candidate / ratio))
        width, height = width_candidate, height_candidate

    width = max(_even_ceil(width), _even_ceil(natural_w))
    height = max(_even_ceil(height), _even_ceil(natural_h))
    return width, height


def _letterbox_rgba(arr, *, width: int, height: int):
    import numpy as np

    h, w = arr.shape[:2]
    if width < w or height < h:
        raise SchemaError(
            f"Target frame is smaller than rendered frame ({width}x{height} < {w}x{h}); "
            "increase outputs.video.width_px/height_px or adjust aspect."
        )
    out = np.ones((height, width, 4), dtype=arr.dtype) * 255
    y0 = (height - h) // 2
    x0 = (width - w) // 2
    out[y0 : y0 + h, x0 : x0 + w, :] = arr
    return out


def _pause_frames(record_id: str, *, output: VideoOutputCfg) -> int:
    raw = output.pauses.get(record_id, 0.0)
    pause_seconds = float(raw)
    if pause_seconds < 0:
        raise SchemaError(f"outputs.video.pauses[{record_id!r}] must be >= 0")
    return int(round(pause_seconds * output.fps))


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

    first_fig = render_record(materialized[0], renderer_name=renderer_name, style=style, palette=palette)
    first_fig.canvas.draw()
    first_arr = np.asarray(first_fig.canvas.buffer_rgba())
    plt.close(first_fig)
    natural_h, natural_w = first_arr.shape[:2]

    frame_w, frame_h = _target_frame_size(natural_w=natural_w, natural_h=natural_h, output=output)

    frames_per_record = int(output.frames_per_record)
    if output.total_duration is not None:
        pause_total = sum(_pause_frames(rec.id, output=output) for rec in materialized)
        frame_budget = max(1, int(round(float(output.total_duration) * output.fps)) - pause_total)
        frames_per_record = max(1, frame_budget // len(materialized))

    fig, ax = plt.subplots(figsize=(frame_w / style.dpi, frame_h / style.dpi), dpi=style.dpi)
    ax.set_position([0, 0, 1, 1])
    ax.set_axis_off()
    image = ax.imshow(
        np.zeros((frame_h, frame_w, 4), dtype=np.uint8),
        interpolation="nearest",
        origin="upper",
    )
    ax.set_aspect("auto")

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
            for rec in materialized:
                panel = render_record(rec, renderer_name=renderer_name, style=style, palette=palette)
                panel.canvas.draw()
                arr = np.asarray(panel.canvas.buffer_rgba())
                plt.close(panel)

                if (arr.shape[1], arr.shape[0]) != (frame_w, frame_h):
                    arr = _letterbox_rgba(arr, width=frame_w, height=frame_h)
                image.set_data(arr)

                repeats = max(1, frames_per_record + _pause_frames(rec.id, output=output))
                for _ in range(repeats):
                    writer.grab_frame()
    finally:
        plt.close(fig)

    return out_path
