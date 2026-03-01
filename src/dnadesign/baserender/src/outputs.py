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
import textwrap
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


def _letterbox_rgba(arr, *, width: int, height: int, vertical_align: str = "center"):
    import numpy as np

    h, w = arr.shape[:2]
    if width < w or height < h:
        arr = np.asarray(_scale_rgba_to_fit(arr, width=width, height=height))
        h, w = arr.shape[:2]
    if width < w or height < h:
        raise SchemaError(f"Unable to fit rendered frame ({w}x{h}) into target ({width}x{height}).")
    out = np.ones((height, width, 4), dtype=arr.dtype) * 255
    align = str(vertical_align).strip().lower()
    if align == "bottom":
        y0 = height - h
    elif align == "top":
        y0 = 0
    else:
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


def _scaled_dimensions_to_fit(
    *,
    source_width: int,
    source_height: int,
    target_width: int,
    target_height: int,
) -> tuple[int, int]:
    if source_width <= 0 or source_height <= 0:
        raise SchemaError("source dimensions must be > 0 while computing rendered content envelope.")
    if target_width <= 0 or target_height <= 0:
        raise SchemaError("target dimensions must be > 0 while computing rendered content envelope.")
    if source_width <= target_width and source_height <= target_height:
        return int(source_width), int(source_height)
    scale = min(float(target_width) / float(source_width), float(target_height) / float(source_height))
    if scale <= 0.0:
        raise SchemaError("Invalid non-positive scale while computing rendered content envelope.")
    scaled_width = max(1, int(round(float(source_width) * scale)))
    scaled_height = max(1, int(round(float(source_height) * scale)))
    return int(scaled_width), int(scaled_height)


def _rendered_content_top_norm_for_video_frame(
    *,
    frame_shapes: list[tuple[int, int]],
    content_bounds: list[tuple[int, int, int, int] | None],
    canvas_width: int,
    canvas_height: int,
    stable_bounds: tuple[int, int, int, int] | None,
    frame_width: int,
    frame_height: int,
    vertical_align: str,
    rendered_content_scale: float = 1.0,
) -> float | None:
    if len(frame_shapes) != len(content_bounds):
        raise SchemaError("frame_shapes/content_bounds length mismatch while computing rendered content envelope")
    if not frame_shapes:
        return None
    crop_top = int(stable_bounds[1]) if stable_bounds is not None else 0
    crop_height = int((stable_bounds[3] - stable_bounds[1] + 1) if stable_bounds is not None else canvas_height)
    crop_width = int((stable_bounds[2] - stable_bounds[0] + 1) if stable_bounds is not None else canvas_width)
    if crop_width <= 0 or crop_height <= 0:
        raise SchemaError("Stable crop bounds produced non-positive dimensions for video frame envelope.")
    scaled_width, scaled_height = _scaled_dimensions_to_fit(
        source_width=int(crop_width),
        source_height=int(crop_height),
        target_width=int(frame_width),
        target_height=int(frame_height),
    )
    content_scale = float(rendered_content_scale)
    if content_scale <= 0.0 or content_scale > 1.0:
        raise SchemaError("rendered_content_scale must satisfy 0 < rendered_content_scale <= 1")
    scaled_content_height = float(scaled_height) * content_scale
    scale_y = float(scaled_content_height) / float(crop_height)
    align = str(vertical_align).strip().lower()
    if align == "bottom":
        y_offset = float(frame_height) - float(scaled_content_height)
    elif align == "top":
        y_offset = 0.0
    else:
        y_offset = (float(frame_height) - float(scaled_content_height)) / 2.0

    top_norm_max = 0.0
    found_content = False
    for (width, height), bounds in zip(frame_shapes, content_bounds):
        if bounds is None:
            continue
        found_content = True
        top_px = int(bounds[1])
        centered_top = top_px + ((int(canvas_height) - int(height)) // 2)
        top_after_crop = max(0.0, float(centered_top - crop_top))
        top_scaled = float(top_after_crop) * scale_y
        top_final_px = float(y_offset) + float(top_scaled)
        top_norm = 1.0 - (top_final_px / float(frame_height))
        top_norm_max = max(top_norm_max, float(top_norm))
    if not found_content:
        return None
    return max(0.0, min(1.0, float(top_norm_max)))


def _pause_frames(record_id: str, *, output: VideoOutputCfg) -> int:
    raw = output.pauses.get(record_id, 0.0)
    pause_seconds = float(raw)
    if pause_seconds < 0:
        raise SchemaError(f"outputs.video.pauses[{record_id!r}] must be >= 0")
    return int(round(pause_seconds * output.fps))


def _sequence_rows_content_extents_px(record: Record, *, style: Style) -> tuple[float, float]:
    from .render.layout import compute_layout

    layout = compute_layout(record, style)
    show_two = bool(style.show_reverse_complement and record.alphabet == "DNA")
    centerline = (layout.y_forward + layout.y_reverse) / 2.0 if show_two else layout.y_forward
    top_extent = float(layout.content_top) - float(centerline)
    bottom_extent = float(centerline) - float(layout.content_bottom)
    return float(top_extent), float(bottom_extent)


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
    fixed_top_extent = 0.0
    fixed_bottom_extent = 0.0
    for record in records:
        top_extent, bottom_extent = _sequence_rows_content_extents_px(record, style=style)
        fixed_top_extent = max(fixed_top_extent, float(top_extent))
        fixed_bottom_extent = max(fixed_bottom_extent, float(bottom_extent))
    out: list[Record] = []
    for record in records:
        meta = dict(record.meta)
        meta["fixed_content_top_extent_px"] = float(fixed_top_extent)
        meta["fixed_content_bottom_extent_px"] = float(fixed_bottom_extent)
        out.append(replace(record, meta=meta))
    return out


def _sequence_rows_layout_context(record: Record, *, style: Style):
    from .render.layout import compute_layout

    fixed_top_raw = record.meta.get("fixed_content_top_extent_px")
    fixed_bottom_raw = record.meta.get("fixed_content_bottom_extent_px")
    fixed_top_extent: float | None = None
    fixed_bottom_extent: float | None = None
    if fixed_top_raw is not None:
        try:
            fixed_top_extent = float(fixed_top_raw)
        except Exception as exc:
            raise SchemaError("record.meta.fixed_content_top_extent_px must be numeric when set") from exc
    if fixed_bottom_raw is not None:
        try:
            fixed_bottom_extent = float(fixed_bottom_raw)
        except Exception as exc:
            raise SchemaError("record.meta.fixed_content_bottom_extent_px must be numeric when set") from exc
    fixed_radius_raw = record.meta.get("fixed_content_radius_px")
    fixed_radius: float | None = None
    if fixed_radius_raw is not None:
        try:
            fixed_radius = float(fixed_radius_raw)
        except Exception as exc:
            raise SchemaError("record.meta.fixed_content_radius_px must be numeric when set") from exc
    extra_bottom_padding_raw = record.meta.get("video_extra_bottom_padding_px")
    extra_bottom_padding = 0.0
    if extra_bottom_padding_raw is not None:
        try:
            extra_bottom_padding = float(extra_bottom_padding_raw)
        except Exception as exc:
            raise SchemaError("record.meta.video_extra_bottom_padding_px must be numeric when set") from exc
    return compute_layout(
        record,
        style,
        fixed_content_top_extent_px=fixed_top_extent,
        fixed_content_bottom_extent_px=fixed_bottom_extent,
        fixed_content_radius_px=fixed_radius,
        extra_bottom_padding_px=extra_bottom_padding,
    )


def _sequence_rows_actual_content_bounds_px(record: Record, *, style: Style) -> tuple[float, float, float]:
    layout = _sequence_rows_layout_context(record, style=style)
    top = max(
        float(layout.y_forward + layout.sequence_extent_up),
        float(layout.y_reverse + layout.sequence_extent_up),
    )
    bottom = min(
        float(layout.y_forward - layout.sequence_extent_down),
        float(layout.y_reverse - layout.sequence_extent_down),
    )
    for placement in layout.placements:
        top = max(top, float(placement.y + (placement.h / 2.0)))
        bottom = min(bottom, float(placement.y - (placement.h / 2.0)))
    for y0 in layout.motif_logo_y0_by_effect.values():
        top = max(top, float(y0 + layout.motif_logo_height))
        bottom = min(bottom, float(y0))
    return top, bottom, float(layout.height)


def _sequence_rows_content_envelope_norms(records: list[Record], style: Style) -> tuple[float, float]:
    if not records:
        raise SchemaError("Cannot compute sequence-rows content envelope with no records.")
    top_norm = 0.0
    bottom_norm = 1.0
    for record in records:
        top_px, bottom_px, height_px = _sequence_rows_actual_content_bounds_px(record, style=style)
        if height_px <= 0:
            raise SchemaError("Sequence-rows layout height must be > 0 for video envelope.")
        top_norm = max(top_norm, float(top_px / height_px))
        bottom_norm = min(bottom_norm, float(bottom_px / height_px))
    top_norm = max(0.0, min(1.0, top_norm))
    bottom_norm = max(0.0, min(1.0, bottom_norm))
    if not bottom_norm < top_norm:
        raise SchemaError("Sequence-rows content envelope is invalid (bottom must be < top).")
    return float(top_norm), float(bottom_norm)


def _apply_sequence_rows_content_envelope(
    records: list[Record],
    *,
    content_top_norm: float,
    content_bottom_norm: float,
) -> list[Record]:
    out: list[Record] = []
    for record in records:
        meta = dict(record.meta)
        meta["video_content_top_norm"] = float(content_top_norm)
        meta["video_content_bottom_norm"] = float(content_bottom_norm)
        out.append(replace(record, meta=meta))
    return out


def _sequence_rows_required_extra_bottom_padding_px(
    records: list[Record],
    *,
    style: Style,
    target_bottom_norm: float,
) -> float:
    if not records:
        return 0.0
    target = float(target_bottom_norm)
    if target <= 0.0 or target >= 1.0:
        raise SchemaError("target_bottom_norm must be in (0, 1) for sequence-rows video padding.")
    required = 0.0
    for record in records:
        _top_px, bottom_px, height_px = _sequence_rows_actual_content_bounds_px(record, style=style)
        candidate = (target * float(height_px) - float(bottom_px)) / (1.0 - target)
        required = max(required, float(candidate))
    return max(0.0, float(required))


def _apply_sequence_rows_extra_bottom_padding(records: list[Record], *, extra_bottom_padding_px: float) -> list[Record]:
    out: list[Record] = []
    extra = max(0.0, float(extra_bottom_padding_px))
    for record in records:
        meta = dict(record.meta)
        meta["video_extra_bottom_padding_px"] = extra
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
    content_top_norm: float | None = None
    if renderer_name == "sequence_rows" and prepared:
        content_top_norm, content_bottom_norm = _sequence_rows_content_envelope_norms(prepared, style=style)
        prepared = _apply_sequence_rows_content_envelope(
            prepared,
            content_top_norm=float(content_top_norm),
            content_bottom_norm=float(content_bottom_norm),
        )

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
        pad_px=6,
    )
    if stable_bounds is None:
        natural_w = int(natural_canvas_w)
        natural_h = int(natural_canvas_h)
    else:
        left, top, right, bottom = stable_bounds
        natural_w = int((right - left) + 1)
        natural_h = int((bottom - top) + 1)

    subtitle_values = [str(record.display.video_subtitle or "").strip() for record in prepared]
    dynamic_subtitle_enabled = any(subtitle_values)
    has_title = output.title_text is not None
    has_header_block = bool(has_title or dynamic_subtitle_enabled)
    frame_vertical_align = "bottom" if has_header_block else "center"
    frame_w, frame_h = _target_frame_size(natural_w=natural_w, natural_h=natural_h, output=output)
    if has_header_block:
        min_header_frame_height = _even_ceil(int(round(float(style.dpi) * 1.2)))
        frame_h = max(int(frame_h), int(min_header_frame_height))
    rendered_content_scale = 0.96 if renderer_name == "sequence_rows" else 1.0
    rendered_content_top_norm = _rendered_content_top_norm_for_video_frame(
        frame_shapes=frame_shapes,
        content_bounds=content_bounds,
        canvas_width=natural_canvas_w,
        canvas_height=natural_canvas_h,
        stable_bounds=stable_bounds,
        frame_width=frame_w,
        frame_height=frame_h,
        vertical_align=frame_vertical_align,
        rendered_content_scale=rendered_content_scale,
    )
    can_expand_frame_height = output.height_px is None and output.aspect_ratio is None
    if (
        can_expand_frame_height
        and has_title
        and dynamic_subtitle_enabled
        and rendered_content_top_norm is not None
        and rendered_content_top_norm > 0.84
    ):
        expansion_scale = min(1.18, max(1.0, float(rendered_content_top_norm) / 0.84))
        expanded_height = _even_ceil(int(round(float(frame_h) * float(expansion_scale))))
        if expanded_height > frame_h:
            frame_h = int(expanded_height)
            rendered_content_top_norm = _rendered_content_top_norm_for_video_frame(
                frame_shapes=frame_shapes,
                content_bounds=content_bounds,
                canvas_width=natural_canvas_w,
                canvas_height=natural_canvas_h,
                stable_bounds=stable_bounds,
                frame_width=frame_w,
                frame_height=frame_h,
                vertical_align=frame_vertical_align,
                rendered_content_scale=rendered_content_scale,
            )
    if rendered_content_top_norm is not None:
        content_top_norm = float(rendered_content_top_norm)

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

    title_artist = None
    if output.title_text is not None:
        title_artist = fig.text(
            x,
            0.985,
            str(output.title_text),
            ha=ha,
            va="top",
            fontsize=title_size,
            family=style.font_label,
            color="#374151",
            zorder=100,
        )

    subtitle_artist = None
    subtitle_layout_text = ""
    if dynamic_subtitle_enabled:
        subtitle_layout_text = max(
            (value for value in subtitle_values if value),
            key=lambda value: (
                value.count("\n") + 1,
                max((len(line) for line in value.splitlines()), default=0),
                len(value),
            ),
        )
    if dynamic_subtitle_enabled:
        subtitle_size = max(7, title_size - 2)
        if title_artist is not None:
            subtitle_size = min(subtitle_size, max(7, int(title_artist.get_fontsize()) - 1))
        subtitle_y = 0.985
        subtitle_artist = fig.text(
            x,
            subtitle_y,
            subtitle_layout_text,
            ha=ha,
            va="top",
            fontsize=subtitle_size,
            family=style.font_label,
            color="#4B5563",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.93, "pad": 1.5},
            zorder=99,
        )
    layout_artists = [artist for artist in (title_artist, subtitle_artist) if artist is not None]

    def _fit_text_layout() -> None:
        if not layout_artists:
            return
        side_margin = 0.01
        top_margin = 0.024
        line_gap = 0.008
        min_title_font = 8.0
        min_subtitle_font = 7.0

        def _wrap_artist_text_to_fit(artist, *, renderer) -> bool:
            raw_text = str(artist.get_text())
            if not raw_text.strip():
                return False
            normalized = " ".join(part.strip() for part in raw_text.splitlines() if part.strip())
            if not normalized:
                return False
            bbox = artist.get_window_extent(renderer=renderer).transformed(fig.transFigure.inverted())
            available_width = max(1.0e-6, 1.0 - (2.0 * side_margin))
            if bbox.width <= 0:
                return False
            estimated_chars = int(round(len(normalized) * (available_width / float(bbox.width))))
            width = min(len(normalized), max(10, estimated_chars))
            for candidate_width in range(width, 9, -2):
                wrapped = "\n".join(
                    textwrap.wrap(
                        normalized,
                        width=candidate_width,
                        break_long_words=True,
                        break_on_hyphens=False,
                    )
                )
                if not wrapped:
                    continue
                artist.set_text(wrapped)
                fig.canvas.draw()
                wrapped_renderer = fig.canvas.get_renderer()
                wrapped_bbox = artist.get_window_extent(renderer=wrapped_renderer).transformed(
                    fig.transFigure.inverted()
                )
                if wrapped_bbox.x0 >= side_margin and wrapped_bbox.x1 <= (1.0 - side_margin):
                    return True
            artist.set_text(raw_text)
            return False

        for _ in range(20):
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            changed = False
            for artist in layout_artists:
                min_font = min_title_font if artist is title_artist else min_subtitle_font
                bbox = artist.get_window_extent(renderer=renderer).transformed(fig.transFigure.inverted())
                out_of_bounds = bbox.x0 < side_margin or bbox.x1 > (1.0 - side_margin)
                if out_of_bounds:
                    if float(artist.get_fontsize()) > min_font:
                        artist.set_fontsize(max(min_font, float(artist.get_fontsize()) - 1.0))
                        changed = True
                    elif _wrap_artist_text_to_fit(artist, renderer=renderer):
                        changed = True
            if title_artist is not None and subtitle_artist is not None:
                title_font = float(title_artist.get_fontsize())
                subtitle_font = float(subtitle_artist.get_fontsize())
                subtitle_cap = max(min_subtitle_font, title_font - 1.0)
                if subtitle_font > (subtitle_cap + 1.0e-6):
                    subtitle_artist.set_fontsize(subtitle_cap)
                    changed = True
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            if title_artist is not None:
                title_bbox = title_artist.get_window_extent(renderer=renderer).transformed(fig.transFigure.inverted())
                title_top_target = 1.0 - top_margin
                title_y = float(title_artist.get_position()[1]) + (title_top_target - float(title_bbox.y1))
                if abs(title_y - float(title_artist.get_position()[1])) > 1.0e-6:
                    title_artist.set_y(title_y)
                    changed = True
            if subtitle_artist is not None:
                if title_artist is not None:
                    title_bbox = title_artist.get_window_extent(renderer=renderer).transformed(
                        fig.transFigure.inverted()
                    )
                    subtitle_target_y = float(title_bbox.y0) - line_gap
                else:
                    subtitle_bbox = subtitle_artist.get_window_extent(renderer=renderer).transformed(
                        fig.transFigure.inverted()
                    )
                    subtitle_target_y = float(subtitle_artist.get_position()[1]) + (
                        (1.0 - top_margin) - float(subtitle_bbox.y1)
                    )
                if abs(subtitle_target_y - float(subtitle_artist.get_position()[1])) > 1.0e-6:
                    subtitle_artist.set_y(subtitle_target_y)
                    changed = True
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            top_bound = max(
                artist.get_window_extent(renderer=renderer).transformed(fig.transFigure.inverted()).y1
                for artist in layout_artists
            )
            if top_bound > (1.0 - top_margin):
                shift = float(top_bound - (1.0 - top_margin))
                for artist in layout_artists:
                    artist.set_y(float(artist.get_position()[1]) - shift)
                changed = True
            if not changed:
                break

    def _anchor_text_layout_to_content() -> None:
        if not layout_artists:
            return
        if content_top_norm is None:
            return
        top_margin = 0.020
        content_gap = 0.008
        line_gap = 0.006
        min_title_font = 8.0
        min_subtitle_font = 7.0
        max_top = float(1.0 - top_margin)
        target_bottom = max(float(content_top_norm) + content_gap, 0.08)
        if target_bottom >= max_top:
            target_bottom = max(0.08, max_top - 1.0e-3)

        def _position_block() -> tuple[float, float]:
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            if subtitle_artist is not None:
                subtitle_bbox = subtitle_artist.get_window_extent(renderer=renderer).transformed(
                    fig.transFigure.inverted()
                )
                subtitle_target_y = float(subtitle_artist.get_position()[1]) + (target_bottom - float(subtitle_bbox.y0))
                subtitle_artist.set_y(subtitle_target_y)
                fig.canvas.draw()
                renderer = fig.canvas.get_renderer()
            if title_artist is not None and subtitle_artist is not None:
                title_bbox = title_artist.get_window_extent(renderer=renderer).transformed(fig.transFigure.inverted())
                subtitle_bbox = subtitle_artist.get_window_extent(renderer=renderer).transformed(
                    fig.transFigure.inverted()
                )
                desired_title_bottom = float(subtitle_bbox.y1) + line_gap
                title_target_y = float(title_artist.get_position()[1]) + (desired_title_bottom - float(title_bbox.y0))
                title_artist.set_y(title_target_y)
            elif title_artist is not None:
                title_bbox = title_artist.get_window_extent(renderer=renderer).transformed(fig.transFigure.inverted())
                title_target_y = float(title_artist.get_position()[1]) + (target_bottom - float(title_bbox.y0))
                title_artist.set_y(title_target_y)
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            bboxes = [
                artist.get_window_extent(renderer=renderer).transformed(fig.transFigure.inverted())
                for artist in layout_artists
            ]
            block_top = max(float(bbox.y1) for bbox in bboxes)
            block_bottom = min(float(bbox.y0) for bbox in bboxes)
            return float(block_bottom), float(block_top)

        for _ in range(20):
            block_bottom, block_top = _position_block()
            if block_top <= (max_top + 1.0e-6):
                break
            available_height = float(max_top - target_bottom)
            block_height = float(block_top - block_bottom)
            if available_height <= 0.0 or block_height <= 0.0:
                break
            scale = max(0.70, min(0.98, float(available_height / block_height)))
            changed = False
            if title_artist is not None:
                current = float(title_artist.get_fontsize())
                updated = max(min_title_font, current * scale)
                if abs(updated - current) > 1.0e-6:
                    title_artist.set_fontsize(updated)
                    changed = True
            if subtitle_artist is not None:
                title_font = (
                    float(title_artist.get_fontsize()) if title_artist is not None else (min_subtitle_font + 1.0)
                )
                subtitle_cap = max(min_subtitle_font, title_font - 1.0)
                current = float(subtitle_artist.get_fontsize())
                updated = max(min_subtitle_font, min(subtitle_cap, current * scale))
                if abs(updated - current) > 1.0e-6:
                    subtitle_artist.set_fontsize(updated)
                    changed = True
            if not changed:
                break

        _block_bottom, block_top = _position_block()
        if block_top > max_top:
            shift = float(block_top - max_top)
            for artist in layout_artists:
                artist.set_y(float(artist.get_position()[1]) - shift)

    _fit_text_layout()
    _anchor_text_layout_to_content()
    if subtitle_artist is not None:
        subtitle_artist.set_text("")
        subtitle_artist.set_visible(False)

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
                    arr = _letterbox_rgba(arr, width=natural_canvas_w, height=natural_canvas_h, vertical_align="center")

                if stable_bounds is not None:
                    left, top, right, bottom = stable_bounds
                    arr = arr[top : bottom + 1, left : right + 1, :]

                if rendered_content_scale < 1.0:
                    scaled_width = max(1, int(round(float(arr.shape[1]) * float(rendered_content_scale))))
                    scaled_height = max(1, int(round(float(arr.shape[0]) * float(rendered_content_scale))))
                    arr = np.asarray(_scale_rgba_to_fit(arr, width=scaled_width, height=scaled_height))

                if (arr.shape[1], arr.shape[0]) != (frame_w, frame_h):
                    arr = _letterbox_rgba(arr, width=frame_w, height=frame_h, vertical_align=frame_vertical_align)
                image.set_data(arr)
                if subtitle_artist is not None:
                    subtitle_text = str(rec.display.video_subtitle or "").strip()
                    subtitle_artist.set_text(subtitle_text)
                    subtitle_artist.set_visible(subtitle_text != "")

                repeats = max(1, frames_per_record + _pause_frames(rec.id, output=output))
                for _ in range(repeats):
                    writer.grab_frame()
    finally:
        plt.close(fig)

    return out_path
