"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/src/public/sequence_panel_layout.py

Pixel-layout helpers for the public fixed-canvas sequence panel.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import math
from typing import Any

from ..core import SchemaError, ensure


def normalize_panel_image(
    image: Any,
    *,
    target_width_px: int,
    target_height_px: int,
    vertical_anchor: str,
    canvas_top_pad_px: int,
    source_anchor_y_px: float | None = None,
) -> tuple[Any, float]:
    """Crop visible content, fit it to a white canvas, and preserve its vertical anchor."""
    import numpy as np
    from PIL import Image

    if int(target_width_px) <= 0 or int(target_height_px) <= 0:
        raise SchemaError("sequence panel target dimensions must be positive")

    rgba = np.asarray(image)
    ensure(rgba.ndim == 3 and rgba.shape[2] in {3, 4}, "sequence panel image must be RGB/RGBA", SchemaError)
    if rgba.shape[2] == 3:
        alpha = np.full(rgba.shape[:2], 255, dtype=np.uint8)
        rgba = np.dstack([rgba[:, :, :3], alpha])

    cropped_anchor_y_px = _validate_source_anchor(source_anchor_y_px, image_height_px=int(rgba.shape[0]))
    alpha = rgba[:, :, 3]
    rgb = rgba[:, :, :3]
    content_mask = ((rgb < 245).any(axis=2)) & (alpha > 0)
    if content_mask.any():
        rgba, cropped_anchor_y_px = _crop_visible_content(
            rgba,
            content_mask=content_mask,
            source_anchor_y_px=cropped_anchor_y_px,
        )

    if cropped_anchor_y_px is not None:
        ensure(
            0.0 <= cropped_anchor_y_px <= float(rgba.shape[0]),
            "sequence panel source anchor must lie within the rendered image",
            SchemaError,
        )

    anchor = str(vertical_anchor).strip().lower()
    if anchor not in {"top", "center", "bottom"}:
        raise SchemaError("sequence panel vertical_anchor must be 'top', 'center', or 'bottom'")

    source = Image.fromarray(rgba.astype(np.uint8, copy=False))
    resized = _fit_source_image(
        source,
        target_width_px=int(target_width_px),
        target_height_px=int(target_height_px),
        vertical_anchor=anchor,
        source_anchor_y_px=cropped_anchor_y_px,
    )
    canvas = Image.new("RGBA", (int(target_width_px), int(target_height_px)), (255, 255, 255, 255))
    x = (canvas.width - resized.width) // 2
    y, normalized_anchor_y_px = _vertical_placement(
        source_height_px=source.height,
        resized_height_px=resized.height,
        canvas_height_px=canvas.height,
        vertical_anchor=anchor,
        canvas_top_pad_px=int(canvas_top_pad_px),
        source_anchor_y_px=cropped_anchor_y_px,
    )
    canvas.alpha_composite(resized, dest=(x, y))
    return np.asarray(canvas), normalized_anchor_y_px


def sequence_center_y_px(fig: Any) -> float:
    """Return the sequence-row midpoint in top-origin figure pixels."""
    axes = [axis for axis in fig.axes if hasattr(axis, "_dnadesign_sequence_center_y")]
    ensure(len(axes) == 1, "sequence panel renderer must expose exactly one strand anchor", SchemaError)
    axis = axes[0]
    anchor_data_y = float(axis._dnadesign_sequence_center_y)
    _x_px, anchor_from_bottom_px = axis.transData.transform((0.0, anchor_data_y))
    _width_px, height_px = fig.canvas.get_width_height()
    return float(height_px) - float(anchor_from_bottom_px)


def _validate_source_anchor(source_anchor_y_px: float | None, *, image_height_px: int) -> float | None:
    if source_anchor_y_px is None:
        return None
    anchor = float(source_anchor_y_px)
    ensure(
        math.isfinite(anchor) and 0.0 <= anchor <= float(image_height_px),
        "sequence panel source anchor must lie within the rendered image",
        SchemaError,
    )
    return anchor


def _crop_visible_content(rgba, *, content_mask, source_anchor_y_px: float | None):
    import numpy as np

    ys, xs = np.where(content_mask)
    pad = 8
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(rgba.shape[0], int(ys.max()) + pad + 1)
    if source_anchor_y_px is not None:
        y0 = min(y0, max(0, int(math.floor(source_anchor_y_px))))
        y1 = max(y1, min(rgba.shape[0], int(math.ceil(source_anchor_y_px))))
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(rgba.shape[1], int(xs.max()) + pad + 1)
    cropped = rgba[y0:y1, x0:x1, :]
    cropped_anchor = None if source_anchor_y_px is None else source_anchor_y_px - float(y0)
    return cropped, cropped_anchor


def _fit_source_image(
    source,
    *,
    target_width_px: int,
    target_height_px: int,
    vertical_anchor: str,
    source_anchor_y_px: float | None,
):
    from PIL import Image

    width_scale = target_width_px / max(source.width, 1)
    height_scale = target_height_px / max(source.height, 1)
    if vertical_anchor == "center" and source_anchor_y_px is not None:
        upper_extent = max(1.0, source_anchor_y_px)
        lower_extent = max(1.0, float(source.height) - source_anchor_y_px)
        height_scale = max(1.0, float(target_height_px) / 2.0) / max(upper_extent, lower_extent)
    scale = min(width_scale, height_scale)
    return source.resize(
        (max(1, int(source.width * scale)), max(1, int(source.height * scale))),
        Image.Resampling.LANCZOS,
    )


def _vertical_placement(
    *,
    source_height_px: int,
    resized_height_px: int,
    canvas_height_px: int,
    vertical_anchor: str,
    canvas_top_pad_px: int,
    source_anchor_y_px: float | None,
) -> tuple[int, float]:
    if vertical_anchor == "top":
        y = min(max(0, canvas_top_pad_px), max(0, canvas_height_px - resized_height_px))
    elif vertical_anchor == "bottom":
        y = max(0, canvas_height_px - resized_height_px)
    elif source_anchor_y_px is None:
        y = (canvas_height_px - resized_height_px) // 2
    else:
        resized_anchor_y_px = source_anchor_y_px * (float(resized_height_px) / float(source_height_px))
        y = int(round((float(canvas_height_px) / 2.0) - resized_anchor_y_px))
        y = min(max(0, y), max(0, canvas_height_px - resized_height_px))

    if source_anchor_y_px is None:
        normalized_anchor_y_px = float(y) + (float(resized_height_px) / 2.0)
    else:
        normalized_anchor_y_px = float(y) + (source_anchor_y_px * (float(resized_height_px) / float(source_height_px)))
    return y, normalized_anchor_y_px


__all__ = ["normalize_panel_image", "sequence_center_y_px"]
