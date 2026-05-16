"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/latentdna/src/presentation/annotation_layout.py

Shared annotation placement helpers for LatentDNA plot and notebook surfaces.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

AnnotationBox = tuple[float, float, float, float]


@dataclass(frozen=True)
class AnnotationPlacement:
    offset_x: float
    offset_y: float
    ha: str
    va: str
    box: AnnotationBox


def annotation_offsets() -> tuple[tuple[float, float], ...]:
    return (
        (12.0, 12.0),
        (12.0, -18.0),
        (-72.0, 12.0),
        (-72.0, -18.0),
        (20.0, 26.0),
        (-84.0, 26.0),
        (20.0, -34.0),
        (-84.0, -34.0),
        (44.0, 0.0),
        (-98.0, 0.0),
        (56.0, 22.0),
        (56.0, -42.0),
        (-116.0, 22.0),
        (-116.0, -42.0),
        (78.0, 38.0),
        (78.0, -56.0),
        (-138.0, 38.0),
        (-138.0, -56.0),
    )


def _approximate_box_size(label_text: str, *, font_size: float) -> tuple[float, float]:
    text = str(label_text).strip()
    width = max(56.0, min(150.0, 7.2 * len(text) + 18.0))
    height = max(24.0, (font_size * 2.15) + 4.0)
    return width, height


def _preferred_offsets(
    *,
    point_x: float,
    point_y: float,
    x_mid: float,
    y_mid: float,
) -> list[tuple[float, float]]:
    x_sign = 1.0 if point_x >= x_mid else -1.0
    y_sign = 1.0 if point_y >= y_mid else -1.0

    def _sort_key(offset: tuple[float, float]) -> tuple[int, int, int]:
        offset_x, offset_y = offset
        outward_x = 0 if (offset_x == 0.0 or (offset_x * x_sign) > 0.0) else 1
        outward_y = 0 if (offset_y == 0.0 or (offset_y * y_sign) > 0.0) else 1
        return (
            outward_x + outward_y,
            outward_x,
            outward_y,
        )

    return sorted(annotation_offsets(), key=_sort_key)


def _shifted_box(
    *,
    display_x: float,
    display_y: float,
    offset_x: float,
    offset_y: float,
    width: float,
    height: float,
) -> tuple[AnnotationBox, str, str]:
    ha = "left" if offset_x >= 0.0 else "right"
    va = "bottom" if offset_y >= 0.0 else "top"
    anchor_x = display_x + offset_x
    anchor_y = display_y + offset_y
    if ha == "left":
        left = anchor_x
        right = left + width
    else:
        right = anchor_x
        left = right - width
    if va == "bottom":
        bottom = anchor_y
        top = bottom + height
    else:
        top = anchor_y
        bottom = top - height
    return (left, bottom, right, top), ha, va


def _clamp_box(
    box: AnnotationBox,
    *,
    axes_box: Any,
    left_padding_px: float,
    right_padding_px: float,
    top_padding_px: float,
    bottom_padding_px: float,
) -> AnnotationBox:
    left, bottom, right, top = box
    min_left = float(axes_box.x0) + float(left_padding_px)
    max_right = float(axes_box.x1) - float(right_padding_px)
    min_bottom = float(axes_box.y0) + float(bottom_padding_px)
    max_top = float(axes_box.y1) - float(top_padding_px)

    if left < min_left:
        shift = min_left - left
        left += shift
        right += shift
    if right > max_right:
        shift = right - max_right
        left -= shift
        right -= shift
    if bottom < min_bottom:
        shift = min_bottom - bottom
        bottom += shift
        top += shift
    if top > max_top:
        shift = top - max_top
        bottom -= shift
        top -= shift
    return (left, bottom, right, top)


def _intersection_area(left: AnnotationBox, right: AnnotationBox) -> float:
    overlap_left = max(left[0], right[0])
    overlap_bottom = max(left[1], right[1])
    overlap_right = min(left[2], right[2])
    overlap_top = min(left[3], right[3])
    if overlap_right <= overlap_left or overlap_top <= overlap_bottom:
        return 0.0
    return float((overlap_right - overlap_left) * (overlap_top - overlap_bottom))


def _expand_box(box: AnnotationBox, *, padding_px: float) -> AnnotationBox:
    return (
        float(box[0] - padding_px),
        float(box[1] - padding_px),
        float(box[2] + padding_px),
        float(box[3] + padding_px),
    )


def choose_annotation_placement(
    *,
    display_x: float,
    display_y: float,
    label_text: str,
    axes_box: Any,
    placed_boxes: list[AnnotationBox],
    x_mid: float,
    y_mid: float,
    font_size: float,
    left_padding_px: float = 0.0,
    right_padding_px: float = 0.0,
    top_padding_px: float = 4.0,
    bottom_padding_px: float = 4.0,
    box_gap_px: float = 8.0,
) -> AnnotationPlacement:
    width, height = _approximate_box_size(label_text, font_size=font_size)
    best: tuple[tuple[float, float, int, float], AnnotationPlacement] | None = None
    for index, (offset_x, offset_y) in enumerate(
        _preferred_offsets(point_x=display_x, point_y=display_y, x_mid=x_mid, y_mid=y_mid)
    ):
        raw_box, ha, va = _shifted_box(
            display_x=display_x,
            display_y=display_y,
            offset_x=offset_x,
            offset_y=offset_y,
            width=width,
            height=height,
        )
        clamped_box = _clamp_box(
            raw_box,
            axes_box=axes_box,
            left_padding_px=left_padding_px,
            right_padding_px=right_padding_px,
            top_padding_px=top_padding_px,
            bottom_padding_px=bottom_padding_px,
        )
        adjusted_offset_x = raw_box[0]
        adjusted_offset_y = raw_box[1]
        if ha == "left":
            adjusted_offset_x = clamped_box[0]
        else:
            adjusted_offset_x = clamped_box[2]
        if va == "bottom":
            adjusted_offset_y = clamped_box[1]
        else:
            adjusted_offset_y = clamped_box[3]
        adjusted_offset_x -= display_x
        adjusted_offset_y -= display_y
        expanded_box = _expand_box(clamped_box, padding_px=box_gap_px)
        overlap_penalty = sum(
            _intersection_area(expanded_box, _expand_box(placed_box, padding_px=box_gap_px))
            for placed_box in placed_boxes
        )
        movement_penalty = abs(adjusted_offset_x - offset_x) + abs(adjusted_offset_y - offset_y)
        score = (
            1 if overlap_penalty > 0.0 else 0,
            round(overlap_penalty, 4),
            index,
            round(movement_penalty, 4),
        )
        placement = AnnotationPlacement(
            offset_x=float(adjusted_offset_x),
            offset_y=float(adjusted_offset_y),
            ha=ha,
            va=va,
            box=clamped_box,
        )
        if best is None or score < best[0]:
            best = (score, placement)
    if best is None:
        return AnnotationPlacement(offset_x=12.0, offset_y=12.0, ha="left", va="bottom", box=(0, 0, 0, 0))
    return best[1]
