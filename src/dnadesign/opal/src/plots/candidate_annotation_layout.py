"""Collision-safe layout for candidate labels inside Matplotlib axes."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from ..core.utils import ExitCodes, OpalError


def layout_candidate_annotations(
    ax: Any,
    annotations: Sequence[Any],
    point_pixels: np.ndarray,
    *,
    requested_font_size: float,
    max_lanes: int,
) -> None:
    """Use the largest readable font that fits one or two annotation lanes."""

    figure = ax.figure
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    axes_box = ax.get_window_extent(renderer=renderer)
    padding = 4.0
    lower = float(axes_box.y0 + padding)
    upper = float(axes_box.y1 - padding)
    _largest_fitting_layout(
        ax,
        figure,
        annotations,
        point_pixels,
        point_pixels[:, 1].tolist(),
        requested_font_size=float(requested_font_size),
        max_lanes=max_lanes,
        horizontal_order=sorted(
            range(len(annotations)),
            key=lambda index: (float(point_pixels[index, 0]), index),
        ),
        lower=lower,
        upper=upper,
        prefer_right=float(np.median(point_pixels[:, 0])) <= float(axes_box.x0 + axes_box.x1) / 2.0,
    )
    figure.canvas.draw()


def _largest_fitting_layout(
    ax: Any,
    figure: Any,
    annotations: Sequence[Any],
    point_pixels: np.ndarray,
    desired_y: Sequence[float],
    *,
    requested_font_size: float,
    max_lanes: int,
    horizontal_order: Sequence[int],
    lower: float,
    upper: float,
    prefer_right: bool,
) -> tuple[list[Any], list[tuple[list[int], bool, list[float]]]]:
    requested = _placed_trial_layout(
        ax,
        figure,
        annotations,
        point_pixels,
        desired_y,
        font_size=requested_font_size,
        max_lanes=max_lanes,
        horizontal_order=horizontal_order,
        lower=lower,
        upper=upper,
        prefer_right=prefer_right,
    )
    if requested is not None:
        return requested
    floor = min(7.0, requested_font_size)
    fitted = _placed_trial_layout(
        ax,
        figure,
        annotations,
        point_pixels,
        desired_y,
        font_size=floor,
        max_lanes=max_lanes,
        horizontal_order=horizontal_order,
        lower=lower,
        upper=upper,
        prefer_right=prefer_right,
    )
    if fitted is None:
        raise OpalError(
            "Candidate annotations cannot fit inside the plot axes at the minimum readable size.",
            ExitCodes.CONTRACT_VIOLATION,
        )
    low = floor
    high = requested_font_size
    for _ in range(10):
        trial_size = (low + high) / 2.0
        trial = _placed_trial_layout(
            ax,
            figure,
            annotations,
            point_pixels,
            desired_y,
            font_size=trial_size,
            max_lanes=max_lanes,
            horizontal_order=horizontal_order,
            lower=lower,
            upper=upper,
            prefer_right=prefer_right,
        )
        if trial is None:
            high = trial_size
        else:
            low = trial_size
            fitted = trial
    return (
        _placed_trial_layout(
            ax,
            figure,
            annotations,
            point_pixels,
            desired_y,
            font_size=low,
            max_lanes=max_lanes,
            horizontal_order=horizontal_order,
            lower=lower,
            upper=upper,
            prefer_right=prefer_right,
        )
        or fitted
    )


def _placed_trial_layout(
    ax: Any,
    figure: Any,
    annotations: Sequence[Any],
    point_pixels: np.ndarray,
    desired_y: Sequence[float],
    *,
    font_size: float,
    max_lanes: int,
    horizontal_order: Sequence[int],
    lower: float,
    upper: float,
    prefer_right: bool,
) -> tuple[list[Any], list[tuple[list[int], bool, list[float]]]] | None:
    trial = _trial_layout(
        figure,
        annotations,
        desired_y,
        font_size=font_size,
        max_lanes=max_lanes,
        horizontal_order=horizontal_order,
        lower=lower,
        upper=upper,
        prefer_right=prefer_right,
    )
    if trial is None:
        return None
    boxes, lanes = trial
    renderer = figure.canvas.get_renderer()
    axes_box = ax.get_window_extent(renderer=renderer)
    padding = 4.0
    inverse = ax.transData.inverted()
    try:
        for indexes, use_right_lane, centers in lanes:
            for index, center in zip(indexes, centers, strict=True):
                annotation = annotations[index]
                anchor_x, alignment = _horizontal_anchor(
                    float(point_pixels[index, 0]),
                    boxes[index].width,
                    left=float(axes_box.x0 + padding),
                    right=float(axes_box.x1 - padding),
                    gap=12.0,
                    prefer_right=use_right_lane,
                )
                annotation.set_ha(alignment)
                annotation.set_position(tuple(inverse.transform((anchor_x, center))))
    except OpalError:
        return None
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    final_boxes = [annotation.get_bbox_patch().get_window_extent(renderer=renderer) for annotation in annotations]
    if not _boxes_fit_axes(final_boxes, axes_box=axes_box, padding=padding):
        return None
    if _boxes_overlap(final_boxes):
        return None
    return final_boxes, lanes


def _trial_layout(
    figure: Any,
    annotations: Sequence[Any],
    desired_y: Sequence[float],
    *,
    font_size: float,
    max_lanes: int,
    horizontal_order: Sequence[int],
    lower: float,
    upper: float,
    prefer_right: bool,
) -> tuple[list[Any], list[tuple[list[int], bool, list[float]]]] | None:
    for annotation in annotations:
        annotation.set_fontsize(font_size)
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    boxes = [annotation.get_bbox_patch().get_window_extent(renderer=renderer) for annotation in annotations]
    heights = [box.height for box in boxes]
    try:
        lanes = _lane_layout(
            desired_y,
            heights,
            max_lanes=max_lanes,
            horizontal_order=horizontal_order,
            lower=lower,
            upper=upper,
            prefer_right=prefer_right,
        )
    except OpalError:
        return None
    return boxes, lanes


def _lane_layout(
    desired_y: Sequence[float],
    heights: Sequence[float],
    *,
    max_lanes: int,
    horizontal_order: Sequence[int],
    lower: float,
    upper: float,
    prefer_right: bool,
) -> list[tuple[list[int], bool, list[float]]]:
    try:
        centers = _spread_centers(desired_y, heights, lower=lower, upper=upper, gap=3.0)
    except OpalError:
        if max_lanes == 1:
            raise
        split = (len(horizontal_order) + 1) // 2
        specifications = ((list(horizontal_order[:split]), False), (list(horizontal_order[split:]), True))
        return [
            (
                indexes,
                use_right,
                _spread_centers(
                    [desired_y[index] for index in indexes],
                    [heights[index] for index in indexes],
                    lower=lower,
                    upper=upper,
                    gap=3.0,
                ),
            )
            for indexes, use_right in specifications
        ]
    return [(list(range(len(desired_y))), prefer_right, centers)]


def _spread_centers(
    desired: Sequence[float],
    heights: Sequence[float],
    *,
    lower: float,
    upper: float,
    gap: float,
) -> list[float]:
    order = sorted(range(len(desired)), key=lambda index: (float(desired[index]), index))
    required = sum(float(heights[index]) for index in order) + gap * max(0, len(order) - 1)
    if required > upper - lower:
        raise OpalError("Candidate annotations cannot fit inside the plot axes.", ExitCodes.CONTRACT_VIOLATION)
    centers: dict[int, float] = {}
    cursor = lower
    for index in order:
        half_height = float(heights[index]) / 2.0
        center = max(float(desired[index]), cursor + half_height)
        centers[index] = center
        cursor = center + half_height + gap
    cursor = upper
    for index in reversed(order):
        half_height = float(heights[index]) / 2.0
        center = min(centers[index], cursor - half_height)
        centers[index] = center
        cursor = center - half_height - gap
    if centers[order[0]] - float(heights[order[0]]) / 2.0 < lower:
        raise OpalError("Candidate annotations cannot fit inside the plot axes.", ExitCodes.CONTRACT_VIOLATION)
    return [centers[index] for index in range(len(desired))]


def _boxes_fit_axes(boxes: Sequence[Any], *, axes_box: Any, padding: float) -> bool:
    tolerance = 0.5
    left = float(axes_box.x0 + padding - tolerance)
    right = float(axes_box.x1 - padding + tolerance)
    lower = float(axes_box.y0 + padding - tolerance)
    upper = float(axes_box.y1 - padding + tolerance)
    return all(
        float(box.x0) >= left and float(box.x1) <= right and float(box.y0) >= lower and float(box.y1) <= upper
        for box in boxes
    )


def _boxes_overlap(boxes: Sequence[Any]) -> bool:
    return any(
        boxes[left_index].overlaps(boxes[right_index])
        for left_index in range(len(boxes))
        for right_index in range(left_index + 1, len(boxes))
    )


def _horizontal_anchor(
    point_x: float,
    width: float,
    *,
    left: float,
    right: float,
    gap: float,
    prefer_right: bool,
) -> tuple[float, str]:
    if width > right - left:
        raise OpalError("Candidate annotation is wider than the plot axes.", ExitCodes.CONTRACT_VIOLATION)
    right_anchor = point_x + gap
    left_anchor = point_x - gap
    right_fits = right_anchor + width <= right
    left_fits = left_anchor - width >= left
    if (prefer_right and right_fits) or not left_fits:
        return min(max(right_anchor, left), right - width), "left"
    return max(min(left_anchor, right), left + width), "right"


__all__ = ["layout_candidate_annotations"]
