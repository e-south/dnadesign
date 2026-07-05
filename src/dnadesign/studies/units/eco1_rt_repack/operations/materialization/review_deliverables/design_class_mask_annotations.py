"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/review_deliverables/design_class_mask_annotations.py

RT annotation span drawing for design-class mask overview plots.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any

from matplotlib.patches import FancyBboxPatch
from matplotlib.transforms import ScaledTranslation

from .rt_annotation_context import RTAnnotationContext, RTAnnotationFeature

_TRACK_CONTEXT = "retron_rt_context_spans"
_TRACK_CORE_INTERVALS = "retron_rt_core_intervals"
_TRACK_MOTIF_ANCHORS = "retron_rt_motif_anchors"
_CONTEXT_FILL = "#e7d4ee"
_CORE_INTERVAL_FILL = "#d7ecf5"
_MOTIF_FILL = "#f4d7bd"
_CONTEXT_TEXT = "#6f4c7d"
_CORE_INTERVAL_TEXT = "#28566a"
_MOTIF_TEXT = "#8a4a11"
_RT_SPAN_LABEL_SIZE = 7.0
_CONTEXT_SPAN_ALPHA = 0.30
_CORE_INTERVAL_SPAN_ALPHA = 0.30
_MOTIF_SPAN_ALPHA = 0.42
_CONTEXT_LABEL_OFFSET_POINTS = 32.0
_CORE_INTERVAL_LABEL_OFFSET_POINTS = 22.0
_MOTIF_LABEL_OFFSET_POINTS = 12.0
MASK_ANNOTATION_SPAN_ZORDER = 0.5


def add_rt_annotation_context(
    ax: Any,
    positions: list[int],
    *,
    row_count: int,
    context: RTAnnotationContext,
) -> None:
    position_to_index = {position: index for index, position in enumerate(positions)}
    for feature in context.features_for_track(_TRACK_CONTEXT):
        _add_context_span(
            ax,
            feature,
            position_to_index=position_to_index,
            row_count=row_count,
            fill_color=_CONTEXT_FILL,
            text_color=_CONTEXT_TEXT,
            alpha=_CONTEXT_SPAN_ALPHA,
            label_offset_points=_CONTEXT_LABEL_OFFSET_POINTS,
        )
    for feature in context.features_for_track(_TRACK_CORE_INTERVALS):
        _add_context_span(
            ax,
            feature,
            position_to_index=position_to_index,
            row_count=row_count,
            fill_color=_CORE_INTERVAL_FILL,
            text_color=_CORE_INTERVAL_TEXT,
            alpha=_CORE_INTERVAL_SPAN_ALPHA,
            label_offset_points=_CORE_INTERVAL_LABEL_OFFSET_POINTS,
        )
    for feature in context.features_for_track(_TRACK_MOTIF_ANCHORS):
        _add_context_span(
            ax,
            feature,
            position_to_index=position_to_index,
            row_count=row_count,
            fill_color=_MOTIF_FILL,
            text_color=_MOTIF_TEXT,
            alpha=_MOTIF_SPAN_ALPHA,
            label_offset_points=_MOTIF_LABEL_OFFSET_POINTS,
        )


def _add_context_span(
    ax: Any,
    feature: RTAnnotationFeature,
    *,
    position_to_index: dict[int, int],
    row_count: int,
    fill_color: str,
    text_color: str,
    alpha: float,
    label_offset_points: float,
) -> None:
    bounds = _feature_bounds(feature, position_to_index)
    if bounds is None:
        return
    x, width = bounds
    patch = FancyBboxPatch(
        (x, -0.5),
        width,
        row_count,
        boxstyle="round,pad=0,rounding_size=0.15",
        facecolor=fill_color,
        edgecolor="none",
        linewidth=0,
        alpha=alpha,
        clip_on=False,
        zorder=MASK_ANNOTATION_SPAN_ZORDER,
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2.0,
        1.0,
        feature.label,
        ha="center",
        va="bottom",
        fontsize=_RT_SPAN_LABEL_SIZE,
        color=text_color,
        transform=_top_axis_offset_transform(ax, label_offset_points),
        clip_on=False,
        zorder=6,
    )


def _top_axis_offset_transform(ax: Any, offset_points: float) -> Any:
    return ax.get_xaxis_transform() + ScaledTranslation(
        0,
        offset_points / 72.0,
        ax.figure.dpi_scale_trans,
    )


def _feature_bounds(feature: RTAnnotationFeature, position_to_index: dict[int, int]) -> tuple[float, float] | None:
    indexes = [index for position, index in position_to_index.items() if feature.start <= position <= feature.end]
    if not indexes:
        return None
    start_index = min(indexes)
    end_index = max(indexes)
    return start_index - 0.5, float(end_index - start_index + 1)
