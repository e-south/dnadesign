"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cruncher/src/snapback/released_plot_foldback.py

Foldback-panel renderer for released-product snapback hit plots.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from math import cos, radians, sin
from typing import Literal

from dnadesign.cruncher.snapback.released_plot_common import (
    _CAP,
    _FOLDBACK,
    _NICK,
    _ROW_BOTTOM_Y,
    _ROW_TOP_Y,
    _SEQUENCE_SIZE,
    _STEM,
    LABEL_NICK,
    PANEL_TITLE_FOLDBACK,
    boundary_label_y,
    configure_axis,
    draw_assignable_base_zstack,
    draw_sequence,
    draw_sequence_pairing,
    draw_strand_boundary,
    x_for_boundary,
)
from dnadesign.cruncher.snapback.released_plot_models import PlotFoldbackPanelContext

_ELLIPSE_BEZIER_KAPPA = 0.5522847498307936
_ACTION_ARROW_MIN_GAP = 0.88
_ACTION_ARROW_MIN_WIDTH = 0.42
_ACTION_ARROW_WIDTH_SCALE = 0.44
_ACTION_ARROW_TERMINAL_TANGENT = 0.055


def _foldback_color_segments(
    *,
    role: str,
    span_start: int,
    span_end: int,
    origin_boundary: int,
) -> list[tuple[int, int, str]]:
    if role == "active_stem":
        stem_start = max(span_start, 0)
        return [(stem_start, span_end, _STEM)] if span_end > stem_start else []
    segments: list[tuple[int, int, str]] = []
    upstream_start = max(span_start, 0)
    upstream_end = min(max(origin_boundary, upstream_start), span_end)
    if upstream_end > upstream_start:
        segments.append((upstream_start, upstream_end, _STEM))
    foldback_start = min(max(origin_boundary, 0, span_start), span_end)
    if span_end > foldback_start:
        segments.append((foldback_start, span_end, _FOLDBACK))
    return segments


def _right_half_ellipse_path(*, center_x: float, center_y: float, width: float, height: float):
    from matplotlib.path import Path

    radius_x = width / 2.0
    radius_y = height / 2.0
    kappa = _ELLIPSE_BEZIER_KAPPA
    vertices = [
        (center_x, center_y + radius_y),
        (center_x + (kappa * radius_x), center_y + radius_y),
        (center_x + radius_x, center_y + (kappa * radius_y)),
        (center_x + radius_x, center_y),
        (center_x + radius_x, center_y - (kappa * radius_y)),
        (center_x + (kappa * radius_x), center_y - radius_y),
        (center_x, center_y - radius_y),
    ]
    codes = [
        Path.MOVETO,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
    ]
    return Path(vertices, codes)


def _foldback_action_direction(context: PlotFoldbackPanelContext) -> Literal["up", "down"]:
    if context.top_row.role == "active_stem" and context.bottom_row.role == "foldback_return":
        return "down"
    if context.bottom_row.role == "active_stem" and context.top_row.role == "foldback_return":
        return "up"
    raise ValueError("Foldback action arrow requires one active_stem row and one foldback_return row.")


def _foldback_action_arrow_path(
    *,
    cap_arc_center_x: float,
    cap_arc_center_y: float,
    arc_width: float,
    arc_height: float,
    direction: Literal["up", "down"],
):
    from matplotlib.path import Path

    top_y = cap_arc_center_y + (arc_height / 2.0)
    bottom_y = cap_arc_center_y - (arc_height / 2.0)
    start_x = cap_arc_center_x + max(_ACTION_ARROW_MIN_GAP, arc_width * 0.72)
    end_x = start_x + max(_ACTION_ARROW_MIN_WIDTH, arc_width * _ACTION_ARROW_WIDTH_SCALE)
    if direction == "down":
        vertices = [
            (start_x, top_y),
            (end_x, top_y),
            (end_x, bottom_y + _ACTION_ARROW_TERMINAL_TANGENT),
            (end_x, bottom_y),
        ]
    else:
        vertices = [
            (start_x, bottom_y),
            (end_x, bottom_y),
            (end_x, top_y - _ACTION_ARROW_TERMINAL_TANGENT),
            (end_x, top_y),
        ]
    return Path(vertices, [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4])


def render_foldback_panel(ax, *, context: PlotFoldbackPanelContext) -> None:
    from matplotlib.patches import Arc, FancyArrowPatch

    top_sequence = context.top_row.sequence
    bottom_sequence = context.bottom_row.sequence
    foldback_mismatches = set(context.foldback_mismatch_positions)
    cap_sequence = context.cap_sequence
    sequence_end = max(context.top_row.span.end, context.bottom_row.span.end)
    arc_width = max(0.96, 0.34 * max(len(cap_sequence), 1))
    arc_center_x = x_for_boundary(sequence_end) + (arc_width / 2.0) + 0.04
    action_direction = _foldback_action_direction(context)
    action_arrow_path = _foldback_action_arrow_path(
        cap_arc_center_x=arc_center_x,
        cap_arc_center_y=(_ROW_TOP_Y + _ROW_BOTTOM_Y) / 2.0,
        arc_width=arc_width,
        arc_height=(_ROW_TOP_Y - _ROW_BOTTOM_Y) + 0.02,
        direction=action_direction,
    )
    action_arrow_right_nt = max(vertex[0] for vertex in action_arrow_path.vertices) / x_for_boundary(1)
    panel_x_max = max(sequence_end, action_arrow_right_nt + 1.1)
    configure_axis(
        ax,
        x_min=min(context.top_row.span.start, context.bottom_row.span.start),
        x_max=panel_x_max,
        title=PANEL_TITLE_FOLDBACK,
    )
    draw_sequence(
        ax,
        sequence=top_sequence,
        y=_ROW_TOP_Y,
        row_label=context.top_row.label,
        start_terminal=context.top_row.left_terminal,
        end_terminal=None,
        x_start=context.top_row.span.start,
        color_segments=_foldback_color_segments(
            role=context.top_row.role,
            span_start=context.top_row.span.start,
            span_end=context.top_row.span.end,
            origin_boundary=context.origin_boundary_from_left,
        ),
        assignable_base_positions=context.top_row.assignable_base_positions,
    )
    draw_sequence(
        ax,
        sequence=bottom_sequence,
        y=_ROW_BOTTOM_Y,
        row_label=context.bottom_row.label,
        start_terminal=context.bottom_row.left_terminal,
        end_terminal=None,
        x_start=context.bottom_row.span.start,
        color_segments=_foldback_color_segments(
            role=context.bottom_row.role,
            span_start=context.bottom_row.span.start,
            span_end=context.bottom_row.span.end,
            origin_boundary=context.origin_boundary_from_left,
        ),
        assignable_base_positions=context.bottom_row.assignable_base_positions,
    )
    draw_sequence_pairing(
        ax,
        start=max(context.top_row.span.start, context.bottom_row.span.start),
        end=min(context.top_row.span.end, context.bottom_row.span.end),
        mismatch_positions=foldback_mismatches,
        linewidth=0.9,
    )

    arc_center_y = (_ROW_TOP_Y + _ROW_BOTTOM_Y) / 2.0
    arc_height = (_ROW_TOP_Y - _ROW_BOTTOM_Y) + 0.02
    ax.add_patch(
        Arc(
            (arc_center_x, arc_center_y),
            width=arc_width,
            height=arc_height,
            theta1=-90,
            theta2=90,
            color="#CBD5E1",
            linewidth=1.2,
        )
    )
    ax.add_patch(
        FancyArrowPatch(
            path=action_arrow_path,
            arrowstyle="-|>",
            mutation_scale=14.5,
            facecolor="#94A3B8",
            edgecolor="#94A3B8",
            linewidth=1.15,
            shrinkA=0.0,
            shrinkB=0.0,
            zorder=2.2,
        )
    )
    if cap_sequence:
        theta_values = [78.0]
        if len(cap_sequence) > 1:
            theta_values = [78.0 - ((156.0 / (len(cap_sequence) - 1)) * index) for index in range(len(cap_sequence))]
        radius_x = (arc_width / 2.0) * 0.98
        radius_y = (arc_height / 2.0) * 0.94
        assignable_cap_positions = set(context.assignable_cap_base_positions)
        for index, (base, theta_deg) in enumerate(zip(cap_sequence, theta_values, strict=True)):
            theta = radians(theta_deg)
            x = arc_center_x + (radius_x * cos(theta))
            y = arc_center_y + (radius_y * sin(theta))
            if index in assignable_cap_positions:
                draw_assignable_base_zstack(ax, x=x, y=y)
            ax.text(
                x,
                y,
                base,
                ha="center",
                va="center",
                fontsize=_SEQUENCE_SIZE - 0.8,
                family="DejaVu Sans Mono",
                color=_CAP,
                zorder=3.0,
            )
    draw_strand_boundary(
        ax,
        boundary=context.origin_boundary_from_left,
        strand=context.nicked_strand,
        label=LABEL_NICK,
        color=_NICK,
        label_y=boundary_label_y(context.nicked_strand, label_above=context.nicked_strand == "top"),
        label_above=context.nicked_strand == "top",
    )


__all__ = ["render_foldback_panel"]
