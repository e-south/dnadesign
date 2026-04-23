"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/snapback/released_plot_foldback.py

Foldback-panel renderer for released-product snapback hit plots.

Module Author(s): Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from math import cos, radians, sin

from dnadesign.cruncher.snapback.released_plot_common import (
    _CAP,
    _FOLDBACK,
    _NICK,
    _ROW_BOTTOM_Y,
    _ROW_TOP_Y,
    _SEQUENCE_SIZE,
    _STEM,
    boundary_label_y,
    configure_axis,
    draw_sequence,
    draw_sequence_pairing,
    draw_strand_boundary,
    x_for_boundary,
)
from dnadesign.cruncher.snapback.released_plot_models import PlotFoldbackPanelContext


def render_foldback_panel(ax, *, context: PlotFoldbackPanelContext) -> None:
    from matplotlib.patches import Arc

    top_sequence = context.top_row.sequence
    bottom_sequence = context.bottom_row.sequence
    foldback_mismatches = set(context.foldback_mismatch_positions)
    cap_sequence = context.cap_sequence
    sequence_end = max(context.top_row.span.end, context.bottom_row.span.end)
    arc_width = max(0.96, 0.34 * max(len(cap_sequence), 1))
    arc_center_x = x_for_boundary(sequence_end) + (arc_width / 2.0) + 0.12
    panel_x_max = max(sequence_end, sequence_end + 1.2)
    configure_axis(
        ax,
        x_min=min(context.top_row.span.start, context.bottom_row.span.start),
        x_max=panel_x_max,
        title="Foldback",
    )
    draw_sequence(
        ax,
        sequence=top_sequence,
        y=_ROW_TOP_Y,
        row_label=context.top_row.label,
        start_terminal=context.top_row.left_terminal,
        end_terminal=None,
        x_start=context.top_row.span.start,
        color_segments=[(0, context.top_row.span.end, _STEM if context.top_row.role == "active_stem" else _FOLDBACK)],
    )
    draw_sequence(
        ax,
        sequence=bottom_sequence,
        y=_ROW_BOTTOM_Y,
        row_label=context.bottom_row.label,
        start_terminal=context.bottom_row.left_terminal,
        end_terminal=None,
        x_start=context.bottom_row.span.start,
        color_segments=[
            (0, context.bottom_row.span.end, _STEM if context.bottom_row.role == "active_stem" else _FOLDBACK)
        ],
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
    if cap_sequence:
        theta_values = [78.0]
        if len(cap_sequence) > 1:
            theta_values = [78.0 - ((156.0 / (len(cap_sequence) - 1)) * index) for index in range(len(cap_sequence))]
        radius_x = (arc_width / 2.0) * 0.98
        radius_y = (arc_height / 2.0) * 0.94
        for base, theta_deg in zip(cap_sequence, theta_values, strict=True):
            theta = radians(theta_deg)
            x = arc_center_x + (radius_x * cos(theta))
            y = arc_center_y + (radius_y * sin(theta))
            ax.text(
                x,
                y,
                base,
                ha="center",
                va="center",
                fontsize=_SEQUENCE_SIZE - 0.8,
                family="DejaVu Sans Mono",
                color=_CAP,
            )
    draw_strand_boundary(
        ax,
        boundary=0,
        strand=context.nicked_strand,
        label="Nick",
        color=_NICK,
        label_y=boundary_label_y(context.nicked_strand, label_above=context.nicked_strand == "top"),
        label_above=context.nicked_strand == "top",
    )


__all__ = ["render_foldback_panel"]
