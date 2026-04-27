"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/snapback/released_plot_released.py

Released-fragment panel renderer for released-product snapback hit plots.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dnadesign.cruncher.snapback.released_plot_common import (
    _CAP,
    _FOLDBACK,
    _NICK,
    _ROW_BOTTOM_Y,
    _ROW_TOP_Y,
    _STEM,
    LABEL_CAP,
    LABEL_FOLDBACK,
    LABEL_NICK,
    LABEL_STEM,
    PANEL_TITLE_POST_RELEASE_FRAGMENTS,
    boundary_label_interval_nt,
    boundary_label_y,
    configure_axis,
    draw_region_label,
    draw_sequence,
    draw_sequence_pairing,
    draw_strand_boundary,
    staggered_label_y,
)
from dnadesign.cruncher.snapback.released_plot_models import PlotReleasedProductContext

_STRUCTURE_LABEL_BASE_Y = 0.90
_STRUCTURE_LABEL_STAGGER_Y = 0.98


def _structure_label_y(index: int, labeled_spans, *, context: PlotReleasedProductContext) -> float:
    obstacles = []
    if context.nicked_strand == "top":
        obstacles.append(boundary_label_interval_nt(boundary=context.nick_boundary, label=LABEL_NICK))
    return staggered_label_y(
        index=index,
        labeled_spans=labeled_spans,
        base_y=_STRUCTURE_LABEL_BASE_Y,
        stagger_y=_STRUCTURE_LABEL_STAGGER_Y,
        obstacles=obstacles,
    )


def render_released_panel(ax, *, context: PlotReleasedProductContext) -> None:
    configure_axis(
        ax,
        x_min=min(context.top_row.span.start, context.bottom_row.span.start),
        x_max=max(context.top_row.span.end, context.bottom_row.span.end),
        title=PANEL_TITLE_POST_RELEASE_FRAGMENTS,
    )
    draw_sequence(
        ax,
        sequence=context.top_row.sequence,
        y=_ROW_TOP_Y,
        row_label=context.top_row.label,
        start_terminal=context.top_row.start_terminal,
        end_terminal=context.top_row.end_terminal,
        x_start=context.top_row.span.start,
        color_segments=(
            [
                (context.stem_span.start, context.stem_span.end, _STEM),
                (context.cap_span.start, context.cap_span.end, _CAP),
                (context.foldback_span.start, context.foldback_span.end, _FOLDBACK),
            ]
            if context.top_row.role == "active_product"
            else None
        ),
        assignable_base_positions=context.top_row.assignable_base_positions,
    )
    draw_sequence(
        ax,
        sequence=context.bottom_row.sequence,
        y=_ROW_BOTTOM_Y,
        row_label=context.bottom_row.label,
        start_terminal=context.bottom_row.start_terminal,
        end_terminal=context.bottom_row.end_terminal,
        x_start=context.bottom_row.span.start,
        color_segments=(
            [
                (context.stem_span.start, context.stem_span.end, _STEM),
                (context.cap_span.start, context.cap_span.end, _CAP),
                (context.foldback_span.start, context.foldback_span.end, _FOLDBACK),
            ]
            if context.bottom_row.role == "active_product"
            else None
        ),
        assignable_base_positions=context.bottom_row.assignable_base_positions,
    )
    if context.duplex_overlap_span is not None:
        draw_sequence_pairing(
            ax,
            start=context.duplex_overlap_span.start,
            end=context.duplex_overlap_span.end,
            mismatch_positions=set(context.duplex_mismatch_positions),
            linewidth=0.8,
        )
    draw_strand_boundary(
        ax,
        boundary=context.nick_boundary,
        strand=context.nicked_strand,
        label=LABEL_NICK,
        color=_NICK,
        label_y=boundary_label_y(context.nicked_strand, label_above=context.nicked_strand == "top"),
        label_above=context.nicked_strand == "top",
    )
    structure_spans = [
        (context.stem_span.start, context.stem_span.end, LABEL_STEM),
        (context.cap_span.start, context.cap_span.end, LABEL_CAP),
        (context.foldback_span.start, context.foldback_span.end, LABEL_FOLDBACK),
    ]
    draw_region_label(
        ax,
        start=context.stem_span.start,
        end=context.stem_span.end,
        y=_structure_label_y(0, structure_spans, context=context),
        label=LABEL_STEM,
        color=_STEM,
    )
    draw_region_label(
        ax,
        start=context.cap_span.start,
        end=context.cap_span.end,
        y=_structure_label_y(1, structure_spans, context=context),
        label=LABEL_CAP,
        color=_CAP,
    )
    draw_region_label(
        ax,
        start=context.foldback_span.start,
        end=context.foldback_span.end,
        y=_structure_label_y(2, structure_spans, context=context),
        label=LABEL_FOLDBACK,
        color=_FOLDBACK,
    )


__all__ = ["render_released_panel"]
