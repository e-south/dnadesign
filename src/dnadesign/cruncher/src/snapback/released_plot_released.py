"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/snapback/released_plot_released.py

Released-fragment panel renderer for released-product snapback hit plots.

Module Author(s): Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dnadesign.cruncher.snapback.released_plot_common import (
    _CAP,
    _FOLDBACK,
    _LOWER_PRIMARY_SPAN_Y,
    _NICK,
    _OVERHANG,
    _RELEASE,
    _ROW_BOTTOM_Y,
    _ROW_TOP_Y,
    _STEM,
    _STRUCTURE_LABEL_Y,
    boundary_label_y,
    configure_axis,
    draw_region_label,
    draw_sequence,
    draw_sequence_pairing,
    draw_span,
    draw_strand_boundary,
    span_contains_boundary,
)
from dnadesign.cruncher.snapback.released_plot_models import PlotReleasedProductContext


def render_released_panel(ax, *, context: PlotReleasedProductContext) -> None:
    configure_axis(
        ax,
        x_min=min(context.top_row.span.start, context.bottom_row.span.start),
        x_max=max(context.top_row.span.end, context.bottom_row.span.end),
        title="Post-Release Fragments",
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
        label="Nick",
        color=_NICK,
        label_y=boundary_label_y(context.nicked_strand, label_above=context.nicked_strand == "top"),
        label_above=context.nicked_strand == "top",
    )
    if span_contains_boundary(context.top_row.span, context.release_top_cut_boundary):
        draw_strand_boundary(
            ax,
            boundary=context.release_top_cut_boundary,
            strand="top",
            label="Top Cut",
            color=_RELEASE,
            dashed=True,
            label_y=boundary_label_y("top", label_above=True),
        )
    if span_contains_boundary(context.bottom_row.span, context.release_bottom_cut_boundary):
        draw_strand_boundary(
            ax,
            boundary=context.release_bottom_cut_boundary,
            strand="bottom",
            label="Bottom Cut",
            color=_RELEASE,
            dashed=True,
            label_y=boundary_label_y("bottom", label_above=False),
            label_above=False,
        )
    draw_region_label(
        ax,
        start=context.stem_span.start,
        end=context.stem_span.end,
        y=_STRUCTURE_LABEL_Y,
        label="Stem",
        color=_STEM,
    )
    draw_region_label(
        ax,
        start=context.cap_span.start,
        end=context.cap_span.end,
        y=_STRUCTURE_LABEL_Y,
        label="Cap",
        color=_CAP,
    )
    draw_region_label(
        ax,
        start=context.foldback_span.start,
        end=context.foldback_span.end,
        y=_STRUCTURE_LABEL_Y,
        label="Foldback",
        color=_FOLDBACK,
    )
    if context.top_only_overhang_span is not None:
        draw_span(
            ax,
            start=context.top_only_overhang_span.start,
            end=context.top_only_overhang_span.end,
            y=_LOWER_PRIMARY_SPAN_Y,
            label="Overhang",
            color=_OVERHANG,
        )
    if context.bottom_only_overhang_span is not None:
        draw_span(
            ax,
            start=context.bottom_only_overhang_span.start,
            end=context.bottom_only_overhang_span.end,
            y=_LOWER_PRIMARY_SPAN_Y,
            label="Overhang",
            color=_OVERHANG,
        )


__all__ = ["render_released_panel"]
