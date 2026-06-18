"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cruncher/src/snapback/released_plot_precursor.py

Precursor-panel renderer for released-product snapback hit plots.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any, Literal

from dnadesign.cruncher.snapback.released_plot_common import (
    _NICK,
    _NICK_SITE_FILL,
    _RELEASE,
    _RELEASE_SITE_FILL,
    _ROW_BOTTOM_Y,
    _ROW_TOP_Y,
    _TAIL,
    LABEL_BOTTOM_CUT,
    LABEL_NICK,
    LABEL_TOP_CUT,
    PANEL_TITLE_PRECURSOR_SITES,
    boundary_label_y,
    configure_axis,
    draw_sequence,
    draw_site_footprint,
    draw_strand_boundary,
    strand_row_label,
    x_for_base,
)
from dnadesign.cruncher.snapback.released_plot_models import PlotPrecursorPanelContext


def _event_boundary(payload: dict[str, Any], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int):
        raise ValueError(f"Released plot context is missing integer precursor event boundary '{key}'.")
    return value


def _site_label_placement_for_event(
    *,
    strand: str,
    boundary: int,
    site_start: int,
    site_end: int,
) -> Literal["above", "below"]:
    if strand == "top" and site_start <= boundary <= site_end:
        return "below"
    return "above"


def _site_orientation(site_payload: dict[str, Any], *, site_label: str) -> Literal["forward", "reverse"]:
    orientation = site_payload.get("orientation")
    if orientation not in {"forward", "reverse"}:
        raise ValueError(
            f"Released plot context for {site_label} is missing canonical site orientation ('forward' or 'reverse')."
        )
    return orientation


def _canonical_site_strand(site_payload: dict[str, Any], *, site_label: str) -> Literal["top", "bottom"]:
    orientation = _site_orientation(site_payload, site_label=site_label)
    return "top" if orientation == "forward" else "bottom"


def _site_emphasis_segments(
    context: PlotPrecursorPanelContext,
    *,
    strand: Literal["top", "bottom"],
) -> list[tuple[int, int]]:
    segments: list[tuple[int, int]] = []
    if _canonical_site_strand(context.nick_site, site_label="nick site") == strand:
        segments.append((context.nick_site_span.start, context.nick_site_span.end))
    if _canonical_site_strand(context.release_site, site_label="release site") == strand:
        segments.append((context.release_site_span.start, context.release_site_span.end))
    return segments


def _subtract_span(span: tuple[int, int], excluded: tuple[int, int]) -> list[tuple[int, int]]:
    start, end = span
    excluded_start, excluded_end = excluded
    if end <= start:
        return []
    if excluded_end <= start or excluded_start >= end:
        return [(start, end)]
    segments: list[tuple[int, int]] = []
    if excluded_start > start:
        segments.append((start, min(excluded_start, end)))
    if excluded_end < end:
        segments.append((max(excluded_end, start), end))
    return [(segment_start, segment_end) for segment_start, segment_end in segments if segment_end > segment_start]


def _inert_color_segments(
    context: PlotPrecursorPanelContext, *, strand: Literal["top", "bottom"]
) -> list[tuple[int, int, str]]:
    sacrificial_span = context.sacrificial_top_tail_span if strand == "top" else context.sacrificial_bottom_tail_span
    segments = [(sacrificial_span.start, sacrificial_span.end)]
    for protected_span in (context.nick_site_span, context.release_site_span):
        next_segments: list[tuple[int, int]] = []
        for segment in segments:
            next_segments.extend(_subtract_span(segment, (protected_span.start, protected_span.end)))
        segments = next_segments
    return [(start, end, _TAIL) for start, end in segments]


def render_precursor_panel(
    ax,
    *,
    context: PlotPrecursorPanelContext,
    nickase_variant_id: str,
    release_variant_id: str,
) -> None:
    configure_axis(
        ax,
        x_min=min(context.top_span.start, context.bottom_span.start),
        x_max=max(context.top_span.end, context.bottom_span.end),
        title=PANEL_TITLE_PRECURSOR_SITES,
    )
    draw_sequence(
        ax,
        sequence=context.top_sequence,
        y=_ROW_TOP_Y,
        row_label=strand_row_label("top"),
        start_terminal="5'",
        end_terminal="3'",
        x_start=context.top_span.start,
        color_segments=_inert_color_segments(context, strand="top"),
        assignable_base_positions=context.top_assignable_base_positions,
        emphasis_segments=_site_emphasis_segments(context, strand="top"),
    )
    draw_sequence(
        ax,
        sequence=context.bottom_sequence,
        y=_ROW_BOTTOM_Y,
        row_label=strand_row_label("bottom"),
        start_terminal="3'",
        end_terminal="5'",
        x_start=context.bottom_span.start,
        color_segments=_inert_color_segments(context, strand="bottom"),
        assignable_base_positions=context.bottom_assignable_base_positions,
        emphasis_segments=_site_emphasis_segments(context, strand="bottom"),
    )
    for index in range(len(context.top_sequence)):
        x = x_for_base(index, x_start=context.top_span.start)
        ax.plot([x, x], [_ROW_BOTTOM_Y + 0.06, _ROW_TOP_Y - 0.06], color="#E2E8F0", linewidth=0.8)
    draw_site_footprint(
        ax,
        start=context.nick_site_span.start,
        end=context.nick_site_span.end,
        label=nickase_variant_id,
        fill_color=_NICK_SITE_FILL,
        text_color=_NICK,
        label_placement=_site_label_placement_for_event(
            strand=context.nicked_strand,
            boundary=context.nick_boundary,
            site_start=context.nick_site_span.start,
            site_end=context.nick_site_span.end,
        ),
    )
    draw_site_footprint(
        ax,
        start=context.release_site_span.start,
        end=context.release_site_span.end,
        label=release_variant_id,
        fill_color=_RELEASE_SITE_FILL,
        text_color=_RELEASE,
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
    draw_strand_boundary(
        ax,
        boundary=_event_boundary(context.release_event, "top_cut_boundary"),
        strand="top",
        label=LABEL_TOP_CUT,
        color=_RELEASE,
        label_y=boundary_label_y("top", label_above=True),
    )
    draw_strand_boundary(
        ax,
        boundary=_event_boundary(context.release_event, "bottom_cut_boundary"),
        strand="bottom",
        label=LABEL_BOTTOM_CUT,
        color=_RELEASE,
        dashed=True,
        label_y=boundary_label_y("bottom", label_above=False),
        label_above=False,
    )


__all__ = ["render_precursor_panel"]
