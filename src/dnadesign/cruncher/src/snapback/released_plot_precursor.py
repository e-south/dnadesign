"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/snapback/released_plot_precursor.py

Precursor-panel renderer for released-product snapback hit plots.

Module Author(s): Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any

from dnadesign.cruncher.snapback.released_plot_common import (
    _NICK,
    _NICK_SITE_FILL,
    _RELEASE,
    _RELEASE_SITE_FILL,
    _ROW_BOTTOM_Y,
    _ROW_TOP_Y,
    boundary_label_y,
    configure_axis,
    draw_sequence,
    draw_site_footprint,
    draw_strand_boundary,
    x_for_base,
)
from dnadesign.cruncher.snapback.released_plot_models import PlotPrecursorPanelContext


def _event_boundary(payload: dict[str, Any], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int):
        raise ValueError(f"Released plot context is missing integer precursor event boundary '{key}'.")
    return value


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
        title="Precursor Sites",
    )
    draw_sequence(
        ax,
        sequence=context.top_sequence,
        y=_ROW_TOP_Y,
        row_label="Top",
        start_terminal="5'",
        end_terminal="3'",
        x_start=context.top_span.start,
    )
    draw_sequence(
        ax,
        sequence=context.bottom_sequence,
        y=_ROW_BOTTOM_Y,
        row_label="Bottom",
        start_terminal="3'",
        end_terminal="5'",
        x_start=context.bottom_span.start,
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
        label="Nick",
        color=_NICK,
        label_y=boundary_label_y(context.nicked_strand, label_above=context.nicked_strand == "top"),
        label_above=context.nicked_strand == "top",
    )
    draw_strand_boundary(
        ax,
        boundary=_event_boundary(context.release_event, "top_cut_boundary"),
        strand="top",
        label="Top Cut",
        color=_RELEASE,
        label_y=boundary_label_y("top", label_above=True),
    )
    draw_strand_boundary(
        ax,
        boundary=_event_boundary(context.release_event, "bottom_cut_boundary"),
        strand="bottom",
        label="Bottom Cut",
        color=_RELEASE,
        dashed=True,
        label_y=boundary_label_y("bottom", label_above=False),
        label_above=False,
    )


__all__ = ["render_precursor_panel"]
