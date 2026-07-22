"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cruncher/src/snapback/visual_plot.py

Plotting for visual-only single-nick snapback examples.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from dnadesign.cruncher.nickases.models import motif_matches, reverse_complement_iupac
from dnadesign.cruncher.snapback import released_plot_common as plot_common
from dnadesign.cruncher.snapback.publication_support import complement_sequence
from dnadesign.cruncher.snapback.released_plot_foldback import render_foldback_panel
from dnadesign.cruncher.snapback.released_plot_models import (
    PlotFoldbackPanelContext,
    PlotFoldbackRow,
    PlotSpan,
)
from dnadesign.cruncher.snapback.visual_models import SingleNickSnapbackVisualSpec


def _span(start: int, end: int) -> dict[str, int]:
    return {"start": start, "end": end}


_POST_NICK_LABEL_BASE_Y = 0.50
_POST_NICK_LABEL_STAGGER_Y = 0.41
_FIXED_IUPAC_BASES = frozenset({"A", "C", "G", "T"})


def _structure_label_y(index: int, labeled_spans: list[tuple[float, float, str]]) -> float:
    return plot_common.staggered_label_y(
        index=index,
        labeled_spans=labeled_spans,
        base_y=_POST_NICK_LABEL_BASE_Y,
        stagger_y=_POST_NICK_LABEL_STAGGER_Y,
    )


def _complement_iupac_base(symbol: str) -> str:
    return reverse_complement_iupac(symbol)[::-1]


def _nick_site_orientation(spec: SingleNickSnapbackVisualSpec) -> Literal["forward", "reverse"]:
    observed_site = spec.input.precursor_top_strand[spec.nick.site_span.start : spec.nick.site_span.end]
    if motif_matches(observed_site, spec.nick.site_sequence):
        return "forward"
    reverse_site = reverse_complement_iupac(spec.nick.site_sequence)
    if motif_matches(observed_site, reverse_site):
        return "reverse"
    raise ValueError("Visual nick site no longer matches the precursor sequence.")


def _degenerate_site_positions(spec: SingleNickSnapbackVisualSpec) -> list[int]:
    site_width = spec.nick.site_span.end - spec.nick.site_span.start
    orientation = _nick_site_orientation(spec)
    return [
        spec.nick.site_span.start + (canonical_index if orientation == "forward" else site_width - 1 - canonical_index)
        for canonical_index, symbol in enumerate(spec.nick.site_sequence)
        if symbol.upper() not in _FIXED_IUPAC_BASES
    ]


def _precursor_display_sequences(
    spec: SingleNickSnapbackVisualSpec,
) -> tuple[str, str, list[int], list[int]]:
    top_bases = list(spec.input.precursor_top_strand)
    bottom_bases = list(complement_sequence(spec.input.precursor_top_strand))
    assignable_positions = _degenerate_site_positions(spec)
    for position in assignable_positions:
        local_offset = position - spec.nick.site_span.start
        symbol = spec.nick.site_sequence[local_offset].upper()
        top_bases[position] = symbol
        bottom_bases[position] = _complement_iupac_base(symbol)
    return (
        "".join(top_bases),
        "".join(bottom_bases),
        assignable_positions,
        assignable_positions,
    )


def _positions_in_span(positions: list[int], *, start: int, end: int) -> list[int]:
    return sorted(position for position in positions if start <= position < end)


def _active_foldback_assignable_positions(
    positions: list[int],
    *,
    upstream_end: int,
    stem_end: int,
) -> list[int]:
    return _positions_in_span(positions, start=0, end=stem_end)


def _return_foldback_assignable_positions(
    positions: list[int],
    *,
    upstream_end: int,
    cap_end: int,
    foldback_end: int,
) -> list[int]:
    return sorted(
        upstream_end + (foldback_end - 1 - position) for position in positions if cap_end <= position < foldback_end
    )


def _cap_assignable_positions(
    positions: list[int],
    *,
    stem_end: int,
    cap_end: int,
) -> list[int]:
    return sorted(position - stem_end for position in positions if stem_end <= position < cap_end)


def _site_emphasis_segments(
    context: dict[str, Any],
    *,
    strand: Literal["top", "bottom"],
) -> list[tuple[int, int]]:
    if context["nick_site_orientation"] == ("forward" if strand == "top" else "reverse"):
        site_span = context["nick_site_span"]
        return [(site_span["start"], site_span["end"])]
    return []


def _build_foldback_panel(spec: SingleNickSnapbackVisualSpec) -> PlotFoldbackPanelContext:
    stem = spec.product.stem_sequence
    foldback = spec.product.foldback_sequence
    upstream = spec.product.upstream_context_nt
    effective_stem_bp = upstream + len(stem)
    stem_end = upstream + len(stem)
    cap_end = stem_end + len(spec.product.cap_sequence)
    foldback_end = cap_end + len(foldback)
    degenerate_positions = _degenerate_site_positions(spec)
    precursor_top_display, precursor_bottom_display, _, _ = _precursor_display_sequences(spec)
    upstream_assignable_positions = _positions_in_span(degenerate_positions, start=0, end=upstream)
    return PlotFoldbackPanelContext(
        origin_boundary_from_left=upstream,
        stem_sequence=stem,
        cap_sequence=spec.product.cap_sequence,
        foldback_sequence=foldback,
        foldback_partner_sequence=foldback[::-1],
        upstream_context_span=PlotSpan(start=0, end=upstream),
        nicked_strand=spec.nick.nicked_strand,
        top_row=PlotFoldbackRow(
            role="foldback_return",
            label=plot_common.ROW_LABEL_FOLDBACK_STEM,
            sequence=f"{precursor_top_display[:upstream]}{foldback[::-1]}",
            span=PlotSpan(start=0, end=effective_stem_bp),
            left_terminal="5'",
            assignable_base_positions=sorted(
                set(upstream_assignable_positions)
                | set(
                    _return_foldback_assignable_positions(
                        degenerate_positions,
                        upstream_end=upstream,
                        cap_end=cap_end,
                        foldback_end=foldback_end,
                    )
                )
            ),
        ),
        bottom_row=PlotFoldbackRow(
            role="active_stem",
            label=plot_common.ROW_LABEL_STEM,
            sequence=f"{precursor_bottom_display[:upstream]}{stem}",
            span=PlotSpan(start=0, end=effective_stem_bp),
            left_terminal="3'",
            assignable_base_positions=sorted(
                set(upstream_assignable_positions)
                | set(
                    _active_foldback_assignable_positions(
                        degenerate_positions,
                        upstream_end=upstream,
                        stem_end=stem_end,
                    )
                )
            ),
        ),
        assignable_cap_base_positions=_cap_assignable_positions(
            degenerate_positions,
            stem_end=stem_end,
            cap_end=cap_end,
        ),
    )


def build_snapback_visual_plot_context(spec: SingleNickSnapbackVisualSpec) -> dict[str, Any]:
    top = spec.input.precursor_top_strand
    bottom = complement_sequence(top)
    (
        precursor_top_display,
        precursor_bottom_display,
        precursor_top_assignable_positions,
        precursor_bottom_assignable_positions,
    ) = _precursor_display_sequences(spec)
    degenerate_positions = _degenerate_site_positions(spec)
    upstream_end = spec.product.upstream_context_nt
    stem_end = upstream_end + len(spec.product.stem_sequence)
    cap_end = stem_end + len(spec.product.cap_sequence)
    foldback_end = cap_end + len(spec.product.foldback_sequence)
    foldback_panel = _build_foldback_panel(spec)
    nick_site_orientation = _nick_site_orientation(spec)
    return {
        "kind": "snapback_visual_plot_v1",
        "name": spec.name,
        "precursor": {
            "top_sequence": precursor_top_display,
            "bottom_sequence": precursor_bottom_display,
            "resolved_top_sequence": top,
            "resolved_bottom_sequence": bottom,
            "top_assignable_base_positions": precursor_top_assignable_positions,
            "bottom_assignable_base_positions": precursor_bottom_assignable_positions,
            "nick_label": spec.nick.label,
            "nick_boundary": spec.nick.nick_boundary,
            "nicked_strand": spec.nick.nicked_strand,
            "nick_site_sequence": spec.nick.site_sequence,
            "nick_site_orientation": nick_site_orientation,
            "nick_site_span": _span(spec.nick.site_span.start, spec.nick.site_span.end),
        },
        "released_product": {
            "released_strand_sequence": precursor_top_display[:upstream_end],
            "active_product_sequence": precursor_bottom_display[:upstream_end]
            + spec.active_product_sequence[upstream_end:],
            "released_strand_assignable_base_positions": _positions_in_span(
                degenerate_positions,
                start=0,
                end=upstream_end,
            ),
            "active_assignable_base_positions": _positions_in_span(
                degenerate_positions,
                start=0,
                end=len(spec.active_product_sequence),
            ),
            "released_strand_label": plot_common.post_release_physical_row_label(
                strand=spec.nick.nicked_strand,
                nicked_strand=spec.nick.nicked_strand,
            ),
            "active_label": plot_common.post_release_physical_row_label(
                strand=spec.product.active_strand,
                nicked_strand=spec.nick.nicked_strand,
            ),
            "nicked_strand": spec.nick.nicked_strand,
            "nick_boundary": spec.nick.nick_boundary,
            "duplex_overlap_span": _span(0, upstream_end),
            "single_strand_span": _span(upstream_end, foldback_end),
            "upstream_context_span": _span(0, upstream_end),
            "stem_span": _span(upstream_end, stem_end),
            "cap_span": _span(stem_end, cap_end),
            "foldback_span": _span(cap_end, foldback_end),
        },
        "foldback": foldback_panel.model_dump(mode="json"),
    }


def _render_precursor_panel(ax, *, context: dict[str, Any]) -> None:
    top = context["top_sequence"]
    bottom = context["bottom_sequence"]
    site_span = context["nick_site_span"]
    x_max = max(len(top), 12)
    plot_common.configure_axis(ax, x_min=0, x_max=x_max, title=plot_common.PANEL_TITLE_PRECURSOR_SITES)
    plot_common.draw_site_footprint(
        ax,
        start=site_span["start"],
        end=site_span["end"],
        label=context["nick_label"],
        fill_color=plot_common._NICK_SITE_FILL,
        text_color=plot_common._NICK,
        label_placement=(
            "below"
            if context["nicked_strand"] == "top" and site_span["start"] <= context["nick_boundary"] <= site_span["end"]
            else "above"
        ),
    )
    plot_common.draw_sequence_pairing(
        ax,
        start=0,
        end=len(top),
        mismatch_positions=set(),
        linewidth=0.55,
    )
    plot_common.draw_sequence(
        ax,
        sequence=top,
        y=plot_common._ROW_TOP_Y,
        row_label=plot_common.strand_row_label("top"),
        start_terminal="5'",
        end_terminal="3'",
        assignable_base_positions=context["top_assignable_base_positions"],
        emphasis_segments=_site_emphasis_segments(context, strand="top"),
    )
    plot_common.draw_sequence(
        ax,
        sequence=bottom,
        y=plot_common._ROW_BOTTOM_Y,
        row_label=plot_common.strand_row_label("bottom"),
        start_terminal="3'",
        end_terminal="5'",
        assignable_base_positions=context["bottom_assignable_base_positions"],
        emphasis_segments=_site_emphasis_segments(context, strand="bottom"),
    )
    plot_common.draw_strand_boundary(
        ax,
        boundary=context["nick_boundary"],
        strand=context["nicked_strand"],
        label=plot_common.LABEL_NICK,
        color=plot_common._NICK,
        label_y=plot_common.boundary_label_y(context["nicked_strand"], label_above=context["nicked_strand"] == "top"),
        label_above=context["nicked_strand"] == "top",
    )


def _render_post_nick_panel(ax, *, context: dict[str, Any]) -> None:
    top = context["released_strand_sequence"]
    active = context["active_product_sequence"]
    upstream = context["upstream_context_span"]
    stem = context["stem_span"]
    cap = context["cap_span"]
    foldback = context["foldback_span"]
    x_max = max(len(active), 12)
    plot_common.configure_axis(ax, x_min=0, x_max=x_max, title=plot_common.PANEL_TITLE_POST_RELEASE_FRAGMENTS)
    if upstream["end"] > upstream["start"]:
        plot_common.draw_sequence_pairing(
            ax,
            start=upstream["start"],
            end=upstream["end"],
            mismatch_positions=set(),
            linewidth=0.9,
        )
    plot_common.draw_sequence(
        ax,
        sequence=top,
        y=plot_common._ROW_TOP_Y,
        row_label=context["released_strand_label"],
        start_terminal="5'" if top else None,
        end_terminal="3'" if top else None,
        assignable_base_positions=context["released_strand_assignable_base_positions"],
    )
    plot_common.draw_sequence(
        ax,
        sequence=active,
        y=plot_common._ROW_BOTTOM_Y,
        row_label=context["active_label"],
        start_terminal="3'",
        end_terminal="5'",
        color_segments=[
            (stem["start"], stem["end"], plot_common._STEM),
            (cap["start"], cap["end"], plot_common._CAP),
            (foldback["start"], foldback["end"], plot_common._FOLDBACK),
        ],
        assignable_base_positions=context["active_assignable_base_positions"],
    )
    structure_spans = [
        (stem["start"], stem["end"], plot_common.LABEL_STEM),
        (cap["start"], cap["end"], plot_common.LABEL_CAP),
        (foldback["start"], foldback["end"], plot_common.LABEL_FOLDBACK),
    ]
    plot_common.draw_region_label(
        ax,
        start=stem["start"],
        end=stem["end"],
        y=_structure_label_y(0, structure_spans),
        label=plot_common.LABEL_STEM,
        color=plot_common._STEM,
    )
    plot_common.draw_region_label(
        ax,
        start=cap["start"],
        end=cap["end"],
        y=_structure_label_y(1, structure_spans),
        label=plot_common.LABEL_CAP,
        color=plot_common._CAP,
    )
    plot_common.draw_region_label(
        ax,
        start=foldback["start"],
        end=foldback["end"],
        y=_structure_label_y(2, structure_spans),
        label=plot_common.LABEL_FOLDBACK,
        color=plot_common._FOLDBACK,
    )
    plot_common.draw_strand_boundary(
        ax,
        boundary=context["nick_boundary"],
        strand=context["nicked_strand"],
        label=plot_common.LABEL_NICK,
        color=plot_common._NICK,
        label_y=plot_common.boundary_label_y(context["nicked_strand"], label_above=True),
    )


def render_snapback_visual_plot(spec: SingleNickSnapbackVisualSpec, output_path: Path) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    context = build_snapback_visual_plot_context(spec)
    foldback_context = PlotFoldbackPanelContext.model_validate(context["foldback"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    width_ratios = [
        max(len(context["precursor"]["top_sequence"]), 12),
        max(len(context["released_product"]["active_product_sequence"]), 12),
        max(len(foldback_context.top_row.sequence), len(foldback_context.bottom_row.sequence), 8),
    ]
    figure_width = max(15.0, sum(width_ratios) * 0.33)
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(figure_width, 4.4),
        dpi=170,
        gridspec_kw={"width_ratios": width_ratios},
    )
    fig.patch.set_facecolor(plot_common._FIGURE_FACE)
    _render_precursor_panel(axes[0], context=context["precursor"])
    _render_post_nick_panel(axes[1], context=context["released_product"])
    render_foldback_panel(axes[2], context=foldback_context)
    fig.tight_layout(pad=0.32, w_pad=0.34)
    fig.savefig(output_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return context


__all__ = ["build_snapback_visual_plot_context", "render_snapback_visual_plot"]
