"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/snapback/released_plot_context.py

Typed context builder for released-product snapback hit plots.

Module Author(s): Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any

from dnadesign.cruncher.snapback.publication_support import complement_sequence
from dnadesign.cruncher.snapback.released_plot_models import (
    PlotFoldbackPanelContext,
    PlotFoldbackRow,
    PlotFragmentRow,
    PlotLabels,
    PlotPrecursorPanelContext,
    PlotReleasedProductContext,
    PlotSpan,
    PlotTarget,
    ReleasedHitPlotContext,
)
from dnadesign.cruncher.snapback.released_search_models import ReleasedTargetSearchHit


def _span(start: int, end: int) -> PlotSpan:
    return PlotSpan(start=start, end=end)


def _display_site_payload(site) -> dict[str, Any]:
    payload = site.model_dump(mode="json")
    local_start = payload.get("local_start")
    local_end = payload.get("local_end")
    if not isinstance(local_start, int) or not isinstance(local_end, int):
        return payload
    if local_start >= 0:
        payload["omitted_left_of_origin_nt"] = 0
        return payload
    omitted = min(-local_start, max(0, local_end - local_start))
    payload["raw_local_start"] = local_start
    payload["raw_local_end"] = local_end
    payload["omitted_left_of_origin_nt"] = omitted
    payload["local_start"] = 0
    payload["local_end"] = max(0, local_end)
    return payload


def _fragment_row(
    *,
    role: str,
    strand: str,
    label: str,
    sequence: str,
    span: PlotSpan,
    start_terminal: str | None,
    end_terminal: str | None,
) -> PlotFragmentRow:
    return PlotFragmentRow(
        role=role,
        strand=strand,
        label=label,
        sequence=sequence,
        span=span,
        start_terminal=start_terminal,
        end_terminal=end_terminal,
    )


def _foldback_row(
    *,
    role: str,
    label: str,
    sequence: str,
    span: PlotSpan,
    left_terminal: str | None,
) -> PlotFoldbackRow:
    return PlotFoldbackRow(
        role=role,
        label=label,
        sequence=sequence,
        span=span,
        left_terminal=left_terminal,
    )


def _watson_crick_mismatch_positions(*, top_sequence: str, bottom_sequence: str) -> list[int]:
    mismatches: list[int] = []
    for index, (top_base, bottom_base) in enumerate(zip(top_sequence, bottom_sequence, strict=True)):
        if complement_sequence(bottom_base) != top_base:
            mismatches.append(index)
    return mismatches


def _row_sequence_for_visible_span(row: PlotFragmentRow, span: PlotSpan | None) -> str:
    if span is None:
        return ""
    if span.start < row.span.start or span.end > row.span.end:
        raise ValueError("Visible overlap span must stay inside the row display span.")
    local_start = span.start - row.span.start
    local_end = span.end - row.span.start
    return row.sequence[local_start:local_end]


def build_released_hit_plot_model(hit: ReleasedTargetSearchHit) -> ReleasedHitPlotContext:
    active_product_length = hit.projection.active_product_length_nt
    retained_partner_length = hit.projection.retained_partner_length_nt
    coordinate_offset = hit.projection.nick_coordinate_precursor - hit.projection.rebased_nick_boundary
    if coordinate_offset < 0:
        raise ValueError("Released solve plot requires a nonnegative precursor nick offset.")

    precursor_nick_boundary = hit.pre_nick_event.boundary_context
    physical_nicked_strand = hit.physical_nicked_strand
    nick_event_payload = hit.pre_nick_event.model_dump(mode="json")
    release_event_payload = hit.release_event.model_dump(mode="json")
    precursor_active_start = coordinate_offset
    precursor_active_end = coordinate_offset + active_product_length
    paired_bp = hit.final_candidate.paired_bp
    cap_nt = hit.final_candidate.cap_nt
    structure_width = (2 * paired_bp) + cap_nt
    structure_start = active_product_length - structure_width
    if structure_start < 0:
        raise ValueError("Released solve plot requires active_product_length_nt >= (2 * paired_bp) + cap_nt.")

    stem_span = _span(structure_start, structure_start + paired_bp)
    cap_span = _span(stem_span.end, stem_span.end + cap_nt)
    foldback_span = _span(cap_span.end, cap_span.end + paired_bp)
    precursor_top_span = _span(0, len(hit.precursor_top_strand))
    precursor_bottom_span = _span(0, len(hit.precursor_top_strand))
    precursor_bottom_sequence = complement_sequence(hit.precursor_top_strand)
    active_product_span = _span(0, active_product_length)
    foldback_sequence = hit.projection.active_product_sequence[foldback_span.start : foldback_span.end]
    foldback_partner_sequence = foldback_sequence[::-1]
    active_start_terminal = "5'" if hit.projection.active_strand == "top" else "3'"
    active_end_terminal = "3'" if hit.projection.active_strand == "top" else "5'"
    partner_start_terminal = "5'" if hit.projection.retained_partner_strand == "top" else "3'"
    partner_end_terminal = "3'" if hit.projection.retained_partner_strand == "top" else "5'"
    upstream_top_sequence = hit.precursor_top_strand[:coordinate_offset]
    upstream_bottom_sequence = precursor_bottom_sequence[:coordinate_offset]
    active_upstream_sequence = (
        upstream_top_sequence if hit.projection.active_strand == "top" else upstream_bottom_sequence
    )

    active_row = _fragment_row(
        role="active_product",
        strand=hit.projection.active_strand,
        label=f"Exposed {hit.projection.active_strand.title()}",
        sequence=f"{active_upstream_sequence}{hit.projection.active_product_sequence}",
        span=_span(-coordinate_offset, active_product_length),
        start_terminal=active_start_terminal,
        end_terminal=active_end_terminal,
    )
    partner_row = _fragment_row(
        role="retained_partner",
        strand=hit.projection.retained_partner_strand,
        label=hit.projection.retained_partner_strand.title(),
        sequence=hit.projection.retained_partner_sequence,
        span=_span(-coordinate_offset, retained_partner_length - coordinate_offset),
        start_terminal=partner_start_terminal,
        end_terminal=partner_end_terminal,
    )
    top_row = active_row if hit.projection.active_strand == "top" else partner_row
    bottom_row = partner_row if hit.projection.active_strand == "top" else active_row
    duplex_overlap_start = max(top_row.span.start, bottom_row.span.start)
    duplex_overlap_end = min(top_row.span.end, bottom_row.span.end)
    duplex_overlap_span = (
        _span(duplex_overlap_start, duplex_overlap_end) if duplex_overlap_end > duplex_overlap_start else None
    )
    duplex_top_sequence = _row_sequence_for_visible_span(top_row, duplex_overlap_span)
    duplex_bottom_sequence = _row_sequence_for_visible_span(bottom_row, duplex_overlap_span)
    top_only_overhang_span = (
        _span(duplex_overlap_end, top_row.span.end) if top_row.span.end > duplex_overlap_end else None
    )
    bottom_only_overhang_span = (
        _span(duplex_overlap_end, bottom_row.span.end) if bottom_row.span.end > duplex_overlap_end else None
    )

    active_foldback_row = _foldback_row(
        role="active_stem",
        label="Stem",
        sequence=f"{active_upstream_sequence}{hit.projection.active_product_sequence[stem_span.start : stem_span.end]}",
        span=_span(-coordinate_offset, paired_bp),
        left_terminal=active_start_terminal,
    )
    foldback_return_upstream = (
        upstream_bottom_sequence if hit.projection.active_strand == "top" else upstream_top_sequence
    )
    return_foldback_row = _foldback_row(
        role="foldback_return",
        label="Foldback Stem",
        sequence=f"{foldback_return_upstream}{foldback_partner_sequence}",
        span=_span(-coordinate_offset, paired_bp),
        left_terminal=active_end_terminal,
    )

    return ReleasedHitPlotContext(
        labels=PlotLabels(
            active_label=active_row.label,
            partner_label=partner_row.label,
            active_start_terminal=active_start_terminal,
            active_end_terminal=active_end_terminal,
            partner_start_terminal=partner_start_terminal,
            partner_end_terminal=partner_end_terminal,
            orientation_note="Rows stay on physical top/bottom lanes; foldback keeps the active row at origin.",
        ),
        target=PlotTarget(
            nick_boundary_from_left=hit.nick_boundary_from_left,
            paired_bp=paired_bp,
            cap_nt=cap_nt,
        ),
        nickase_variant_id=hit.nickase_variant_id,
        release_variant_id=hit.release_variant_id,
        precursor=PlotPrecursorPanelContext(
            top_sequence=hit.precursor_top_strand,
            bottom_sequence=precursor_bottom_sequence,
            nick_site=_display_site_payload(hit.pre_nick_site),
            nick_event=nick_event_payload,
            nicked_strand=physical_nicked_strand,
            release_site=_display_site_payload(hit.release_site),
            release_event=release_event_payload,
            top_span=precursor_top_span,
            bottom_span=precursor_bottom_span,
            nick_boundary=precursor_nick_boundary,
            nick_site_span=_span(hit.pre_nick_event.source_site_start, hit.pre_nick_event.source_site_end),
            release_site_span=_span(hit.release_event.source_site_start, hit.release_event.source_site_end),
            retained_partner_span=_span(0, precursor_nick_boundary),
            active_product_span=_span(precursor_active_start, precursor_active_end),
            sacrificial_top_tail_span=(
                _span(precursor_active_end, len(hit.precursor_top_strand))
                if hit.projection.active_strand == "top"
                else _span(precursor_nick_boundary, len(hit.precursor_top_strand))
            ),
            sacrificial_bottom_tail_span=(
                _span(precursor_nick_boundary, len(hit.precursor_top_strand))
                if hit.projection.active_strand == "top"
                else _span(precursor_active_end, len(hit.precursor_top_strand))
            ),
        ),
        released_product=PlotReleasedProductContext(
            retained_partner_sequence=hit.projection.retained_partner_sequence,
            active_product_sequence=hit.projection.active_product_sequence,
            nick_boundary=hit.projection.rebased_nick_boundary,
            release_top_cut_boundary=hit.projection.release_top_cut_precursor - coordinate_offset,
            release_bottom_cut_boundary=hit.projection.release_bottom_cut_precursor - coordinate_offset,
            upstream_context_span=_span(-coordinate_offset, 0),
            retained_partner_span=partner_row.span,
            active_product_span=active_product_span,
            nicked_strand=physical_nicked_strand,
            top_row=top_row,
            bottom_row=bottom_row,
            duplex_overlap_span=duplex_overlap_span,
            duplex_top_sequence=duplex_top_sequence,
            duplex_bottom_sequence=duplex_bottom_sequence,
            duplex_mismatch_positions=[
                duplex_overlap_start + mismatch_position
                for mismatch_position in _watson_crick_mismatch_positions(
                    top_sequence=duplex_top_sequence,
                    bottom_sequence=duplex_bottom_sequence,
                )
            ],
            top_only_overhang_span=top_only_overhang_span,
            bottom_only_overhang_span=bottom_only_overhang_span,
            active_prefix_span=_span(0, structure_start),
            stem_span=stem_span,
            cap_span=cap_span,
            foldback_span=foldback_span,
            nickase_site_survives_post_release=hit.projection.nickase_site_survives_post_release,
            release_site_survives_post_release=hit.projection.release_site_survives_post_release,
        ),
        foldback=PlotFoldbackPanelContext(
            origin_boundary_from_left=stem_span.start,
            stem_sequence=hit.projection.active_product_sequence[stem_span.start : stem_span.end],
            cap_sequence=hit.projection.active_product_sequence[cap_span.start : cap_span.end],
            foldback_sequence=foldback_sequence,
            foldback_partner_sequence=foldback_partner_sequence,
            upstream_context_span=_span(-coordinate_offset, 0),
            nicked_strand=physical_nicked_strand,
            top_row=active_foldback_row if hit.projection.active_strand == "top" else return_foldback_row,
            bottom_row=return_foldback_row if hit.projection.active_strand == "top" else active_foldback_row,
            foldback_mismatch_positions=_watson_crick_mismatch_positions(
                top_sequence=hit.projection.active_product_sequence[stem_span.start : stem_span.end],
                bottom_sequence=foldback_partner_sequence,
            ),
        ),
    )


__all__ = ["build_released_hit_plot_model"]
