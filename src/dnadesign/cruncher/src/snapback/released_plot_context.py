"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cruncher/src/snapback/released_plot_context.py

Typed context builder for released-product snapback hit plots.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any

from dnadesign.cruncher.snapback.publication_support import complement_sequence
from dnadesign.cruncher.snapback.released_plot_common import (
    ROW_LABEL_FOLDBACK_STEM,
    ROW_LABEL_STEM,
    post_release_physical_row_label,
)
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

_FIXED_IUPAC_BASES = frozenset({"A", "C", "G", "T"})


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
    physical_state: str,
    strand: str,
    label: str,
    sequence: str,
    span: PlotSpan,
    start_terminal: str | None,
    end_terminal: str | None,
    assignable_base_positions: list[int] | None = None,
) -> PlotFragmentRow:
    return PlotFragmentRow(
        role=role,
        physical_state=physical_state,
        strand=strand,
        label=label,
        sequence=sequence,
        span=span,
        start_terminal=start_terminal,
        end_terminal=end_terminal,
        assignable_base_positions=assignable_base_positions or [],
    )


def _foldback_row(
    *,
    role: str,
    label: str,
    sequence: str,
    span: PlotSpan,
    left_terminal: str | None,
    assignable_base_positions: list[int] | None = None,
) -> PlotFoldbackRow:
    return PlotFoldbackRow(
        role=role,
        label=label,
        sequence=sequence,
        span=span,
        left_terminal=left_terminal,
        assignable_base_positions=assignable_base_positions or [],
    )


def _watson_crick_mismatch_positions(*, top_sequence: str, bottom_sequence: str, span_start: int = 0) -> list[int]:
    mismatches: list[int] = []
    for index, (top_base, bottom_base) in enumerate(zip(top_sequence, bottom_sequence, strict=True)):
        if top_base not in _FIXED_IUPAC_BASES or bottom_base not in _FIXED_IUPAC_BASES:
            continue
        if complement_sequence(bottom_base) != top_base:
            mismatches.append(span_start + index)
    return mismatches


def _row_sequence_for_visible_span(row: PlotFragmentRow, span: PlotSpan | None) -> str:
    if span is None:
        return ""
    if span.start < row.span.start or span.end > row.span.end:
        raise ValueError("Visible overlap span must stay inside the row display span.")
    local_start = span.start - row.span.start
    local_end = span.end - row.span.start
    return row.sequence[local_start:local_end]


def _degenerate_active_product_indexes(hit: ReleasedTargetSearchHit) -> set[int]:
    return {
        base.active_index
        for base in hit.projection.active_product_provenance
        if base.source_constraint == "degenerate_motif_base"
    }


def _symbolic_positions(*, sequence: str, span_start: int) -> set[int]:
    return {span_start + index for index, base in enumerate(sequence) if base.upper() not in _FIXED_IUPAC_BASES}


def _precursor_display_sequences(
    *,
    hit: ReleasedTargetSearchHit,
    precursor_bottom_sequence: str,
) -> tuple[str, str, list[int], list[int]]:
    top_bases = list(hit.precursor_top_strand)
    bottom_bases = list(precursor_bottom_sequence)
    top_assignable_positions: set[int] = set()
    bottom_assignable_positions: set[int] = set()

    def mark_position(position: int) -> None:
        if position < 0 or position >= len(top_bases):
            return
        top_bases[position] = "N"
        bottom_bases[position] = "N"
        top_assignable_positions.add(position)
        bottom_assignable_positions.add(position)

    def mark_site_degenerate_positions(
        *,
        site_start: int,
        site_end: int,
        orientation: str,
        canonical_pattern: str,
    ) -> None:
        site_width = site_end - site_start
        if site_width <= 0:
            return
        pattern = canonical_pattern.upper()
        if len(pattern) < site_width:
            pattern = f"{pattern}{'N' * (site_width - len(pattern))}"
        for canonical_index, symbol in enumerate(pattern[:site_width]):
            if symbol in _FIXED_IUPAC_BASES:
                continue
            offset = canonical_index if orientation == "forward" else site_width - 1 - canonical_index
            mark_position(site_start + offset)

    mark_site_degenerate_positions(
        site_start=hit.pre_nick_site.start,
        site_end=hit.pre_nick_site.end,
        orientation=hit.pre_nick_site.orientation,
        canonical_pattern=hit.nickase.motif_top_5to3,
    )
    mark_site_degenerate_positions(
        site_start=hit.release_site.start,
        site_end=hit.release_site.end,
        orientation=hit.release_site.orientation,
        canonical_pattern=hit.release_enzyme.recognition_sequence,
    )
    for base in hit.projection.active_product_provenance:
        if base.source_constraint == "degenerate_motif_base":
            mark_position(base.precursor_index)

    return (
        "".join(top_bases),
        "".join(bottom_bases),
        sorted(top_assignable_positions),
        sorted(bottom_assignable_positions),
    )


def _active_foldback_assignable_positions(
    *,
    degenerate_active_indexes: set[int],
    upstream_retained_duplex_bp: int,
    stem_span: PlotSpan,
) -> list[int]:
    positions: set[int] = set()
    for active_index in degenerate_active_indexes:
        if 0 <= active_index < upstream_retained_duplex_bp:
            positions.add(active_index)
        if stem_span.start <= active_index < stem_span.end:
            positions.add(upstream_retained_duplex_bp + (active_index - stem_span.start))
    return sorted(positions)


def _return_foldback_assignable_positions(
    *,
    degenerate_active_indexes: set[int],
    upstream_retained_duplex_bp: int,
    foldback_span: PlotSpan,
) -> list[int]:
    positions: set[int] = set()
    for active_index in degenerate_active_indexes:
        if foldback_span.start <= active_index < foldback_span.end:
            reversed_local_index = foldback_span.end - 1 - active_index
            positions.add(upstream_retained_duplex_bp + reversed_local_index)
    return sorted(positions)


def _cap_assignable_positions(
    *,
    degenerate_active_indexes: set[int],
    cap_span: PlotSpan,
) -> list[int]:
    return sorted(
        active_index - cap_span.start
        for active_index in degenerate_active_indexes
        if cap_span.start <= active_index < cap_span.end
    )


def _symbolic_duplex_prefix(
    *,
    top_row: PlotFragmentRow,
    bottom_row: PlotFragmentRow,
) -> tuple[int, str, str, list[int]]:
    prefix_start = max(top_row.span.start, bottom_row.span.start)
    prefix_end = min(0, top_row.span.end, bottom_row.span.end)
    if prefix_end <= prefix_start:
        return (0, "", "", [])

    prefix_span = _span(prefix_start, prefix_end)
    top_sequence = _row_sequence_for_visible_span(top_row, prefix_span)
    bottom_sequence = _row_sequence_for_visible_span(bottom_row, prefix_span)

    selected_start = prefix_end
    for index in range(len(top_sequence) - 1, -1, -1):
        top_base = top_sequence[index].upper()
        bottom_base = bottom_sequence[index].upper()
        if top_base in _FIXED_IUPAC_BASES and bottom_base in _FIXED_IUPAC_BASES:
            break
        selected_start = prefix_start + index

    if selected_start == prefix_end:
        return (0, "", "", [])

    local_start = selected_start - prefix_start
    selected_top_sequence = top_sequence[local_start:]
    selected_bottom_sequence = bottom_sequence[local_start:]
    assignable_positions = [
        selected_start + index
        for index, (top_base, bottom_base) in enumerate(
            zip(selected_top_sequence, selected_bottom_sequence, strict=True)
        )
        if top_base.upper() not in _FIXED_IUPAC_BASES or bottom_base.upper() not in _FIXED_IUPAC_BASES
    ]
    return (selected_start, selected_top_sequence, selected_bottom_sequence, assignable_positions)


def _physical_state_for_strand(*, strand: str, nicked_strand: str) -> str:
    return "released" if strand == nicked_strand else "retained"


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
    (
        precursor_top_display_sequence,
        precursor_bottom_display_sequence,
        precursor_top_assignable_positions,
        precursor_bottom_assignable_positions,
    ) = _precursor_display_sequences(
        hit=hit,
        precursor_bottom_sequence=precursor_bottom_sequence,
    )
    active_product_span = _span(0, active_product_length)
    foldback_sequence = hit.projection.active_product_sequence[foldback_span.start : foldback_span.end]
    foldback_partner_sequence = foldback_sequence[::-1]
    active_start_terminal = "5'" if hit.projection.active_strand == "top" else "3'"
    active_end_terminal = "3'" if hit.projection.active_strand == "top" else "5'"
    partner_start_terminal = "5'" if hit.projection.retained_partner_strand == "top" else "3'"
    partner_end_terminal = "3'" if hit.projection.retained_partner_strand == "top" else "5'"
    upstream_top_sequence = precursor_top_display_sequence[:coordinate_offset]
    upstream_bottom_sequence = precursor_bottom_display_sequence[:coordinate_offset]
    displayed_retained_partner_sequence = (
        precursor_top_display_sequence[:retained_partner_length]
        if hit.projection.retained_partner_strand == "top"
        else precursor_bottom_display_sequence[:retained_partner_length]
    )
    active_upstream_sequence = (
        upstream_top_sequence if hit.projection.active_strand == "top" else upstream_bottom_sequence
    )
    active_label = post_release_physical_row_label(
        strand=hit.projection.active_strand,
        nicked_strand=physical_nicked_strand,
    )
    partner_label = post_release_physical_row_label(
        strand=hit.projection.retained_partner_strand,
        nicked_strand=physical_nicked_strand,
    )
    degenerate_active_indexes = _degenerate_active_product_indexes(hit)

    active_row = _fragment_row(
        role="active_product",
        physical_state=_physical_state_for_strand(
            strand=hit.projection.active_strand,
            nicked_strand=physical_nicked_strand,
        ),
        strand=hit.projection.active_strand,
        label=active_label,
        sequence=f"{active_upstream_sequence}{hit.projection.active_product_sequence}",
        span=_span(-coordinate_offset, active_product_length),
        start_terminal=active_start_terminal,
        end_terminal=active_end_terminal,
        assignable_base_positions=sorted(
            degenerate_active_indexes
            | _symbolic_positions(sequence=active_upstream_sequence, span_start=-coordinate_offset)
        ),
    )
    partner_row = _fragment_row(
        role="retained_partner",
        physical_state=_physical_state_for_strand(
            strand=hit.projection.retained_partner_strand,
            nicked_strand=physical_nicked_strand,
        ),
        strand=hit.projection.retained_partner_strand,
        label=partner_label,
        sequence=displayed_retained_partner_sequence,
        span=_span(-coordinate_offset, retained_partner_length - coordinate_offset),
        start_terminal=partner_start_terminal,
        end_terminal=partner_end_terminal,
        assignable_base_positions=sorted(
            _symbolic_positions(sequence=displayed_retained_partner_sequence, span_start=-coordinate_offset)
        ),
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
    (
        foldback_prefix_start,
        foldback_top_prefix_sequence,
        foldback_bottom_prefix_sequence,
        foldback_prefix_assignable_positions,
    ) = _symbolic_duplex_prefix(top_row=top_row, bottom_row=bottom_row)
    foldback_display_start = min(0, foldback_prefix_start)

    upstream_retained_duplex_bp = hit.upstream_retained_duplex_bp
    effective_stem_end = upstream_retained_duplex_bp + paired_bp
    active_prefix_sequence = (
        foldback_top_prefix_sequence if hit.projection.active_strand == "top" else foldback_bottom_prefix_sequence
    )
    return_prefix_sequence = (
        foldback_bottom_prefix_sequence if hit.projection.active_strand == "top" else foldback_top_prefix_sequence
    )
    active_foldback_sequence = (
        active_prefix_sequence
        + hit.projection.active_product_sequence[:upstream_retained_duplex_bp]
        + hit.projection.active_product_sequence[stem_span.start : stem_span.end]
    )
    retained_upstream_sequence = hit.projection.retained_partner_sequence[:upstream_retained_duplex_bp]
    active_foldback_row = _foldback_row(
        role="active_stem",
        label=ROW_LABEL_STEM,
        sequence=active_foldback_sequence,
        span=_span(foldback_display_start, effective_stem_end),
        left_terminal=active_start_terminal,
        assignable_base_positions=sorted(
            set(foldback_prefix_assignable_positions)
            | set(
                _active_foldback_assignable_positions(
                    degenerate_active_indexes=degenerate_active_indexes,
                    upstream_retained_duplex_bp=upstream_retained_duplex_bp,
                    stem_span=stem_span,
                )
            )
        ),
    )
    return_foldback_row = _foldback_row(
        role="foldback_return",
        label=ROW_LABEL_FOLDBACK_STEM,
        sequence=f"{return_prefix_sequence}{retained_upstream_sequence}{foldback_partner_sequence}",
        span=_span(foldback_display_start, effective_stem_end),
        left_terminal=active_end_terminal,
        assignable_base_positions=sorted(
            set(foldback_prefix_assignable_positions)
            | set(
                _return_foldback_assignable_positions(
                    degenerate_active_indexes=degenerate_active_indexes,
                    upstream_retained_duplex_bp=upstream_retained_duplex_bp,
                    foldback_span=foldback_span,
                )
            )
        ),
    )
    foldback_top_row = active_foldback_row if hit.projection.active_strand == "top" else return_foldback_row
    foldback_bottom_row = return_foldback_row if hit.projection.active_strand == "top" else active_foldback_row

    return ReleasedHitPlotContext(
        labels=PlotLabels(
            active_label=active_row.label,
            partner_label=partner_row.label,
            active_start_terminal=active_start_terminal,
            active_end_terminal=active_end_terminal,
            partner_start_terminal=partner_start_terminal,
            partner_end_terminal=partner_end_terminal,
            orientation_note=(
                "Rows stay on physical top/bottom lanes; foldback includes retained upstream duplex before the nick."
            ),
        ),
        target=PlotTarget(
            nick_boundary_from_left=hit.nick_boundary_from_left,
            paired_bp=paired_bp,
            cap_nt=cap_nt,
        ),
        nickase_variant_id=hit.nickase_variant_id,
        release_variant_id=hit.release_variant_id,
        precursor=PlotPrecursorPanelContext(
            top_sequence=precursor_top_display_sequence,
            bottom_sequence=precursor_bottom_display_sequence,
            top_assignable_base_positions=precursor_top_assignable_positions,
            bottom_assignable_base_positions=precursor_bottom_assignable_positions,
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
            origin_boundary_from_left=upstream_retained_duplex_bp,
            stem_sequence=hit.projection.active_product_sequence[stem_span.start : stem_span.end],
            cap_sequence=hit.projection.active_product_sequence[cap_span.start : cap_span.end],
            foldback_sequence=foldback_sequence,
            foldback_partner_sequence=foldback_partner_sequence,
            upstream_context_span=_span(foldback_display_start, upstream_retained_duplex_bp),
            nicked_strand=physical_nicked_strand,
            top_row=foldback_top_row,
            bottom_row=foldback_bottom_row,
            foldback_mismatch_positions=_watson_crick_mismatch_positions(
                top_sequence=foldback_top_row.sequence,
                bottom_sequence=foldback_bottom_row.sequence,
                span_start=max(foldback_top_row.span.start, foldback_bottom_row.span.start),
            ),
            assignable_cap_base_positions=_cap_assignable_positions(
                degenerate_active_indexes=degenerate_active_indexes,
                cap_span=cap_span,
            ),
        ),
    )


__all__ = ["build_released_hit_plot_model"]
