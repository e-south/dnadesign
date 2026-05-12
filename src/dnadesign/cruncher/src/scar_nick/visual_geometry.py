"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/scar_nick/visual_geometry.py

Coordinate helpers for scar_nick terminal-nick visual contracts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from dnadesign.cruncher.nickases.models import iupac_bases_for_symbol, reverse_complement_iupac
from dnadesign.cruncher.scar_nick.models import ScarNickCandidate

_ALL_BASES = frozenset({"A", "C", "G", "T"})


@dataclass(frozen=True)
class ScarNickVisualContext:
    primary_sequence_5to3: str
    context_start: int
    context_end: int
    retained_product_span: dict[str, int]
    release_site_span: dict[str, int]
    type_iis_offset_span: dict[str, int] | None
    retained_scar_span: dict[str, int]
    junction_partner_span: dict[str, int] | None
    nickase_site_span: dict[str, int]
    nickase_site_source_span: dict[str, int]


def complement_sequence(sequence: str) -> str:
    return reverse_complement_iupac(sequence)[::-1]


def recognition_nt(motif: str) -> int:
    return sum(1 for symbol in motif if frozenset(iupac_bases_for_symbol(symbol)) != _ALL_BASES)


def span(raw_start: int, raw_end: int, *, context_start: int) -> dict[str, int]:
    return {"start": raw_start - context_start, "end": raw_end - context_start}


def shift_span(raw_span: dict[str, int], offset: int) -> dict[str, int]:
    return {"start": raw_span["start"] + offset, "end": raw_span["end"] + offset}


def shift_optional_span(raw_span: dict[str, int] | None, offset: int) -> dict[str, int] | None:
    if raw_span is None:
        return None
    return shift_span(raw_span, offset)


def pairing_complement_sequence(
    *,
    sequence: str,
    context: ScarNickVisualContext,
    candidate: ScarNickCandidate,
) -> str:
    complement = list(complement_sequence(sequence))
    scar_start = context.retained_scar_span["start"]
    scar_end = context.retained_scar_span["end"]
    complement[scar_start:scar_end] = list(candidate.right_base[::-1])
    return "".join(complement)


def build_visual_context(candidate: ScarNickCandidate) -> ScarNickVisualContext:
    if candidate.release_placement is None:
        raise ValueError("scar-nick visual requires a release placement")
    if candidate.nickase_placement is None:
        raise ValueError("scar-nick visual requires a nickase placement")
    release = candidate.release_placement
    nickase = candidate.nickase_placement
    product_end = len(candidate.retained_product_sequence)
    context_start = min(release.recognition_site_start, nickase.source_site_start, 0)
    context_end = max(release.recognition_site_end, nickase.source_site_end, product_end)

    symbols = ["N"] * (context_end - context_start)
    _write_span(
        symbols,
        context_start=context_start,
        raw_start=release.recognition_site_start,
        sequence=release.recognition_sequence,
        semantic="type_iis_release_site",
    )
    _write_span(
        symbols,
        context_start=context_start,
        raw_start=nickase.source_site_start,
        sequence=nickase.motif_top_5to3,
        semantic="nickase_footprint",
    )
    _write_span(
        symbols,
        context_start=context_start,
        raw_start=0,
        sequence=candidate.retained_product_sequence,
        semantic="retained_product",
    )

    offset_span = None
    if release.recognition_site_end < release.top_cut_boundary:
        offset_span = span(release.recognition_site_end, release.top_cut_boundary, context_start=context_start)

    return ScarNickVisualContext(
        primary_sequence_5to3="".join(symbols),
        context_start=context_start,
        context_end=context_end,
        retained_product_span=span(0, product_end, context_start=context_start),
        release_site_span=span(
            release.recognition_site_start,
            release.recognition_site_end,
            context_start=context_start,
        ),
        type_iis_offset_span=offset_span,
        retained_scar_span=span(0, candidate.retained_scar_nt, context_start=context_start),
        junction_partner_span=None,
        nickase_site_span=span(nickase.source_site_start, nickase.source_site_end, context_start=context_start),
        nickase_site_source_span=_nickase_source_span(candidate),
    )


def protected_sequence_spans(candidate: ScarNickCandidate, context: ScarNickVisualContext) -> list[dict[str, Any]]:
    release = candidate.release_placement
    if release is None:
        return []
    return [
        {
            "semantic": "type_iis_release_site",
            "start": context.release_site_span["start"],
            "end": context.release_site_span["end"],
            "raw_start": release.recognition_site_start,
            "raw_end": release.recognition_site_end,
            "mutable": False,
        },
        {
            "semantic": "retained_product",
            "start": context.retained_product_span["start"],
            "end": context.retained_product_span["end"],
            "raw_start": 0,
            "raw_end": len(candidate.retained_product_sequence),
            "mutable": False,
        },
    ]


def nickase_downstream_symbols(candidate: ScarNickCandidate, context: ScarNickVisualContext) -> list[dict[str, Any]]:
    placement = candidate.nickase_placement
    if placement is None:
        return []
    payload: list[dict[str, Any]] = []
    for offset, symbol in enumerate(placement.motif_top_5to3):
        coordinate = placement.source_site_start + offset
        if coordinate < candidate.terminal_boundary:
            continue
        payload.append(
            {
                "raw_coordinate": coordinate,
                "display_index": coordinate - context.context_start,
                "symbol": symbol,
                "fully_degenerate": len(iupac_bases_for_symbol(symbol)) == 4,
            }
        )
    return payload


def degenerate_nickase_motif_indices(candidate: ScarNickCandidate, context: ScarNickVisualContext) -> list[int]:
    placement = candidate.nickase_placement
    if placement is None:
        return []
    indices: list[int] = []
    for offset, symbol in enumerate(placement.motif_top_5to3):
        if len(iupac_bases_for_symbol(symbol)) != 4:
            continue
        coordinate = placement.source_site_start + offset
        display_index = coordinate - context.context_start
        if 0 <= display_index < len(context.primary_sequence_5to3):
            indices.append(display_index)
    return indices


def _nickase_source_span(candidate: ScarNickCandidate) -> dict[str, int]:
    placement = candidate.nickase_placement
    if placement is None:
        raise ValueError("scar-nick visual requires a nickase placement")
    return {"start": placement.source_site_start, "end": placement.source_site_end}


def _iupac_symbols_overlap(left_symbol: str, right_symbol: str) -> bool:
    return bool(iupac_bases_for_symbol(left_symbol) & iupac_bases_for_symbol(right_symbol))


def _merge_symbol(existing: str, incoming: str, *, coordinate: int, semantic: str) -> str:
    if not _iupac_symbols_overlap(existing, incoming):
        raise ValueError(
            f"scar-nick visual sequence conflict at raw coordinate {coordinate}: "
            f"{existing!r} cannot satisfy {semantic} symbol {incoming!r}"
        )
    if existing == "N":
        return incoming
    if incoming == "N":
        return existing
    if len(iupac_bases_for_symbol(existing)) == 1:
        return existing
    if len(iupac_bases_for_symbol(incoming)) == 1:
        return incoming
    return existing


def _write_span(
    symbols: list[str],
    *,
    context_start: int,
    raw_start: int,
    sequence: str,
    semantic: str,
) -> None:
    for offset, symbol in enumerate(sequence):
        coordinate = raw_start + offset
        index = coordinate - context_start
        symbols[index] = _merge_symbol(symbols[index], symbol, coordinate=coordinate, semantic=semantic)


__all__ = [
    "ScarNickVisualContext",
    "build_visual_context",
    "complement_sequence",
    "degenerate_nickase_motif_indices",
    "nickase_downstream_symbols",
    "pairing_complement_sequence",
    "protected_sequence_spans",
    "recognition_nt",
    "shift_optional_span",
    "shift_span",
]
