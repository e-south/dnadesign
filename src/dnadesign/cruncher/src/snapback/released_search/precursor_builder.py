"""
Precursor construction for released-product target-search.
"""

from __future__ import annotations

from dnadesign.cruncher.nickases.models import iupac_bases_for_symbol
from dnadesign.cruncher.snapback.released_search.placement_models import (
    BuiltPrecursor,
    BuiltPrecursorResult,
    NickPlacement,
    ReleasePlacement,
)
from dnadesign.cruncher.snapback.released_spec_models import ReleasedFinalTargetGeometry

_DNA_BASES = ("A", "C", "G", "T")
_COMPLEMENT = {"A": "T", "C": "G", "G": "C", "T": "A"}


def apply_site_constraint(allowed: list[set[str]], *, motif: str, site_start: int) -> bool:
    if site_start + len(motif) > len(allowed):
        return False
    motif_offset = 0
    sequence_start = site_start
    if site_start < 0:
        motif_offset = -site_start
        if any(set(iupac_bases_for_symbol(symbol)) != set(_DNA_BASES) for symbol in motif[:motif_offset]):
            return False
        sequence_start = 0
    for offset, symbol in enumerate(motif[motif_offset:]):
        allowed[sequence_start + offset] &= set(iupac_bases_for_symbol(symbol))
        if not allowed[sequence_start + offset]:
            return False
    return True


def pair_map(*, boundary: int, target: ReleasedFinalTargetGeometry, coordinate_offset: int) -> dict[int, int]:
    mapping: dict[int, int] = {}
    input_length = coordinate_offset + boundary + target.paired_bp + target.cap_nt
    for index in range(target.paired_bp):
        left = coordinate_offset + boundary + index
        right = input_length + (target.paired_bp - 1 - index)
        mapping[left] = right
        mapping[right] = left
    return mapping


def build_precursor_sequence(
    *,
    boundary: int,
    target: ReleasedFinalTargetGeometry,
    nick_placement: NickPlacement,
    release_placement: ReleasePlacement,
    allow_precut_footprint_outside_active_product: bool = False,
) -> BuiltPrecursorResult:
    active_product_length = boundary + (2 * target.paired_bp) + target.cap_nt
    nick_site_start = nick_placement.site_start_for_boundary(boundary)
    release_site_start = release_placement.site_start_for_boundary(boundary)
    if not allow_precut_footprint_outside_active_product:
        if not nick_placement.allows_left_of_origin_prefix(boundary):
            return BuiltPrecursorResult(precursor=None, blocker_code="PRE_NICK_SITE_LEFT_OF_ORIGIN")
        if release_site_start < 0:
            return BuiltPrecursorResult(precursor=None, blocker_code="RELEASE_SITE_LEFT_OF_ORIGIN")
        coordinate_offset = 0
    else:
        if nick_site_start < 0 and not nick_placement.allows_left_of_origin_prefix(boundary):
            return BuiltPrecursorResult(precursor=None, blocker_code="PRE_NICK_SITE_LEFT_OF_ORIGIN")
        if release_site_start < 0:
            return BuiltPrecursorResult(precursor=None, blocker_code="RELEASE_SITE_LEFT_OF_ORIGIN")
        coordinate_offset = max(0, -nick_site_start, -release_site_start)
    top_cut = release_placement.top_cut_for_boundary(boundary) + coordinate_offset
    bottom_cut = release_placement.bottom_cut_for_boundary(boundary) + coordinate_offset
    active_cut = release_placement.active_cut_for_boundary(boundary) + coordinate_offset
    nick_site_start += coordinate_offset
    release_site_start += coordinate_offset
    precursor_length = max(
        coordinate_offset + active_product_length,
        top_cut + 1,
        bottom_cut + 1,
        active_cut + 1,
        nick_site_start + len(nick_placement.motif),
        release_site_start + len(release_placement.motif),
    )
    allowed = [set(_DNA_BASES) for _ in range(precursor_length)]
    if not apply_site_constraint(allowed, motif=nick_placement.motif, site_start=nick_site_start):
        return BuiltPrecursorResult(precursor=None, blocker_code="FOOTPRINT_NOT_CONSTRUCTABLE")
    if not apply_site_constraint(allowed, motif=release_placement.motif, site_start=release_site_start):
        return BuiltPrecursorResult(precursor=None, blocker_code="FOOTPRINT_NOT_CONSTRUCTABLE")
    pairs = pair_map(boundary=boundary, target=target, coordinate_offset=coordinate_offset)
    assigned: list[str | None] = [None] * precursor_length
    for index in range(precursor_length):
        if assigned[index] is not None:
            continue
        partner = pairs.get(index)
        if partner is None:
            if not allowed[index]:
                return BuiltPrecursorResult(precursor=None, blocker_code="FOOTPRINT_NOT_CONSTRUCTABLE")
            assigned[index] = sorted(allowed[index])[0]
            continue
        if partner < index:
            candidate = _COMPLEMENT[str(assigned[partner])]
            if candidate not in allowed[index]:
                return BuiltPrecursorResult(precursor=None, blocker_code="FOOTPRINT_NOT_CONSTRUCTABLE")
            assigned[index] = candidate
            continue
        choices = [base for base in sorted(allowed[index]) if _COMPLEMENT[base] in allowed[partner]]
        if not choices:
            return BuiltPrecursorResult(precursor=None, blocker_code="FOOTPRINT_NOT_CONSTRUCTABLE")
        assigned[index] = choices[0]
        assigned[partner] = _COMPLEMENT[choices[0]]
    return BuiltPrecursorResult(
        precursor=BuiltPrecursor(
            top_strand="".join(str(base) for base in assigned),
            coordinate_offset=coordinate_offset,
        )
    )


__all__ = ["apply_site_constraint", "build_precursor_sequence", "pair_map"]
