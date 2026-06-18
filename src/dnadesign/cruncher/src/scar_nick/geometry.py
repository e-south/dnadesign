"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cruncher/src/scar_nick/geometry.py

Coordinate-domain helpers for terminal scar-nick enzyme geometry.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from itertools import product

from dnadesign.cruncher.nickases.models import NickaseCatalog, NickaseCatalogEntry, iupac_bases_for_symbol
from dnadesign.cruncher.nickases.scanning import (
    display_footprint_for_orientation,
    enumerate_boundary_placements,
)
from dnadesign.cruncher.scar_nick.models import (
    NickaseDownstreamSymbol,
    NickaseGeometryAuditEntry,
    NickasePlacement,
    NickaseReleaseOverlapConflict,
    ReleasePlacement,
    RetainedScarDomain,
)

_BASES = ("A", "C", "G", "T")
_ALL_BASES = frozenset(_BASES)


def iupac_symbol_is_fully_degenerate(symbol: str) -> bool:
    return frozenset(iupac_bases_for_symbol(symbol)) == _ALL_BASES


def iupac_symbols_overlap(left_symbol: str, right_symbol: str) -> bool:
    return bool(iupac_bases_for_symbol(left_symbol) & iupac_bases_for_symbol(right_symbol))


def nickase_recognition_nt(entry: NickaseCatalogEntry) -> int:
    return sum(1 for symbol in entry.motif_top_5to3 if not iupac_symbol_is_fully_degenerate(symbol))


def entry_warning_codes(entry: NickaseCatalogEntry) -> list[str]:
    if entry.selection is None:
        return []
    return [str(code).strip().upper() for code in entry.selection.warning_codes]


def entry_commercial_confidence(entry: NickaseCatalogEntry) -> str | None:
    if entry.selection is None:
        return None
    return entry.selection.commercial_confidence


def nickase_entry_rejection_reasons(
    entry: NickaseCatalogEntry,
    *,
    min_recognition_nt: int,
    disallowed_warning_codes: list[str],
) -> list[str]:
    reasons: list[str] = []
    if nickase_recognition_nt(entry) < min_recognition_nt:
        reasons.append("NICKASE_RECOGNITION_SITE_TOO_SHORT")
    if str(entry.source_family or "") != "nicking_endonuclease":
        reasons.append("NICKASE_SOURCE_FAMILY_NOT_NICKING_ENDONUCLEASE")
    if not entry.vendor:
        reasons.append("NICKASE_VENDOR_METADATA_MISSING")
    if not entry.source_url:
        reasons.append("NICKASE_SOURCE_URL_MISSING")
    if entry_commercial_confidence(entry) is None:
        reasons.append("NICKASE_COMMERCIAL_CONFIDENCE_MISSING")
    present_warnings = set(entry_warning_codes(entry))
    disallowed = sorted(present_warnings & set(disallowed_warning_codes))
    if disallowed:
        reasons.append("NICKASE_WARNING_CODE_DISALLOWED:" + ",".join(disallowed))
    return reasons


def placement_respects_terminal_downstream_rule(placement: NickasePlacement) -> bool:
    for offset, symbol in enumerate(placement.motif_top_5to3):
        coordinate = placement.source_site_start + offset
        if coordinate >= placement.terminal_boundary and not iupac_symbol_is_fully_degenerate(symbol):
            return False
    return True


def _placement_for_entry(
    entry: NickaseCatalogEntry,
    *,
    orientation: str,
    start: int,
    target_strand: str,
    terminal_boundary: int,
    boundary: int,
) -> NickasePlacement | None:
    if entry.vendor is None or entry.source_url is None or entry.source_family != "nicking_endonuclease":
        return None
    commercial_confidence = entry_commercial_confidence(entry)
    if commercial_confidence is None:
        return None
    motif = display_footprint_for_orientation(entry, orientation=orientation)
    return NickasePlacement(
        variant_id=entry.id,
        specificity_id=entry.specificity_id,
        orientation=orientation,
        motif_top_5to3=motif,
        canonical_motif_top_5to3=entry.resolved_vendor_diagram_top_5to3,
        vendor=entry.vendor,
        source_url=entry.source_url,
        source_family=entry.source_family,
        commercial_confidence=commercial_confidence,
        warning_codes=entry_warning_codes(entry),
        source_site_start=start,
        source_site_end=start + len(motif),
        strand=target_strand,
        boundary=boundary,
        terminal_boundary=terminal_boundary,
        boundary_distance=abs(boundary - terminal_boundary),
        exact_terminal=boundary == terminal_boundary,
    )


def placements_for_entry(
    entry: NickaseCatalogEntry,
    *,
    terminal_boundary: int,
    boundary: int,
    target_strand: str = "bottom",
    min_recognition_nt: int = 4,
    disallowed_warning_codes: list[str] | None = None,
) -> list[NickasePlacement]:
    if nickase_entry_rejection_reasons(
        entry,
        min_recognition_nt=min_recognition_nt,
        disallowed_warning_codes=disallowed_warning_codes or [],
    ):
        return []

    target_strands = ("top", "bottom") if target_strand == "either" else (target_strand,)
    placements: list[NickasePlacement] = []
    for strand in target_strands:
        required_strand = "primary" if strand == "top" else "complement"
        for orientation, start in enumerate_boundary_placements(
            entry,
            boundary=boundary,
            required_strand=required_strand,
            use_vendor_diagram=True,
        ):
            placement = _placement_for_entry(
                entry,
                orientation=orientation,
                start=start,
                target_strand=strand,
                terminal_boundary=terminal_boundary,
                boundary=boundary,
            )
            if placement is not None:
                placements.append(placement)
    return [placement for placement in placements if placement_respects_terminal_downstream_rule(placement)]


def _motif_map(*, motif: str, source_site_start: int) -> dict[int, str]:
    return {source_site_start + offset: symbol for offset, symbol in enumerate(motif)}


def _release_requirement_map(release_placement: ReleasePlacement) -> dict[int, str]:
    return {
        release_placement.recognition_site_start + offset: symbol
        for offset, symbol in enumerate(release_placement.recognition_sequence)
    }


def _release_overlap_conflicts(
    motif_map: dict[int, str],
    release_placement: ReleasePlacement,
) -> list[NickaseReleaseOverlapConflict]:
    release_requirements = _release_requirement_map(release_placement)
    conflicts: list[NickaseReleaseOverlapConflict] = []
    for coordinate, nickase_symbol in sorted(motif_map.items()):
        release_symbol = release_requirements.get(coordinate)
        if release_symbol is None:
            continue
        if not iupac_symbols_overlap(nickase_symbol, release_symbol):
            conflicts.append(
                NickaseReleaseOverlapConflict(
                    raw_coordinate=coordinate,
                    nickase_symbol=nickase_symbol,
                    release_symbol=release_symbol,
                )
            )
    return conflicts


def _downstream_symbols(
    motif_map: dict[int, str],
    *,
    terminal_boundary: int,
) -> list[NickaseDownstreamSymbol]:
    return [
        NickaseDownstreamSymbol(
            raw_coordinate=coordinate,
            symbol=symbol,
            fully_degenerate=iupac_symbol_is_fully_degenerate(symbol),
        )
        for coordinate, symbol in sorted(motif_map.items())
        if coordinate >= terminal_boundary
    ]


def _retained_scar_domains(
    motif_map: dict[int, str],
    release_placement: ReleasePlacement,
) -> list[RetainedScarDomain]:
    release_requirements = _release_requirement_map(release_placement)
    domains: list[RetainedScarDomain] = []
    for coordinate in range(release_placement.retained_scar_start, release_placement.retained_scar_end):
        bases = set(_BASES)
        motif_symbol = motif_map.get(coordinate)
        if motif_symbol is not None:
            bases &= iupac_bases_for_symbol(motif_symbol)
        release_symbol = release_requirements.get(coordinate)
        if release_symbol is not None:
            bases &= iupac_bases_for_symbol(release_symbol)
        domains.append(RetainedScarDomain(raw_coordinate=coordinate, bases=sorted(bases)))
    return domains


def _feasible_scar_count(domains: list[RetainedScarDomain]) -> int:
    count = 1
    for domain in domains:
        count *= len(domain.bases)
    return count


def _upstream_flank_sequence(
    motif_map: dict[int, str],
    release_placement: ReleasePlacement,
) -> str:
    return "".join(
        symbol
        for coordinate, symbol in sorted(motif_map.items())
        if coordinate < release_placement.recognition_site_start
    )


def _type_iis_offset_sequence(
    motif_map: dict[int, str],
    release_placement: ReleasePlacement,
) -> str:
    if release_placement.recognition_site_end >= release_placement.top_cut_boundary:
        return ""
    return "".join(
        motif_map.get(coordinate, "N")
        for coordinate in range(release_placement.recognition_site_end, release_placement.top_cut_boundary)
    )


def _audit_entry_for_orientation(
    entry: NickaseCatalogEntry,
    *,
    orientation: str,
    start: int,
    target_strand: str,
    terminal_boundary: int,
    release_placement: ReleasePlacement,
    policy_rejection_reasons: list[str],
) -> NickaseGeometryAuditEntry:
    motif = display_footprint_for_orientation(entry, orientation=orientation)
    motif_map = _motif_map(motif=motif, source_site_start=start)
    downstream_symbols = _downstream_symbols(motif_map, terminal_boundary=terminal_boundary)
    release_conflicts = _release_overlap_conflicts(motif_map, release_placement)
    scar_domains = _retained_scar_domains(motif_map, release_placement)
    feasible_scar_count = _feasible_scar_count(scar_domains)
    rejection_reasons: list[str] = []
    if policy_rejection_reasons:
        rejection_reasons.append("NICKASE_POLICY_REJECTED")
    if any(not symbol.fully_degenerate for symbol in downstream_symbols):
        rejection_reasons.append("NON_DEGENERATE_DOWNSTREAM_OF_TERMINAL_NICK")
    if release_conflicts:
        rejection_reasons.append("NICKASE_RELEASE_SITE_OVERLAP_CONFLICT")
    if feasible_scar_count == 0:
        rejection_reasons.append("NO_COMPATIBLE_RETAINED_SCAR_DOMAIN")

    return NickaseGeometryAuditEntry(
        variant_id=entry.id,
        specificity_id=entry.specificity_id,
        orientation=orientation,
        motif_top_5to3=motif,
        terminal_candidate=True,
        source_site_start=start,
        source_site_end=start + len(motif),
        boundary=terminal_boundary,
        terminal_boundary=terminal_boundary,
        exact_terminal=True,
        strand=target_strand,
        policy_rejection_reasons=list(policy_rejection_reasons),
        rejection_reasons=rejection_reasons,
        downstream_symbols=downstream_symbols,
        release_overlap_conflicts=release_conflicts,
        retained_scar_domains=scar_domains,
        feasible_scar_count=feasible_scar_count,
        upstream_flank_sequence=_upstream_flank_sequence(motif_map, release_placement),
        type_iis_offset_sequence=_type_iis_offset_sequence(motif_map, release_placement),
        compatible=not rejection_reasons,
    )


def _audit_entry_without_orientation(
    entry: NickaseCatalogEntry,
    *,
    target_strand: str,
    terminal_boundary: int,
    policy_rejection_reasons: list[str],
) -> NickaseGeometryAuditEntry:
    rejection_reasons = []
    if policy_rejection_reasons:
        rejection_reasons.append("NICKASE_POLICY_REJECTED")
    rejection_reasons.append("NO_TERMINAL_NICK_ORIENTATION")
    return NickaseGeometryAuditEntry(
        variant_id=entry.id,
        specificity_id=entry.specificity_id,
        motif_top_5to3=entry.resolved_vendor_diagram_top_5to3,
        terminal_candidate=False,
        strand=target_strand,
        terminal_boundary=terminal_boundary,
        policy_rejection_reasons=list(policy_rejection_reasons),
        rejection_reasons=rejection_reasons,
        compatible=False,
    )


def build_nickase_geometry_audit(
    nickase_catalog: NickaseCatalog,
    *,
    release_placement: ReleasePlacement,
    terminal_boundary: int,
    target_strand: str,
    min_recognition_nt: int,
    disallowed_warning_codes: list[str],
) -> list[NickaseGeometryAuditEntry]:
    audit: list[NickaseGeometryAuditEntry] = []
    for entry in nickase_catalog.entries:
        policy_reasons = nickase_entry_rejection_reasons(
            entry,
            min_recognition_nt=min_recognition_nt,
            disallowed_warning_codes=disallowed_warning_codes,
        )
        target_strands = ("top", "bottom") if target_strand == "either" else (target_strand,)
        for strand in target_strands:
            required_strand = "primary" if strand == "top" else "complement"
            placements = enumerate_boundary_placements(
                entry,
                boundary=terminal_boundary,
                required_strand=required_strand,
                use_vendor_diagram=True,
            )
            if not placements:
                audit.append(
                    _audit_entry_without_orientation(
                        entry,
                        target_strand=strand,
                        terminal_boundary=terminal_boundary,
                        policy_rejection_reasons=policy_reasons,
                    )
                )
                continue
            for orientation, start in placements:
                audit.append(
                    _audit_entry_for_orientation(
                        entry,
                        orientation=orientation,
                        start=start,
                        target_strand=strand,
                        terminal_boundary=terminal_boundary,
                        release_placement=release_placement,
                        policy_rejection_reasons=policy_reasons,
                    )
                )
    return sorted(
        audit,
        key=lambda entry: (
            entry.variant_id,
            entry.strand or "",
            entry.orientation or "",
            entry.source_site_start if entry.source_site_start is not None else 0,
        ),
    )


def compatible_scar_sequences_from_audit(audit: list[NickaseGeometryAuditEntry]) -> tuple[str, ...]:
    scars: set[str] = set()
    for entry in audit:
        if not entry.compatible:
            continue
        if not entry.retained_scar_domains:
            continue
        domains = [domain.bases for domain in entry.retained_scar_domains]
        if any(not domain for domain in domains):
            continue
        for bases in product(*domains):
            scars.add("".join(bases))
    return tuple(sorted(scars))


__all__ = [
    "build_nickase_geometry_audit",
    "compatible_scar_sequences_from_audit",
    "entry_commercial_confidence",
    "entry_warning_codes",
    "iupac_symbol_is_fully_degenerate",
    "iupac_symbols_overlap",
    "nickase_entry_rejection_reasons",
    "nickase_recognition_nt",
    "placement_respects_terminal_downstream_rule",
    "placements_for_entry",
]
