"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/snapback/solve_search.py

Frontier enumeration and co-design input ordering for v3 snapback solve.

Module Author(s): Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

from dnadesign.cruncher.nickases.models import NickaseCatalogEntry, iupac_bases_for_symbol
from dnadesign.cruncher.nickases.scanning import (
    EvaluatedMatch,
    build_evaluated_match,
    display_motif_for_orientation,
    enumerate_boundary_placements,
)
from dnadesign.cruncher.nickases.selection import snapback_entry_priority_key
from dnadesign.cruncher.snapback.models import EFFECTIVE_CAP_LOOP_NT, CoordinateSpan
from dnadesign.cruncher.snapback.solve_models import SingleNickSnapbackSolveSpec


@dataclass(frozen=True)
class SnapbackSearchFrontier:
    nick_boundary: int
    paired_bp: int
    cap_extension_nt: int

    @property
    def retained_homology_window(self) -> CoordinateSpan:
        return CoordinateSpan(start=self.nick_boundary, end=self.nick_boundary + self.paired_bp)

    def key(self) -> tuple[int, int, int]:
        return (self.nick_boundary, self.paired_bp, self.cap_extension_nt)


@dataclass(frozen=True)
class SnapbackCodesignedInput:
    entry: NickaseCatalogEntry
    input_sequence: str
    intended_match: EvaluatedMatch
    site_mutation_count: int


def build_ordered_search_frontiers(spec: SingleNickSnapbackSolveSpec) -> list[SnapbackSearchFrontier]:
    input_length = len(spec.input.canonical_top_strand.sequence)
    resolved_search_space = spec.resolved_search_space()
    frontiers: list[SnapbackSearchFrontier] = []
    for boundary in range(
        resolved_search_space.nick_boundary_window.min,
        resolved_search_space.nick_boundary_window.max + 1,
    ):
        min_retained_length = max(
            resolved_search_space.retained_homology_length.min,
            input_length - boundary - EFFECTIVE_CAP_LOOP_NT,
        )
        max_retained_length = min(
            resolved_search_space.retained_homology_length.max,
            input_length - boundary,
        )
        if min_retained_length > max_retained_length:
            continue
        for retained_length in range(min_retained_length, max_retained_length + 1):
            source_cap_nt = input_length - (boundary + retained_length)
            cap_extension_nt = EFFECTIVE_CAP_LOOP_NT - source_cap_nt
            if cap_extension_nt < 0:
                continue
            frontiers.append(
                SnapbackSearchFrontier(
                    nick_boundary=boundary,
                    paired_bp=retained_length,
                    cap_extension_nt=cap_extension_nt,
                )
            )
    return frontiers


def _enumerate_codesigned_inputs_for_boundary(
    template_sequence: str,
    *,
    entry: NickaseCatalogEntry,
    boundary: int,
    duplex_window: CoordinateSpan,
    normalize_to_top_strand_nick: bool,
) -> list[SnapbackCodesignedInput]:
    required_strand = "primary" if normalize_to_top_strand_nick else None
    candidates: list[SnapbackCodesignedInput] = []
    for orientation, site_start in enumerate_boundary_placements(
        entry,
        boundary=boundary,
        required_strand=required_strand,
    ):
        oriented_motif = display_motif_for_orientation(entry, orientation=orientation)
        site_end = site_start + len(oriented_motif)
        if site_start < duplex_window.start or site_end > duplex_window.end:
            continue
        if site_start < 0 or site_end > len(template_sequence):
            continue
        template_window = template_sequence[site_start:site_end]
        choices: list[list[str]] = []
        for template_base, motif_symbol in zip(template_window, oriented_motif, strict=True):
            allowed = sorted(iupac_bases_for_symbol(motif_symbol))
            if template_base in allowed:
                ordered = [template_base, *[base for base in allowed if base != template_base]]
            else:
                ordered = allowed
            choices.append(ordered)
        if not choices:
            continue
        for bases in product(*choices):
            site_sequence = "".join(bases)
            site_mutation_count = sum(
                1
                for designed_base, template_base in zip(site_sequence, template_window, strict=True)
                if designed_base != template_base
            )
            candidate_input_sequence = f"{template_sequence[:site_start]}{site_sequence}{template_sequence[site_end:]}"
            candidates.append(
                SnapbackCodesignedInput(
                    entry=entry,
                    input_sequence=candidate_input_sequence,
                    intended_match=build_evaluated_match(
                        entry=entry,
                        start=site_start,
                        orientation=orientation,
                        coordinate_offset=0,
                        matched_span_sequence=site_sequence,
                    ),
                    site_mutation_count=site_mutation_count,
                )
            )
    return candidates


def _codesigned_input_priority(candidate: SnapbackCodesignedInput) -> tuple[object, ...]:
    return (
        candidate.site_mutation_count,
        snapback_entry_priority_key(candidate.entry),
        candidate.intended_match.site.start,
        candidate.intended_match.site.orientation,
        candidate.intended_match.site.matched_span_sequence,
        candidate.input_sequence,
        candidate.entry.id,
    )


def enumerate_frontier_codesigned_inputs(
    template_sequence: str,
    *,
    frontier: SnapbackSearchFrontier,
    catalog_entries: list[NickaseCatalogEntry],
    duplex_window: CoordinateSpan,
    normalize_to_top_strand_nick: bool,
) -> list[SnapbackCodesignedInput]:
    candidates: list[SnapbackCodesignedInput] = []
    for entry in sorted(catalog_entries, key=snapback_entry_priority_key):
        candidates.extend(
            _enumerate_codesigned_inputs_for_boundary(
                template_sequence,
                entry=entry,
                boundary=frontier.nick_boundary,
                duplex_window=duplex_window,
                normalize_to_top_strand_nick=normalize_to_top_strand_nick,
            )
        )
    return sorted(candidates, key=_codesigned_input_priority)


__all__ = [
    "SnapbackCodesignedInput",
    "SnapbackSearchFrontier",
    "build_ordered_search_frontiers",
    "enumerate_frontier_codesigned_inputs",
]
