"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cruncher/src/snapback/preserved_search/candidate_builder.py

Candidate construction helpers for preserved-site target search.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from functools import lru_cache
from itertools import product

from dnadesign.cruncher.nickases.models import NickaseCatalogEntry, iupac_bases_for_symbol, reverse_complement
from dnadesign.cruncher.nickases.scanning import (
    build_evaluated_match,
    enumerate_site_instances,
    enumerate_site_instances_starting_at_or_after,
    suffix_sensitive_scan_start,
)
from dnadesign.cruncher.nickases.selection import snapback_entry_priority_key
from dnadesign.cruncher.snapback.models import CoordinateSpan, build_catalog_info
from dnadesign.cruncher.snapback.planner import evaluate_snapback_candidate
from dnadesign.cruncher.snapback.preserved_search.placement_models import Placement
from dnadesign.cruncher.snapback.target_models import (
    SnapbackTargetGeometry,
    SnapbackTargetSearchHit,
)


@lru_cache(maxsize=None)
def lexical_dna_sequence_pool(length: int) -> tuple[str, ...]:
    if length == 0:
        return ("",)
    return tuple("".join(bases) for bases in product("ACGT", repeat=length))


@lru_cache(maxsize=None)
def exact_site_sequence_pool(oriented_motif: str) -> tuple[str, ...]:
    choices = [tuple(sorted(iupac_bases_for_symbol(symbol))) for symbol in oriented_motif]
    return tuple("".join(bases) for bases in product(*choices))


def build_candidate_for_input(
    *,
    placement: Placement,
    input_sequence: str,
    site_start: int,
    site_sequence: str,
    boundary: int,
    target: SnapbackTargetGeometry,
    cap_sequence: str,
):
    retained_homology_window = CoordinateSpan(start=boundary, end=boundary + target.paired_bp)
    retained_homology_sequence = input_sequence[retained_homology_window.start : retained_homology_window.end]
    foldback_arm = reverse_complement(retained_homology_sequence)
    intended_match = build_evaluated_match(
        entry=placement.entry,
        start=site_start,
        orientation=placement.orientation,
        coordinate_offset=0,
        matched_span_sequence=site_sequence,
    )
    invariant_prefix_matches = enumerate_site_instances(
        input_sequence,
        coordinate_offset=0,
        entry=placement.entry,
    )
    suffix_sensitive_matches = enumerate_site_instances_starting_at_or_after(
        f"{input_sequence}{cap_sequence}{foldback_arm}",
        coordinate_offset=0,
        entry=placement.entry,
        start_min=suffix_sensitive_scan_start(placement.entry, prefix_length=len(input_sequence)),
    )
    candidate, issues = evaluate_snapback_candidate(
        input_sequence=input_sequence,
        protected_region=CoordinateSpan(start=site_start, end=site_start + len(site_sequence)),
        pre_nick_duplex_window=CoordinateSpan(start=0, end=len(input_sequence)),
        retained_homology_window=retained_homology_window,
        cap_sequence=cap_sequence,
        foldback_arm=foldback_arm,
        homology_max_mismatches=0,
        terminal_ligatable_duplex_min=target.paired_bp,
        terminal_ligatable_duplex_max=target.paired_bp,
        max_uninterrupted_duplex_bp=target.paired_bp,
        max_added_nt=len(cap_sequence) + len(foldback_arm),
        gc_bounds=None,
        max_homopolymer_run_allowed=None,
        intended_match=intended_match,
        site_mutation_count=0,
        all_matches=[*invariant_prefix_matches, *suffix_sensitive_matches],
        forbid_additional_target_strand_nicks=False,
        forbid_any_additional_nicks=False,
    )
    return candidate, issues


def input_sequence_priority(candidate, *, entry: NickaseCatalogEntry) -> tuple[object, ...]:
    return (
        len(candidate.extra_target_strand_nicks),
        len(candidate.extra_nick_events),
        round(candidate.gc_fraction_added, 6),
        candidate.max_homopolymer_run_added,
        snapback_entry_priority_key(entry),
        candidate.cap_sequence,
        candidate.intended_site.matched_span_sequence,
        candidate.input_sequence,
    )


def best_hit_for_boundary(
    *,
    placement: Placement,
    boundary: int,
    input_length_nt: int,
    target: SnapbackTargetGeometry,
    hit_kind: str,
) -> SnapbackTargetSearchHit | None:
    site_start = placement.site_start_for_boundary(boundary)
    site_end = site_start + len(placement.motif)
    prefix_length = site_start
    suffix_length = input_length_nt - site_end
    source_cap_nt = input_length_nt - (boundary + target.paired_bp)
    cap_extension_nt = target.cap_nt - source_cap_nt
    if prefix_length < 0 or suffix_length < 0 or cap_extension_nt < 0:
        return None
    best_candidate = None
    for prefix in lexical_dna_sequence_pool(prefix_length):
        for site_sequence in exact_site_sequence_pool(placement.motif):
            for suffix in lexical_dna_sequence_pool(suffix_length):
                input_sequence = f"{prefix}{site_sequence}{suffix}"
                for cap_sequence in lexical_dna_sequence_pool(cap_extension_nt):
                    candidate, issues = build_candidate_for_input(
                        placement=placement,
                        input_sequence=input_sequence,
                        site_start=site_start,
                        site_sequence=site_sequence,
                        boundary=boundary,
                        target=target,
                        cap_sequence=cap_sequence,
                    )
                    if issues or candidate is None:
                        continue
                    if best_candidate is None or input_sequence_priority(
                        candidate,
                        entry=placement.entry,
                    ) < input_sequence_priority(best_candidate, entry=placement.entry):
                        best_candidate = candidate
    if best_candidate is None:
        return None
    return SnapbackTargetSearchHit(
        rank=1,
        hit_kind=hit_kind,
        variant_id=placement.entry.id,
        intended_site_orientation=placement.orientation,
        intended_site_sequence=best_candidate.intended_site.matched_span_sequence,
        nick_boundary_from_left=best_candidate.nick_boundary_from_left,
        site_start=best_candidate.intended_site.start,
        site_end=best_candidate.intended_site.end,
        input_sequence=best_candidate.input_sequence,
        designed_sequence=best_candidate.designed_sequence,
        input_length_nt=len(best_candidate.input_sequence),
        designed_length_nt=len(best_candidate.designed_sequence),
        paired_bp=best_candidate.paired_bp,
        cap_nt=best_candidate.cap_nt,
        source_cap_nt=len(best_candidate.source_cap_sequence),
        cap_extension_nt=best_candidate.cap_extension_nt,
        site_mutation_count=best_candidate.site_mutation_count,
        extra_nick_event_count=len(best_candidate.extra_nick_events),
        extra_target_strand_nick_count=len(best_candidate.extra_target_strand_nicks),
        nickase=build_catalog_info(placement.entry),
        explicit_report=best_candidate,
    )
