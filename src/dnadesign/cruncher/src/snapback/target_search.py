"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/snapback/target_search.py

Target-first snapback catalog search for exact preserved-site geometry hits.

Module Author(s): Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import product
from pathlib import Path

from dnadesign.cruncher.nickases.models import (
    NickaseCatalog,
    NickaseCatalogEntry,
    iupac_bases_for_symbol,
    reverse_complement,
)
from dnadesign.cruncher.nickases.scanning import (
    build_evaluated_match,
    display_motif_for_orientation,
    enumerate_boundary_placements,
    enumerate_site_instances,
    enumerate_site_instances_starting_at_or_after,
    suffix_sensitive_scan_start,
)
from dnadesign.cruncher.nickases.selection import snapback_entry_priority_key
from dnadesign.cruncher.snapback.models import CoordinateSpan, build_catalog_info
from dnadesign.cruncher.snapback.planner import evaluate_snapback_candidate
from dnadesign.cruncher.snapback.target_models import (
    SnapbackTargetFeasibilityRow,
    SnapbackTargetGeometry,
    SnapbackTargetSearchHit,
    SnapbackTargetSearchMetadata,
    SnapbackTargetSearchReport,
)


def _catalog_source_label(*, preset_ids: list[str], additional_paths: list[Path]) -> str:
    labels: list[str] = []
    labels.extend(f"preset:{preset_id}" for preset_id in preset_ids)
    labels.extend(str(path) for path in additional_paths)
    return ", ".join(labels) if labels else "resolved_catalog"


@lru_cache(maxsize=None)
def _lexical_dna_sequence_pool(length: int) -> tuple[str, ...]:
    if length == 0:
        return ("",)
    return tuple("".join(bases) for bases in product("ACGT", repeat=length))


@lru_cache(maxsize=None)
def _exact_site_sequence_pool(oriented_motif: str) -> tuple[str, ...]:
    choices = [tuple(sorted(iupac_bases_for_symbol(symbol))) for symbol in oriented_motif]
    return tuple("".join(bases) for bases in product(*choices))


@dataclass(frozen=True)
class _Placement:
    entry: NickaseCatalogEntry
    orientation: str
    motif: str
    site_start_at_target_boundary: int
    boundary_offset: int
    exact_input_length_nt: int | None
    earliest_feasible_boundary: int | None
    earliest_input_length_nt: int | None
    exact_boundary_blockers: tuple[str, ...]

    @property
    def exact_boundary_hit_possible(self) -> bool:
        return self.exact_input_length_nt is not None

    @property
    def any_boundary_hit_possible(self) -> bool:
        return self.earliest_feasible_boundary is not None

    def site_start_for_boundary(self, boundary: int) -> int:
        return boundary - self.boundary_offset


def _placement_rank_key(placement: _Placement) -> tuple[object, ...]:
    outside_site = placement.entry.selection.outside_site if placement.entry.selection is not None else None
    outside_rank = 0 if outside_site is True else 1 if outside_site is False else 2
    earliest_boundary = (
        placement.earliest_feasible_boundary if placement.earliest_feasible_boundary is not None else 10**9
    )
    return (
        0 if placement.exact_boundary_hit_possible else 1,
        earliest_boundary,
        outside_rank,
        snapback_entry_priority_key(placement.entry),
        placement.orientation,
        placement.motif,
        placement.entry.id,
    )


def _build_placement(
    *,
    entry: NickaseCatalogEntry,
    orientation: str,
    site_start_at_target_boundary: int,
    target: SnapbackTargetGeometry,
) -> _Placement:
    motif = display_motif_for_orientation(entry, orientation=orientation)
    boundary_offset = target.nick_boundary_from_left - site_start_at_target_boundary
    site_end_at_target_boundary = site_start_at_target_boundary + len(motif)
    max_input_length_at_target = target.nick_boundary_from_left + target.paired_bp + target.cap_nt
    exact_input_length_nt: int | None = None
    exact_boundary_blockers: list[str] = []
    if site_start_at_target_boundary < 0:
        exact_boundary_blockers.append("NEGATIVE_SITE_START_AT_TARGET_BOUNDARY")
    if site_end_at_target_boundary > max_input_length_at_target:
        exact_boundary_blockers.append("SITE_EXCEEDS_MAX_INPUT_AT_TARGET_BOUNDARY")
    if not exact_boundary_blockers:
        exact_input_length_nt = max(target.nick_boundary_from_left + target.paired_bp, site_end_at_target_boundary)

    boundary_invariant_site_extent = len(motif) - boundary_offset
    earliest_feasible_boundary: int | None = None
    earliest_input_length_nt: int | None = None
    if boundary_invariant_site_extent <= target.paired_bp + target.cap_nt:
        earliest_feasible_boundary = max(0, boundary_offset)
        site_start = earliest_feasible_boundary - boundary_offset
        site_end = site_start + len(motif)
        earliest_input_length_nt = max(earliest_feasible_boundary + target.paired_bp, site_end)

    return _Placement(
        entry=entry,
        orientation=orientation,
        motif=motif,
        site_start_at_target_boundary=site_start_at_target_boundary,
        boundary_offset=boundary_offset,
        exact_input_length_nt=exact_input_length_nt,
        earliest_feasible_boundary=earliest_feasible_boundary,
        earliest_input_length_nt=earliest_input_length_nt,
        exact_boundary_blockers=tuple(exact_boundary_blockers),
    )


def _iter_target_strand_placements(
    *,
    catalog_entries: list[NickaseCatalogEntry],
    target: SnapbackTargetGeometry,
    normalize_to_top_strand_nick: bool,
) -> list[_Placement]:
    required_strand = "primary" if normalize_to_top_strand_nick else None
    placements: list[_Placement] = []
    for entry in catalog_entries:
        for orientation, site_start_at_target_boundary in enumerate_boundary_placements(
            entry,
            boundary=target.nick_boundary_from_left,
            required_strand=required_strand,
        ):
            placements.append(
                _build_placement(
                    entry=entry,
                    orientation=orientation,
                    site_start_at_target_boundary=site_start_at_target_boundary,
                    target=target,
                )
            )
    return sorted(placements, key=_placement_rank_key)


_SNAPBACK_TIER_RANK = {
    "tier1": 0,
    "tier2": 1,
    "tier3": 2,
    None: 3,
}
_COMMERCIAL_CONFIDENCE_RANK = {
    "primary_vendor_current": 0,
    "secondary_vendor_current": 1,
    "produced_on_demand": 2,
    "literature_only": 3,
    None: 4,
}


def _catalog_info_priority_key(hit: SnapbackTargetSearchHit) -> tuple[object, ...]:
    selection = hit.nickase.selection
    warning_codes = selection.warning_codes if selection is not None else []
    return (
        _SNAPBACK_TIER_RANK[selection.snapback_tier if selection is not None else None],
        0 if selection is not None and selection.outside_site is True else 1 if selection is not None else 2,
        -(hit.nickase.motif_len or len(hit.nickase.motif_top_5to3)),
        _COMMERCIAL_CONFIDENCE_RANK[selection.commercial_confidence if selection is not None else None],
        len(warning_codes),
        hit.variant_id,
    )


def _build_feasibility_row(placement: _Placement) -> SnapbackTargetFeasibilityRow:
    return SnapbackTargetFeasibilityRow(
        variant_id=placement.entry.id,
        orientation=placement.orientation,
        motif_top_5to3=placement.motif,
        motif_len=len(placement.motif),
        site_start_at_target_boundary=placement.site_start_at_target_boundary,
        site_end_at_target_boundary=placement.site_start_at_target_boundary + len(placement.motif),
        boundary_offset=placement.boundary_offset,
        outside_site=placement.entry.outside_site,
        exact_boundary_hit_possible=placement.exact_boundary_hit_possible,
        exact_boundary_blockers=list(placement.exact_boundary_blockers),
        any_boundary_hit_possible=placement.any_boundary_hit_possible,
        earliest_feasible_boundary=placement.earliest_feasible_boundary,
        exact_input_length_nt=placement.exact_input_length_nt,
        earliest_input_length_nt=placement.earliest_input_length_nt,
    )


def _build_candidate_for_input(
    *,
    placement: _Placement,
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


def _input_sequence_priority(candidate, *, entry: NickaseCatalogEntry) -> tuple[object, ...]:
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


def _best_hit_for_boundary(
    *,
    placement: _Placement,
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
    for prefix in _lexical_dna_sequence_pool(prefix_length):
        for site_sequence in _exact_site_sequence_pool(placement.motif):
            for suffix in _lexical_dna_sequence_pool(suffix_length):
                input_sequence = f"{prefix}{site_sequence}{suffix}"
                for cap_sequence in _lexical_dna_sequence_pool(cap_extension_nt):
                    candidate, issues = _build_candidate_for_input(
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
                    if best_candidate is None or _input_sequence_priority(
                        candidate,
                        entry=placement.entry,
                    ) < _input_sequence_priority(best_candidate, entry=placement.entry):
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


def _exact_hit_rank_key(hit: SnapbackTargetSearchHit) -> tuple[object, ...]:
    outside_site = hit.nickase.selection.outside_site if hit.nickase.selection is not None else None
    outside_rank = 0 if outside_site is True else 1 if outside_site is False else 2
    return (
        hit.extra_target_strand_nick_count,
        hit.extra_nick_event_count,
        hit.input_length_nt,
        outside_rank,
        _catalog_info_priority_key(hit),
        hit.intended_site_sequence,
        hit.input_sequence,
        hit.variant_id,
    )


def _near_hit_rank_key(hit: SnapbackTargetSearchHit, *, target: SnapbackTargetGeometry) -> tuple[object, ...]:
    exact_key = _exact_hit_rank_key(hit)
    return (
        abs(hit.nick_boundary_from_left - target.nick_boundary_from_left),
        hit.nick_boundary_from_left,
        *exact_key,
    )


def _rank_hits(
    hits: list[SnapbackTargetSearchHit],
    *,
    target: SnapbackTargetGeometry,
    exact: bool,
) -> list[SnapbackTargetSearchHit]:
    ordered = sorted(
        hits,
        key=(_exact_hit_rank_key if exact else lambda hit: _near_hit_rank_key(hit, target=target)),
    )
    return [hit.model_copy(update={"rank": index}) for index, hit in enumerate(ordered, start=1)]


def search_snapback_target_hits(
    *,
    catalog: NickaseCatalog,
    target: SnapbackTargetGeometry,
    workspace_root: Path,
    catalog_preset: str | None,
    catalog_presets: list[str],
    catalog_additional_paths: list[Path],
    normalize_to_top_strand_nick: bool = True,
    max_results: int = 8,
) -> SnapbackTargetSearchReport:
    placements = _iter_target_strand_placements(
        catalog_entries=catalog.entries,
        target=target,
        normalize_to_top_strand_nick=normalize_to_top_strand_nick,
    )
    feasibility = [_build_feasibility_row(placement) for placement in placements]

    exact_hits: list[SnapbackTargetSearchHit] = []
    near_hits: list[SnapbackTargetSearchHit] = []
    for placement in placements:
        if placement.exact_boundary_hit_possible and placement.exact_input_length_nt is not None:
            hit = _best_hit_for_boundary(
                placement=placement,
                boundary=target.nick_boundary_from_left,
                input_length_nt=placement.exact_input_length_nt,
                target=target,
                hit_kind="exact",
            )
            if hit is not None:
                exact_hits.append(hit)
        if placement.any_boundary_hit_possible and placement.earliest_feasible_boundary is not None:
            if (
                placement.exact_boundary_hit_possible
                and placement.earliest_feasible_boundary == target.nick_boundary_from_left
            ):
                continue
            hit = _best_hit_for_boundary(
                placement=placement,
                boundary=placement.earliest_feasible_boundary,
                input_length_nt=int(placement.earliest_input_length_nt),
                target=target,
                hit_kind="nearest",
            )
            if hit is not None:
                near_hits.append(hit)

    exact_hits = _rank_hits(exact_hits, target=target, exact=True)[:max_results]
    near_hits = _rank_hits(near_hits, target=target, exact=False)[:max_results]
    if exact_hits:
        status = "exact_hits_found"
    elif near_hits:
        status = "near_hits_only"
    else:
        status = "no_hits"
    return SnapbackTargetSearchReport(
        status=status,
        workspace_root=str(workspace_root),
        metadata=SnapbackTargetSearchMetadata(
            catalog_preset=catalog_preset,
            catalog_presets=catalog_presets,
            catalog_additional_paths=[str(path) for path in catalog_additional_paths],
            catalog_source=_catalog_source_label(
                preset_ids=catalog_presets,
                additional_paths=catalog_additional_paths,
            ),
            target=target,
            evaluated_orientation_count=len(feasibility),
            exact_hit_count=len(exact_hits),
            near_hit_count=len(near_hits),
        ),
        issues=[],
        exact_hits=exact_hits,
        near_hits=near_hits,
        feasibility=feasibility,
    )


def render_target_search_markdown_report(report: SnapbackTargetSearchReport) -> str:
    lines = [
        "# Snapback Target Search Report",
        "",
        f"- status: {report.status}",
        f"- catalog_source: {report.metadata.catalog_source}",
        f"- target_boundary: {report.metadata.target.nick_boundary_from_left}",
        f"- target_paired_bp: {report.metadata.target.paired_bp}",
        f"- target_cap_nt: {report.metadata.target.cap_nt}",
        f"- exact_hit_count: {report.metadata.exact_hit_count}",
        f"- near_hit_count: {report.metadata.near_hit_count}",
        f"- evaluated_orientation_count: {report.metadata.evaluated_orientation_count}",
    ]
    if report.exact_hits:
        lines.extend(["", "## Exact Hits"])
        for hit in report.exact_hits:
            outside_site = hit.nickase.selection.outside_site if hit.nickase.selection is not None else "unknown"
            lines.append(
                f"- rank {hit.rank}: {hit.variant_id} boundary={hit.nick_boundary_from_left} "
                f"site={hit.intended_site_orientation}:{hit.intended_site_sequence} "
                f"input_nt={hit.input_length_nt} extra_target_nicks={hit.extra_target_strand_nick_count} "
                f"extra_nicks={hit.extra_nick_event_count} outside_site={outside_site}"
            )
    if report.near_hits:
        lines.extend(["", "## Near Hits"])
        for hit in report.near_hits:
            outside_site = hit.nickase.selection.outside_site if hit.nickase.selection is not None else "unknown"
            lines.append(
                f"- rank {hit.rank}: {hit.variant_id} boundary={hit.nick_boundary_from_left} "
                f"site={hit.intended_site_orientation}:{hit.intended_site_sequence} "
                f"input_nt={hit.input_length_nt} extra_target_nicks={hit.extra_target_strand_nick_count} "
                f"extra_nicks={hit.extra_nick_event_count} outside_site={outside_site}"
            )
    if report.feasibility:
        lines.extend(["", "## Feasibility"])
        for row in report.feasibility:
            blockers = ",".join(row.exact_boundary_blockers) if row.exact_boundary_blockers else "-"
            lines.append(
                f"- {row.variant_id} {row.orientation} exact={row.exact_boundary_hit_possible} "
                f"target_site_start={row.site_start_at_target_boundary} earliest_boundary="
                f"{row.earliest_feasible_boundary if row.earliest_feasible_boundary is not None else '-'} "
                f"blockers={blockers}"
            )
    return "\n".join(lines).strip() + "\n"
