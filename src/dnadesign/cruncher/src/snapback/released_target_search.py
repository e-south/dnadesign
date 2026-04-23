"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/snapback/released_target_search.py

Target-first paired nickase plus release-enzyme search for released-product
snapback designs.

Module Author(s): Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from dnadesign.cruncher.nickases.models import (
    NickaseCatalog,
    NickaseCatalogEntry,
    iupac_bases_for_symbol,
)
from dnadesign.cruncher.nickases.scanning import (
    display_motif_for_orientation,
    enumerate_boundary_placements,
    leading_fully_degenerate_prefix_nt,
)
from dnadesign.cruncher.nickases.selection import matching_nickase_warning_codes, snapback_entry_priority_key
from dnadesign.cruncher.release_enzymes.models import ReleaseEnzymeCatalog, ReleaseEnzymeEntry
from dnadesign.cruncher.release_enzymes.scanning import derive_release_cut
from dnadesign.cruncher.release_enzymes.scanning import display_motif_for_orientation as display_release_motif
from dnadesign.cruncher.release_enzymes.selection import release_entry_priority_key
from dnadesign.cruncher.snapback.released_models import (
    ReleasedFinalTargetGeometry,
    ReleasedSnapbackConstraintsSpec,
    ReleasedTargetSearchHit,
    ReleasedTargetSearchMetadata,
    ReleasedTargetSearchReport,
    SingleNickReleasedTargetSearchRequest,
    build_release_catalog_info,
    build_released_nickase_catalog_info,
)
from dnadesign.cruncher.snapback.released_projection import evaluate_released_precursor

_DNA_BASES = ("A", "C", "G", "T")
_COMPLEMENT = {"A": "T", "C": "G", "G": "C", "T": "A"}
_SNAPBACK_TIER_RANK = {
    "tier1": 0,
    "tier2": 1,
    "tier3": 2,
    None: 3,
}
_COMMERCIAL_CONFIDENCE_RANK = {
    "primary_vendor_current": 0,
    "secondary_vendor_current": 1,
    "legacy_vendor_page": 2,
    None: 3,
}


@dataclass(frozen=True)
class _NickPlacement:
    entry: NickaseCatalogEntry
    orientation: str
    motif: str
    site_start_at_boundary_zero: int
    left_of_origin_slack_nt: int = 0

    def site_start_for_boundary(self, boundary: int) -> int:
        return self.site_start_at_boundary_zero + boundary

    def site_end_for_boundary(self, boundary: int) -> int:
        return self.site_start_for_boundary(boundary) + len(self.motif)

    def earliest_allowed_boundary(self) -> int:
        return max(0, -self.site_start_at_boundary_zero - self.left_of_origin_slack_nt)

    def allows_left_of_origin_prefix(self, boundary: int) -> bool:
        return self.site_start_for_boundary(boundary) >= -self.left_of_origin_slack_nt


@dataclass(frozen=True)
class _ReleasePlacement:
    entry: ReleaseEnzymeEntry
    orientation: str
    motif: str
    active_bottom_length_offset: int
    site_shift_from_boundary: int
    top_cut_shift_from_boundary: int
    bottom_cut_shift_from_boundary: int

    def site_start_for_boundary(self, boundary: int) -> int:
        return boundary + self.site_shift_from_boundary

    def site_end_for_boundary(self, boundary: int) -> int:
        return self.site_start_for_boundary(boundary) + len(self.motif)

    def top_cut_for_boundary(self, boundary: int) -> int:
        return boundary + self.top_cut_shift_from_boundary

    def bottom_cut_for_boundary(self, boundary: int) -> int:
        return boundary + self.bottom_cut_shift_from_boundary

    def earliest_nonnegative_boundary(self) -> int:
        return max(0, -self.site_shift_from_boundary)

    def starts_downstream_of_boundary(self) -> bool:
        return self.site_shift_from_boundary >= 0


@dataclass(frozen=True)
class _BuiltPrecursor:
    top_strand: str
    coordinate_offset: int


def _blocker(counts: dict[str, int], code: str) -> None:
    counts[code] = counts.get(code, 0) + 1


def _nickase_entry_is_demo_only(entry: NickaseCatalogEntry) -> bool:
    demo_only = entry.metadata.get("demo_only")
    if isinstance(demo_only, bool):
        return demo_only
    return entry.source == "local_demo"


def _release_entry_is_demo_only(entry: ReleaseEnzymeEntry) -> bool:
    demo_only = entry.metadata.get("demo_only")
    return isinstance(demo_only, bool) and demo_only


def _nickase_entry_has_disallowed_warning_code(entry: NickaseCatalogEntry, *, warning_codes: list[str]) -> bool:
    return bool(matching_nickase_warning_codes(entry, warning_codes=warning_codes))


def _nick_placements(catalog: NickaseCatalog, *, normalize_to_top_strand_nick: bool) -> list[_NickPlacement]:
    placements: list[_NickPlacement] = []
    required_strand = "primary" if normalize_to_top_strand_nick else None
    for entry in catalog.entries:
        for orientation, site_start in enumerate_boundary_placements(
            entry,
            boundary=0,
            required_strand=required_strand,
        ):
            placements.append(
                _NickPlacement(
                    entry=entry,
                    orientation=orientation,
                    motif=display_motif_for_orientation(entry, orientation=orientation),
                    site_start_at_boundary_zero=site_start,
                    left_of_origin_slack_nt=leading_fully_degenerate_prefix_nt(entry, orientation=orientation),
                )
            )
    return sorted(
        placements,
        key=lambda placement: (
            snapback_entry_priority_key(placement.entry),
            placement.orientation,
            placement.motif,
            placement.entry.id,
        ),
    )


def _release_placements(
    catalog: ReleaseEnzymeCatalog,
    *,
    target: ReleasedFinalTargetGeometry,
) -> list[_ReleasePlacement]:
    placements: list[_ReleasePlacement] = []
    active_bottom_length_offset = (2 * target.paired_bp) + target.cap_nt
    for entry in catalog.entries:
        for orientation in ("forward", "reverse"):
            motif = display_release_motif(entry, orientation=orientation)
            cut = derive_release_cut(entry=entry, start=0, orientation=orientation)
            placements.append(
                _ReleasePlacement(
                    entry=entry,
                    orientation=orientation,
                    motif=motif,
                    active_bottom_length_offset=active_bottom_length_offset,
                    site_shift_from_boundary=active_bottom_length_offset - cut.bottom_cut_boundary,
                    top_cut_shift_from_boundary=active_bottom_length_offset
                    + (cut.top_cut_boundary - cut.bottom_cut_boundary),
                    bottom_cut_shift_from_boundary=active_bottom_length_offset,
                )
            )
    return sorted(
        placements,
        key=lambda placement: (
            release_entry_priority_key(placement.entry),
            placement.orientation,
            placement.motif,
            placement.entry.variant_id,
        ),
    )


def _apply_site_constraint(allowed: list[set[str]], *, motif: str, site_start: int) -> bool:
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


def _pair_map(*, boundary: int, target: ReleasedFinalTargetGeometry, coordinate_offset: int) -> dict[int, int]:
    mapping: dict[int, int] = {}
    input_length = coordinate_offset + boundary + target.paired_bp + target.cap_nt
    for index in range(target.paired_bp):
        left = coordinate_offset + boundary + index
        right = input_length + (target.paired_bp - 1 - index)
        mapping[left] = right
        mapping[right] = left
    return mapping


def _build_precursor_sequence(
    *,
    boundary: int,
    target: ReleasedFinalTargetGeometry,
    nick_placement: _NickPlacement,
    release_placement: _ReleasePlacement,
) -> _BuiltPrecursor | None:
    active_bottom_length = boundary + (2 * target.paired_bp) + target.cap_nt
    nick_site_start = nick_placement.site_start_for_boundary(boundary)
    release_site_start = release_placement.site_start_for_boundary(boundary)
    if not nick_placement.allows_left_of_origin_prefix(boundary) or release_site_start < 0:
        return None
    coordinate_offset = 0
    top_cut = release_placement.top_cut_for_boundary(boundary) + coordinate_offset
    bottom_cut = release_placement.bottom_cut_for_boundary(boundary) + coordinate_offset
    nick_site_start += coordinate_offset
    release_site_start += coordinate_offset
    precursor_length = max(
        coordinate_offset + active_bottom_length,
        top_cut + 1,
        bottom_cut + 1,
        nick_site_start + len(nick_placement.motif),
        release_site_start + len(release_placement.motif),
    )
    allowed = [set(_DNA_BASES) for _ in range(precursor_length)]
    if not _apply_site_constraint(allowed, motif=nick_placement.motif, site_start=nick_site_start):
        return None
    if not _apply_site_constraint(allowed, motif=release_placement.motif, site_start=release_site_start):
        return None
    pairs = _pair_map(boundary=boundary, target=target, coordinate_offset=coordinate_offset)
    assigned: list[str | None] = [None] * precursor_length
    for index in range(precursor_length):
        if assigned[index] is not None:
            continue
        partner = pairs.get(index)
        if partner is None:
            if not allowed[index]:
                return None
            assigned[index] = sorted(allowed[index])[0]
            continue
        if partner < index:
            candidate = _COMPLEMENT[str(assigned[partner])]
            if candidate not in allowed[index]:
                return None
            assigned[index] = candidate
            continue
        choices = [base for base in sorted(allowed[index]) if _COMPLEMENT[base] in allowed[partner]]
        if not choices:
            return None
        assigned[index] = choices[0]
        assigned[partner] = _COMPLEMENT[choices[0]]
    return _BuiltPrecursor(
        top_strand="".join(str(base) for base in assigned),
        coordinate_offset=coordinate_offset,
    )


def _hit_from_evaluation(
    *,
    boundary: int,
    hit_kind: str,
    precursor_top_strand: str,
    nick_placement: _NickPlacement,
    release_placement: _ReleasePlacement,
    evaluation,
) -> ReleasedTargetSearchHit | None:
    if (
        evaluation.candidate is None
        or evaluation.projection is None
        or evaluation.pre_nick_match is None
        or evaluation.release_match is None
    ):
        return None
    sacrificial_downstream_tail_nt = len(precursor_top_strand) - max(
        evaluation.release_match.cut.top_cut_boundary,
        evaluation.release_match.cut.bottom_cut_boundary,
    )
    return ReleasedTargetSearchHit(
        rank=1,
        hit_kind=hit_kind,  # type: ignore[arg-type]
        nickase_variant_id=nick_placement.entry.id,
        release_variant_id=release_placement.entry.variant_id,
        intended_nick_site_orientation=nick_placement.orientation,  # type: ignore[arg-type]
        intended_nick_site_sequence=evaluation.pre_nick_match.site.matched_span_sequence,
        release_site_orientation=release_placement.orientation,  # type: ignore[arg-type]
        release_site_sequence=evaluation.release_match.site.matched_span_sequence,
        nick_boundary_from_left=boundary,
        active_bottom_input_length_nt=evaluation.candidate.active_bottom_input_length_nt,
        active_bottom_length_nt=evaluation.candidate.active_bottom_length_nt,
        precursor_length_nt=len(precursor_top_strand),
        sacrificial_downstream_tail_nt=sacrificial_downstream_tail_nt,
        extra_nick_event_count=evaluation.candidate.extra_nick_event_count,
        extra_target_strand_nick_count=evaluation.candidate.extra_target_strand_nick_count,
        precursor_top_strand=precursor_top_strand,
        pre_nick_site=evaluation.pre_nick_match.site,
        pre_nick_event=evaluation.pre_nick_match.nick,
        release_site=evaluation.release_match.site,
        release_event=evaluation.release_match.cut,
        nickase=build_released_nickase_catalog_info(nick_placement.entry),
        release_enzyme=build_release_catalog_info(release_placement.entry),
        projection=evaluation.projection,
        final_candidate=evaluation.candidate,
    )


def _stem_cap_key(hit: ReleasedTargetSearchHit) -> tuple[str, str]:
    post_nick_sequence = hit.final_candidate.post_nick_sequence
    paired_bp = hit.final_candidate.paired_bp
    cap_nt = hit.final_candidate.cap_nt
    return (
        post_nick_sequence[:paired_bp],
        post_nick_sequence[paired_bp : paired_bp + cap_nt],
    )


def _dedupe_ranked_hits_by_stem_cap(ranked_hits: list[ReleasedTargetSearchHit]) -> list[ReleasedTargetSearchHit]:
    deduped_hits: list[ReleasedTargetSearchHit] = []
    seen_keys: set[tuple[str, str]] = set()
    for hit in ranked_hits:
        key = _stem_cap_key(hit)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        deduped_hits.append(hit)
    return deduped_hits


def _exact_hit_rank_key(hit: ReleasedTargetSearchHit) -> tuple[object, ...]:
    nick_selection = hit.nickase.selection
    nick_warning_codes = nick_selection.warning_codes if nick_selection is not None else []
    return (
        hit.extra_target_strand_nick_count,
        hit.extra_nick_event_count,
        (
            _SNAPBACK_TIER_RANK[nick_selection.snapback_tier if nick_selection is not None else None],
            (
                0
                if nick_selection is not None and nick_selection.outside_site is True
                else 1
                if nick_selection is not None
                else 2
            ),
            -(hit.nickase.motif_len or len(hit.nickase.motif_top_5to3)),
            _COMMERCIAL_CONFIDENCE_RANK[nick_selection.commercial_confidence if nick_selection is not None else None],
            len(nick_warning_codes),
            hit.nickase.variant_id,
        ),
        (
            _COMMERCIAL_CONFIDENCE_RANK[hit.release_enzyme.commercial_confidence],
            len(hit.release_enzyme.warning_codes),
            min(
                abs(hit.release_enzyme.top_cut_offset - hit.release_enzyme.recognition_len),
                abs(hit.release_enzyme.bottom_cut_offset - hit.release_enzyme.recognition_len),
            ),
            hit.release_enzyme.recognition_len,
            hit.release_enzyme.variant_id,
        ),
        hit.projection.retained_top_length_nt,
        hit.active_bottom_length_nt,
        hit.precursor_length_nt,
        hit.sacrificial_downstream_tail_nt,
        hit.precursor_top_strand,
        hit.nickase_variant_id,
        hit.release_variant_id,
    )


def _near_hit_rank_key(hit: ReleasedTargetSearchHit, *, target: ReleasedFinalTargetGeometry) -> tuple[object, ...]:
    target_input_length = target.nick_boundary_from_left + target.paired_bp + target.cap_nt
    return (
        abs(hit.nick_boundary_from_left - target.nick_boundary_from_left),
        abs(hit.active_bottom_input_length_nt - target_input_length),
        *_exact_hit_rank_key(hit),
    )


def _rank_hits(
    hits: list[ReleasedTargetSearchHit],
    *,
    target: ReleasedFinalTargetGeometry,
    exact: bool,
) -> list[ReleasedTargetSearchHit]:
    ranked = sorted(
        hits,
        key=(_exact_hit_rank_key if exact else lambda hit: _near_hit_rank_key(hit, target=target)),
    )
    deduped = _dedupe_ranked_hits_by_stem_cap(ranked)
    return [hit.model_copy(update={"rank": index}) for index, hit in enumerate(deduped, start=1)]


def _search_pair(
    *,
    request: SingleNickReleasedTargetSearchRequest,
    nick_placement: _NickPlacement,
    release_placement: _ReleasePlacement,
    blocker_counts: dict[str, int],
) -> tuple[ReleasedTargetSearchHit | None, list[ReleasedTargetSearchHit]]:
    target = request.target
    if request.search.retained_side != "upstream" or request.search.stage_order != "nick_then_release":
        raise ValueError(
            "released-product target-search only supports retained_side=upstream and stage_order=nick_then_release."
        )
    if not release_placement.starts_downstream_of_boundary():
        _blocker(blocker_counts, "RELEASE_OVERLAPS_REQUIRED_TARGET_REGION")
        return None, []
    target_boundary = target.nick_boundary_from_left
    minimum_boundary = max(
        nick_placement.earliest_allowed_boundary(),
        release_placement.earliest_nonnegative_boundary(),
    )
    boundaries = [
        (target_boundary, "exact"),
        *[
            (boundary_value, "nearest")
            for boundary_value in range(
                max(minimum_boundary, target_boundary - request.search.near_boundary_search_limit),
                target_boundary + request.search.near_boundary_search_limit + 1,
            )
            if boundary_value != target_boundary
        ],
    ]
    exact_hit: ReleasedTargetSearchHit | None = None
    near_hits: list[ReleasedTargetSearchHit] = []
    for boundary, hit_kind in boundaries:
        if not nick_placement.allows_left_of_origin_prefix(boundary):
            _blocker(blocker_counts, "PRE_NICK_SITE_LEFT_OF_ORIGIN")
            continue
        if release_placement.site_start_for_boundary(boundary) < 0:
            _blocker(blocker_counts, "RELEASE_SITE_LEFT_OF_ORIGIN")
            continue
        built_precursor = _build_precursor_sequence(
            boundary=boundary,
            target=target,
            nick_placement=nick_placement,
            release_placement=release_placement,
        )
        if built_precursor is None:
            _blocker(blocker_counts, "NO_DOWNSTREAM_RELEASE_PLACEMENT")
            continue
        precursor_top_strand = built_precursor.top_strand
        local_target = ReleasedFinalTargetGeometry(
            nick_boundary_from_left=boundary,
            paired_bp=target.paired_bp,
            cap_nt=target.cap_nt,
        )
        evaluation = evaluate_released_precursor(
            precursor_top_strand=precursor_top_strand,
            nick_entry=nick_placement.entry,
            release_entry=release_placement.entry,
            target=local_target,
            constraints=ReleasedSnapbackConstraintsSpec(
                allow_post_release_loss_of_nickase_site=request.search.allow_post_release_loss_of_nickase_site,
                allow_post_release_loss_of_release_site=True,
                require_nick_survives_in_retained_product=False,
                require_release_site_downstream_of_nick=True,
                require_complete_downstream_fragment_separation=True,
            ),
            precursor_coordinate_offset=built_precursor.coordinate_offset,
            normalize_to_top_strand_nick=True,
        )
        if evaluation.status == "satisfied":
            hit = _hit_from_evaluation(
                boundary=boundary,
                hit_kind=hit_kind,
                precursor_top_strand=precursor_top_strand,
                nick_placement=nick_placement,
                release_placement=release_placement,
                evaluation=evaluation,
            )
            if hit_kind == "exact":
                exact_hit = hit
                continue
            if hit is not None:
                near_hits.append(hit)
            continue
        for issue in evaluation.issues:
            if issue.code == "POST_RELEASE_NICK_LOST":
                _blocker(blocker_counts, "POST_RELEASE_NICK_LOST")
            elif issue.code == "RELEASE_DOES_NOT_SEPARATE_DOWNSTREAM_FRAGMENT":
                _blocker(blocker_counts, "RELEASE_DOES_NOT_SEPARATE_DOWNSTREAM_FRAGMENT")
            elif evaluation.status == "post_release_projection_failed":
                _blocker(blocker_counts, "POST_RELEASE_PROJECTION_INVALID")
            else:
                _blocker(blocker_counts, "FINAL_GEOMETRY_UNSATISFIED")
    return exact_hit, near_hits


def search_released_target_hits(
    *,
    request: SingleNickReleasedTargetSearchRequest,
    nick_catalog: NickaseCatalog,
    release_catalog: ReleaseEnzymeCatalog,
    workspace_root: Path,
    nick_catalog_source: str,
    release_catalog_source: str,
) -> ReleasedTargetSearchReport:
    blocker_counts: dict[str, int] = {}
    nick_placements = _nick_placements(
        nick_catalog,
        normalize_to_top_strand_nick=True,
    )
    if not nick_placements:
        _blocker(blocker_counts, "NO_NICKASE_PLACEMENT")
    release_placements = _release_placements(release_catalog, target=request.target)
    exact_hits: list[ReleasedTargetSearchHit] = []
    near_hits: list[ReleasedTargetSearchHit] = []
    evaluated_pair_count = 0
    for nick_placement in nick_placements:
        for release_placement in release_placements:
            if not request.search.allow_demo_hits and (
                _nickase_entry_is_demo_only(nick_placement.entry)
                or _release_entry_is_demo_only(release_placement.entry)
            ):
                _blocker(blocker_counts, "DEMO_ONLY_PAIR_SUPPRESSED")
                continue
            if _nickase_entry_has_disallowed_warning_code(
                nick_placement.entry,
                warning_codes=request.search.disallowed_nickase_warning_codes,
            ):
                _blocker(blocker_counts, "DISALLOWED_NICKASE_WARNING_CODE")
                continue
            evaluated_pair_count += 1
            exact_hit, pair_near_hits = _search_pair(
                request=request,
                nick_placement=nick_placement,
                release_placement=release_placement,
                blocker_counts=blocker_counts,
            )
            if exact_hit is not None:
                exact_hits.append(exact_hit)
            if pair_near_hits:
                near_hits.extend(pair_near_hits)
    ranked_exact_hits = _rank_hits(exact_hits, target=request.target, exact=True)
    ranked_near_hits = _rank_hits(near_hits, target=request.target, exact=False)
    pre_exact = len(ranked_exact_hits)
    pre_near = len(ranked_near_hits)
    exact_hits = ranked_exact_hits[: request.search.max_results]
    near_hits = ranked_near_hits[: request.search.max_results]
    if exact_hits:
        status = "exact_hits_found"
    elif near_hits:
        status = "near_hits_only"
    else:
        status = "no_hits"
    return ReleasedTargetSearchReport(
        status=status,
        workspace_root=str(workspace_root),
        metadata=ReleasedTargetSearchMetadata(
            target=request.target,
            nick_catalog_source=nick_catalog_source,
            release_catalog_source=release_catalog_source,
            disallowed_nickase_warning_codes=list(request.search.disallowed_nickase_warning_codes),
            evaluated_pair_count=evaluated_pair_count,
            pre_truncation_exact_hit_count=pre_exact,
            post_truncation_exact_hit_count=len(exact_hits),
            pre_truncation_near_hit_count=pre_near,
            post_truncation_near_hit_count=len(near_hits),
            blocker_counts=blocker_counts,
        ),
        exact_hits=exact_hits,
        near_hits=near_hits,
    )


__all__ = ["search_released_target_hits"]
