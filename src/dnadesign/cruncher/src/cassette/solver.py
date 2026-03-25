"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/cassette/solver.py

Deterministic solve/search layer for hairpin dual-nick cassette design.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

from dnadesign.cruncher.cassette.models import (
    CassetteCatalogRef,
    CassetteOutputConfig,
    CatalogNormalizationInfo,
    ConstructContextSpec,
    DuplexNickingPlanSpec,
    FlankNickingRequest,
    HairpinCassetteSpec,
    HairpinTopologySpec,
    HairpinValidationSpec,
    NickaseCatalog,
    NickaseCatalogEntry,
    NickWindow,
    SitePolicySpec,
    ValidationIssue,
    iupac_bases_for_symbol,
    motif_matches,
    reverse_complement,
    reverse_complement_iupac,
)
from dnadesign.cruncher.cassette.planner import build_cassette_report
from dnadesign.cruncher.cassette.scanning import display_motif_for_orientation
from dnadesign.cruncher.cassette.selection import (
    CandidateHitRecord,
    SelectedCandidate,
    build_accepted_candidate_pool,
    select_hits,
)
from dnadesign.cruncher.cassette.solve_filters import (
    expanded_blacklist_motifs,
    extra_site_count,
    gc_fraction,
    max_homopolymer_run,
    sequence_blacklist_issues,
    sequence_quality_issues,
    site_blacklist_issues,
)
from dnadesign.cruncher.cassette.solve_models import (
    CandidateHit,
    CandidateScoreBreakdown,
    HairpinCassetteSolveSpec,
    SolveReport,
    SolveReportMetadata,
    SolveSelectionSummary,
)
from dnadesign.cruncher.utils.hashing import sha256_bytes

_BASE_ORDER = ("A", "C", "G", "T")
_COMPLEMENT = {"A": "T", "C": "G", "G": "C", "T": "A"}
_WARNING_MESSAGES = {
    "MAX_SEARCH_NODES_REACHED": "search.max_search_nodes reached before exhausting the solve search tree.",
    "MAX_ENUMERATED_CANDIDATES_REACHED": (
        "search.max_enumerated_candidates reached before exhausting the solve search space."
    ),
    "ACCEPTED_POOL_TRUNCATED": "accepted_pool truncated to keep solve memory bounded.",
    "SELECTION_RESULTS_POOL_BOUNDED": (
        "Selected hits are best only among the bounded accepted pool retained under the configured solve caps."
    ),
    "SELECTION_RESULTS_SEARCH_BOUNDED": (
        "Selected hits are search-bounded under the configured solve caps and are not guaranteed globally optimal."
    ),
    "SELECTION_POLICY_LIMITED_HITS": (
        "Selection policy constraints returned fewer hits than the accepted pool could otherwise provide."
    ),
}


@dataclass(frozen=True)
class VariantPairChoice:
    left: NickaseCatalogEntry
    right: NickaseCatalogEntry


@dataclass(frozen=True)
class PlacementChoice:
    variant: NickaseCatalogEntry
    start: int
    end: int
    orientation: Literal["forward", "reverse"]
    display_motif: str
    nick_boundary: int
    nick_strand: Literal["primary", "complement"]

    @property
    def key(self) -> tuple[str, int, int, str]:
        return (self.variant.id, self.start, self.end, self.orientation)


@dataclass(frozen=True)
class ConcreteCandidate:
    stem5p_arm: str
    loop: str
    stem3p_arm: str
    cassette_sequence: str
    evaluation_primary_sequence: str


@dataclass
class SolveSearchResult:
    status: Literal["solved", "no_hits", "invalid_spec"]
    issues: list[ValidationIssue]
    warnings: list[str]
    warning_codes: list[str]
    hits: list[SelectedCandidate]
    selection_summary: SolveSelectionSummary | None
    enumerated_candidate_count: int
    accepted_candidate_count: int
    considered_variant_pair_count: int
    visited_search_node_count: int


def _catalog_variants_for_report(catalog: NickaseCatalog) -> list[CatalogNormalizationInfo]:
    return [
        CatalogNormalizationInfo(
            variant_id=entry.id,
            specificity_id=entry.specificity_id,
            motif_top_5to3=entry.motif_top_5to3,
            motif_len=entry.motif_len or len(entry.motif_top_5to3),
            top_cut_offset=entry.top_cut_offset,
            bottom_cut_offset=entry.bottom_cut_offset,
            source=entry.source,
            raw_cut_notation=entry.raw_cut_notation,
            metadata=entry.metadata,
        )
        for entry in sorted(catalog.entries, key=lambda item: item.id)
    ]


def _score_hit(
    *,
    cassette_sequence: str,
    bounded_segment_length: int,
    extra_site_count: int,
    gc_fraction: float,
    homopolymer_run: int,
    solve_spec: HairpinCassetteSolveSpec,
) -> tuple[CandidateScoreBreakdown, tuple[float | int | str, ...]]:
    bounded_distance = (
        abs(bounded_segment_length - solve_spec.search.bounded_segment_target)
        if solve_spec.search.bounded_segment_target is not None
        else 0.0
    )
    gc_distance = abs(gc_fraction - solve_spec.search.gc_target) if solve_spec.search.gc_target is not None else 0.0
    homopolymer_penalty = homopolymer_run
    breakdown = CandidateScoreBreakdown(
        extra_site_count=extra_site_count,
        bounded_segment_distance=float(bounded_distance),
        gc_distance=float(gc_distance),
        homopolymer_penalty=homopolymer_penalty,
    )
    return breakdown, (
        extra_site_count,
        float(bounded_distance),
        float(gc_distance),
        homopolymer_penalty,
        cassette_sequence,
    )


def _min_max_gc_from_domains(domains: list[set[str]]) -> tuple[int, int]:
    minimum = 0
    maximum = 0
    for domain in domains:
        gc_values = {1 if base in {"G", "C"} else 0 for base in domain}
        minimum += min(gc_values)
        maximum += max(gc_values)
    return minimum, maximum


def _build_initial_domains(pattern: str) -> list[set[str]]:
    return [iupac_bases_for_symbol(symbol) for symbol in pattern]


def _placement_window(
    side: Literal["left", "right"],
    *,
    solve_spec: HairpinCassetteSolveSpec,
) -> NickWindow:
    return solve_spec.nick_goal.left_nick_window if side == "left" else solve_spec.nick_goal.right_nick_window


def _enumerate_placements_for_side(
    *,
    side: Literal["left", "right"],
    variant: NickaseCatalogEntry,
    solve_spec: HairpinCassetteSolveSpec,
) -> list[PlacementChoice]:
    motif_len = variant.motif_len or len(variant.motif_top_5to3)
    if motif_len > solve_spec.stem_length_nt:
        return []
    cassette_length = solve_spec.cassette_length_nt
    stem_length = solve_spec.stem_length_nt
    loop_length = solve_spec.loop_length_nt
    window = _placement_window(side, solve_spec=solve_spec)
    if side == "left":
        starts = range(0, stem_length - motif_len + 1)
    else:
        right_start = stem_length + loop_length
        starts = range(right_start, right_start + stem_length - motif_len + 1)

    placements: list[PlacementChoice] = []
    orientations = (
        ("forward",)
        if reverse_complement_iupac(variant.motif_top_5to3) == variant.motif_top_5to3
        else (
            "forward",
            "reverse",
        )
    )
    for start in starts:
        for orientation in orientations:
            if orientation == "forward":
                if variant.top_cut_offset is not None:
                    nick_strand = "primary"
                    nick_boundary = start + variant.top_cut_offset
                else:
                    nick_strand = "complement"
                    nick_boundary = start + int(variant.bottom_cut_offset)
            else:
                if variant.top_cut_offset is not None:
                    nick_strand = "complement"
                    nick_boundary = start + (motif_len - variant.top_cut_offset)
                else:
                    nick_strand = "primary"
                    nick_boundary = start + (motif_len - int(variant.bottom_cut_offset))
            if nick_boundary < 0 or nick_boundary > cassette_length:
                continue
            if nick_strand != solve_spec.nick_goal.target_strand:
                continue
            if not (window.start <= nick_boundary <= window.end):
                continue
            placements.append(
                PlacementChoice(
                    variant=variant,
                    start=start,
                    end=start + motif_len,
                    orientation=orientation,  # type: ignore[arg-type]
                    display_motif=display_motif_for_orientation(variant, orientation=orientation),
                    nick_boundary=nick_boundary,
                    nick_strand=nick_strand,  # type: ignore[arg-type]
                )
            )
    placements.sort(key=lambda placement: (placement.nick_boundary, placement.start, placement.orientation))
    return placements


def _mirror_coupled(left: PlacementChoice, right: PlacementChoice, *, cassette_length: int) -> bool:
    return right.start == cassette_length - left.end and right.end == cassette_length - left.start


def _pairwise_variant_choices(
    *,
    solve_spec: HairpinCassetteSolveSpec,
    catalog: NickaseCatalog,
) -> tuple[list[VariantPairChoice], list[ValidationIssue]]:
    catalog_by_id = catalog.by_id()
    missing = sorted(
        {
            *solve_spec.assignment_policy.allowed_left_variant_ids,
            *solve_spec.assignment_policy.allowed_right_variant_ids,
            *solve_spec.assignment_policy.forbidden_intended_variant_ids,
        }
        - set(catalog_by_id)
    )
    if missing:
        return (
            [],
            [
                ValidationIssue(
                    code="UNKNOWN_VARIANT_ID",
                    message="Solve assignment policy references nickase ids not present in the merged catalog.",
                    details={"variant_ids": missing},
                )
            ],
        )

    forbidden_variants = set(solve_spec.assignment_policy.forbidden_intended_variant_ids)
    forbidden_specificities = set(solve_spec.assignment_policy.forbidden_intended_specificity_ids)

    choices: list[VariantPairChoice] = []
    for left_id in sorted(solve_spec.assignment_policy.allowed_left_variant_ids):
        left_variant = catalog_by_id[left_id]
        if left_variant.id in forbidden_variants or left_variant.specificity_id in forbidden_specificities:
            continue
        for right_id in sorted(solve_spec.assignment_policy.allowed_right_variant_ids):
            right_variant = catalog_by_id[right_id]
            if right_variant.id in forbidden_variants or right_variant.specificity_id in forbidden_specificities:
                continue
            if not solve_spec.assignment_policy.allow_same_variant and left_variant.id == right_variant.id:
                continue
            if (
                not solve_spec.assignment_policy.allow_same_specificity_opposite_variant
                and left_variant.specificity_id == right_variant.specificity_id
                and left_variant.id != right_variant.id
            ):
                continue
            choices.append(VariantPairChoice(left=left_variant, right=right_variant))

    if not choices:
        return (
            [],
            [
                ValidationIssue(
                    code="NO_ALLOWED_VARIANT_PAIRS",
                    message="Assignment policy eliminated every left/right intended nickase pairing.",
                    details={},
                )
            ],
        )
    return choices, []


def _validate_bounded_geometry(solve_spec: HairpinCassetteSolveSpec) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    min_possible = solve_spec.nick_goal.right_nick_window.start - solve_spec.nick_goal.left_nick_window.end
    max_possible = solve_spec.nick_goal.right_nick_window.end - solve_spec.nick_goal.left_nick_window.start
    if max_possible <= 0:
        issues.append(
            ValidationIssue(
                code="IMPOSSIBLE_BOUNDED_SEGMENT_GEOMETRY",
                message="Requested nick windows cannot produce left_boundary < right_boundary.",
                details={"min_possible_length": min_possible, "max_possible_length": max_possible},
            )
        )
    if solve_spec.nick_goal.bounded_segment_length is not None:
        allowed = solve_spec.nick_goal.bounded_segment_length
        if max_possible < allowed.min or min_possible > allowed.max:
            issues.append(
                ValidationIssue(
                    code="IMPOSSIBLE_BOUNDED_SEGMENT_GEOMETRY",
                    message="Requested bounded segment interval does not overlap the achievable nick-window geometry.",
                    details={
                        "min_possible_length": min_possible,
                        "max_possible_length": max_possible,
                        "requested_min": allowed.min,
                        "requested_max": allowed.max,
                    },
                )
            )
    if (
        solve_spec.search.bounded_segment_target is not None
        and solve_spec.nick_goal.bounded_segment_length is not None
        and not (
            solve_spec.nick_goal.bounded_segment_length.min
            <= solve_spec.search.bounded_segment_target
            <= solve_spec.nick_goal.bounded_segment_length.max
        )
    ):
        issues.append(
            ValidationIssue(
                code="BOUNDED_SEGMENT_TARGET_OUT_OF_RANGE",
                message=(
                    "search.bounded_segment_target must lie inside nick_goal.bounded_segment_length when both are set."
                ),
                details={"bounded_segment_target": solve_spec.search.bounded_segment_target},
            )
        )
    return issues


def _apply_left_constraint(
    domains: list[set[str]],
    *,
    left_index: int,
    allowed_bases: set[str],
) -> bool:
    updated = domains[left_index].intersection(allowed_bases)
    if not updated:
        return False
    domains[left_index] = updated
    return True


def _project_constraints_to_left_arm(
    *,
    solve_spec: HairpinCassetteSolveSpec,
    left_placement: PlacementChoice,
    right_placement: PlacementChoice,
) -> tuple[list[set[str]] | None, list[int]]:
    domains = [set(domain) for domain in _build_initial_domains(solve_spec.topology.stem5p_arm_pattern)]
    touched: set[int] = set()
    cassette_length = solve_spec.cassette_length_nt

    for offset, symbol in enumerate(left_placement.display_motif):
        index = left_placement.start + offset
        if not _apply_left_constraint(domains, left_index=index, allowed_bases=iupac_bases_for_symbol(symbol)):
            return None, []
        touched.add(index)

    for offset, symbol in enumerate(right_placement.display_motif):
        right_index = right_placement.start + offset
        left_index = cassette_length - 1 - right_index
        allowed_right = iupac_bases_for_symbol(symbol)
        allowed_left = {_COMPLEMENT[base] for base in allowed_right}
        if not _apply_left_constraint(domains, left_index=left_index, allowed_bases=allowed_left):
            return None, []
        touched.add(left_index)

    ordered_touched = sorted(touched, key=lambda index: (len(domains[index]), index))
    return domains, ordered_touched


def _build_full_sequences(
    *,
    left_stem_chars: list[str | None],
    loop_chars: list[str | None],
    solve_spec: HairpinCassetteSolveSpec,
) -> tuple[list[str | None], list[str | None]]:
    stem_length = solve_spec.stem_length_nt
    cassette_chars: list[str | None] = [None] * solve_spec.cassette_length_nt
    for index, base in enumerate(left_stem_chars):
        cassette_chars[index] = base
        if base is not None:
            cassette_chars[-(index + 1)] = _COMPLEMENT[base]
    for index, base in enumerate(loop_chars):
        cassette_chars[stem_length + index] = base

    evaluation_chars: list[str | None] = (
        list(solve_spec.construct_context.left_flank) + cassette_chars + list(solve_spec.construct_context.right_flank)
    )
    return cassette_chars, evaluation_chars


def _relevant_sequence_chars(
    *,
    solve_spec: HairpinCassetteSolveSpec,
    cassette_chars: list[str | None],
    evaluation_chars: list[str | None],
    scope: Literal["cassette_only", "evaluation_context"],
) -> list[str | None]:
    return cassette_chars if scope == "cassette_only" else evaluation_chars


def _violates_forbidden_sequence_windows(
    *,
    sequence_chars: list[str | None],
    motifs: Sequence[str],
) -> bool:
    concrete_motifs = sorted(set(motifs))
    for motif in concrete_motifs:
        motif_len = len(motif)
        for start in range(0, len(sequence_chars) - motif_len + 1):
            window_chars = sequence_chars[start : start + motif_len]
            if any(base is None for base in window_chars):
                continue
            window = "".join(base for base in window_chars if base is not None)
            if motif_matches(window, motif):
                return True
    return False


def _partial_homopolymer_violation(
    *,
    cassette_chars: list[str | None],
    max_run: int | None,
) -> bool:
    if max_run is None:
        return False
    current_base: str | None = None
    current_run = 0
    for base in cassette_chars:
        if base is None:
            current_base = None
            current_run = 0
            continue
        if base == current_base:
            current_run += 1
        else:
            current_base = base
            current_run = 1
        if current_run > max_run:
            return True
    return False


def _gc_bounds_still_feasible(
    *,
    left_domains: list[set[str]],
    loop_domains: list[set[str]],
    assigned_left: list[str | None],
    assigned_loop: list[str | None],
    solve_spec: HairpinCassetteSolveSpec,
) -> bool:
    gc_range = solve_spec.sequence_quality.gc_fraction
    if gc_range is None:
        return True

    left_assigned_gc = sum(1 for base in assigned_left if base in {"G", "C"})
    loop_assigned_gc = sum(1 for base in assigned_loop if base in {"G", "C"})

    remaining_left_domains = [
        domains for domains, base in zip(left_domains, assigned_left, strict=True) if base is None
    ]
    remaining_loop_domains = [
        domains for domains, base in zip(loop_domains, assigned_loop, strict=True) if base is None
    ]
    min_left_gc, max_left_gc = _min_max_gc_from_domains(remaining_left_domains)
    min_loop_gc, max_loop_gc = _min_max_gc_from_domains(remaining_loop_domains)

    min_total_gc = (2 * left_assigned_gc) + loop_assigned_gc + (2 * min_left_gc) + min_loop_gc
    max_total_gc = (2 * left_assigned_gc) + loop_assigned_gc + (2 * max_left_gc) + max_loop_gc
    min_fraction = min_total_gc / solve_spec.cassette_length_nt
    max_fraction = max_total_gc / solve_spec.cassette_length_nt
    return not (max_fraction < gc_range.min or min_fraction > gc_range.max)


def _build_candidate(
    *,
    assigned_left: list[str | None],
    assigned_loop: list[str | None],
    solve_spec: HairpinCassetteSolveSpec,
) -> ConcreteCandidate | None:
    if any(base is None for base in assigned_left) or any(base is None for base in assigned_loop):
        return None
    stem5p_arm = "".join(base for base in assigned_left if base is not None)
    loop = "".join(base for base in assigned_loop if base is not None)
    stem3p_arm = reverse_complement(stem5p_arm)
    cassette_sequence = f"{stem5p_arm}{loop}{stem3p_arm}"
    evaluation_primary_sequence = (
        f"{solve_spec.construct_context.left_flank}{cassette_sequence}{solve_spec.construct_context.right_flank}"
    )
    return ConcreteCandidate(
        stem5p_arm=stem5p_arm,
        loop=loop,
        stem3p_arm=stem3p_arm,
        cassette_sequence=cassette_sequence,
        evaluation_primary_sequence=evaluation_primary_sequence,
    )


def _make_explicit_spec(
    *,
    solve_spec: HairpinCassetteSolveSpec,
    candidate: ConcreteCandidate,
    left_placement: PlacementChoice,
    right_placement: PlacementChoice,
    catalog_path: Path,
    name: str,
) -> HairpinCassetteSpec:
    observed_length = right_placement.nick_boundary - left_placement.nick_boundary
    bounded_segment = solve_spec.nick_goal.bounded_segment_length
    if bounded_segment is None:
        bounded_segment_payload = {"min": observed_length, "max": observed_length}
    else:
        bounded_segment_payload = {"min": bounded_segment.min, "max": bounded_segment.max}
    return HairpinCassetteSpec(
        schema_version=2,
        name=name,
        topology=HairpinTopologySpec(
            stem5p_arm=candidate.stem5p_arm,
            loop=candidate.loop,
            stem3p_arm_mode="derived_reverse_complement",
        ),
        construct_context=ConstructContextSpec(
            left_flank=solve_spec.construct_context.left_flank,
            right_flank=solve_spec.construct_context.right_flank,
        ),
        nicking=DuplexNickingPlanSpec(
            target_strand=solve_spec.nick_goal.target_strand,
            left=FlankNickingRequest(
                nickase=left_placement.variant.id,
                nick_window=NickWindow(start=left_placement.nick_boundary, end=left_placement.nick_boundary),
            ),
            right=FlankNickingRequest(
                nickase=right_placement.variant.id,
                nick_window=NickWindow(start=right_placement.nick_boundary, end=right_placement.nick_boundary),
            ),
            require_exactly_two_intended_nicks=True,
            bounded_segment_length=bounded_segment_payload,
        ),
        site_policy=SitePolicySpec(
            forbid_additional_designated_strand_nicks=False,
            scan_scope="requested_variants",
        ),
        hairpin_validation=HairpinValidationSpec(
            require_topological_hairpin=True,
            require_energetic_hairpin=False,
        ),
        catalog=CassetteCatalogRef(path=catalog_path),
        output=CassetteOutputConfig(
            run_dir=solve_spec.output.run_dir,
            emit_visual_contracts=solve_spec.output.emit_visual_contracts,
            emit_baserender_jobs=solve_spec.output.emit_baserender_jobs,
            baserender_profiles=[
                profile for profile in solve_spec.output.baserender_profiles if profile in {"duplex_qa", "hairpin_qa"}
            ],
        ),
    )


def _candidate_hit_id(
    *,
    cassette_sequence: str,
    target_strand: Literal["primary", "complement"],
    left_variant_id: str,
    right_variant_id: str,
    left_boundary: int,
    right_boundary: int,
) -> str:
    payload = "\n".join(
        [
            cassette_sequence,
            target_strand,
            left_variant_id,
            right_variant_id,
            str(left_boundary),
            str(right_boundary),
        ]
    ).encode("utf-8")
    return sha256_bytes(payload)[:12]


def _append_warning(
    *,
    code: str,
    warnings: list[str],
    warning_codes: list[str],
) -> None:
    if code not in warning_codes:
        warning_codes.append(code)
        warnings.append(_WARNING_MESSAGES[code])


def _variable_order(
    *,
    left_domains: list[set[str]],
    loop_domains: list[set[str]],
    touched_left_positions: list[int],
) -> list[tuple[str, int]]:
    touched = set(touched_left_positions)
    touched_left = [("left", index) for index in touched_left_positions]
    remaining_left = [("left", index) for index in range(len(left_domains)) if index not in touched]
    remaining_left.sort(key=lambda item: (len(left_domains[item[1]]), item[1]))
    loop_positions = [("loop", index) for index in range(len(loop_domains))]
    loop_positions.sort(key=lambda item: (len(loop_domains[item[1]]), item[1]))
    return touched_left + remaining_left + loop_positions


def solve_cassette_search(
    *,
    solve_spec: HairpinCassetteSolveSpec,
    spec_path: Path,
    workspace_root: Path,
    catalog: NickaseCatalog,
    catalog_path: Path,
) -> SolveSearchResult:
    issues = _validate_bounded_geometry(solve_spec)
    variant_pairs, pair_issues = _pairwise_variant_choices(solve_spec=solve_spec, catalog=catalog)
    issues.extend(pair_issues)
    if issues:
        return SolveSearchResult(
            status="invalid_spec",
            issues=issues,
            warnings=[],
            warning_codes=[],
            hits=[],
            selection_summary=None,
            enumerated_candidate_count=0,
            accepted_candidate_count=0,
            considered_variant_pair_count=0,
            visited_search_node_count=0,
        )

    warnings: list[str] = []
    warning_codes: list[str] = []
    enumerated_candidates = 0
    accepted_candidates = 0
    visited_search_nodes = 0
    considered_variant_pair_count = 0
    admissible_pair_count = 0
    search_stopped = False
    accepted_pool = build_accepted_candidate_pool(pool_size=solve_spec.search.selection.pool_size)
    loop_domains = _build_initial_domains(solve_spec.topology.loop_pattern)
    blacklist_motifs = expanded_blacklist_motifs(
        literals=solve_spec.sequence_blacklist.forbidden_literals,
        iupac_motifs=solve_spec.sequence_blacklist.forbidden_iupac_motifs,
        include_reverse_complements=solve_spec.sequence_blacklist.forbid_reverse_complements,
    )

    for variant_pair in variant_pairs:
        considered_variant_pair_count += 1
        left_placements = _enumerate_placements_for_side(side="left", variant=variant_pair.left, solve_spec=solve_spec)
        right_placements = _enumerate_placements_for_side(
            side="right",
            variant=variant_pair.right,
            solve_spec=solve_spec,
        )
        if not left_placements or not right_placements:
            continue
        saw_admissible_placement = False
        for left_placement in left_placements:
            for right_placement in right_placements:
                if left_placement.nick_boundary >= right_placement.nick_boundary:
                    continue
                if (
                    variant_pair.left.id == variant_pair.right.id
                    and reverse_complement_iupac(variant_pair.left.motif_top_5to3) != variant_pair.left.motif_top_5to3
                    and _mirror_coupled(left_placement, right_placement, cassette_length=solve_spec.cassette_length_nt)
                ):
                    continue
                saw_admissible_placement = True
                projected_domains, touched_positions = _project_constraints_to_left_arm(
                    solve_spec=solve_spec,
                    left_placement=left_placement,
                    right_placement=right_placement,
                )
                if projected_domains is None:
                    continue
                assigned_left: list[str | None] = [None] * solve_spec.stem_length_nt
                assigned_loop: list[str | None] = [None] * solve_spec.loop_length_nt
                variable_order = _variable_order(
                    left_domains=projected_domains,
                    loop_domains=loop_domains,
                    touched_left_positions=touched_positions,
                )

                def dfs(depth: int) -> None:
                    nonlocal accepted_candidates, enumerated_candidates, search_stopped, visited_search_nodes
                    if search_stopped:
                        return
                    if visited_search_nodes >= solve_spec.search.max_search_nodes:
                        _append_warning(
                            code="MAX_SEARCH_NODES_REACHED",
                            warnings=warnings,
                            warning_codes=warning_codes,
                        )
                        search_stopped = True
                        return
                    visited_search_nodes += 1
                    cassette_chars, evaluation_chars = _build_full_sequences(
                        left_stem_chars=assigned_left,
                        loop_chars=assigned_loop,
                        solve_spec=solve_spec,
                    )
                    if _violates_forbidden_sequence_windows(
                        sequence_chars=_relevant_sequence_chars(
                            solve_spec=solve_spec,
                            cassette_chars=cassette_chars,
                            evaluation_chars=evaluation_chars,
                            scope=solve_spec.sequence_blacklist.scope,
                        ),
                        motifs=blacklist_motifs,
                    ):
                        return
                    if _partial_homopolymer_violation(
                        cassette_chars=cassette_chars,
                        max_run=solve_spec.sequence_quality.max_homopolymer_run,
                    ):
                        return
                    if not _gc_bounds_still_feasible(
                        left_domains=projected_domains,
                        loop_domains=loop_domains,
                        assigned_left=assigned_left,
                        assigned_loop=assigned_loop,
                        solve_spec=solve_spec,
                    ):
                        return
                    if depth == len(variable_order):
                        candidate = _build_candidate(
                            assigned_left=assigned_left,
                            assigned_loop=assigned_loop,
                            solve_spec=solve_spec,
                        )
                        if candidate is None:
                            return
                        if enumerated_candidates >= solve_spec.search.max_enumerated_candidates:
                            _append_warning(
                                code="MAX_ENUMERATED_CANDIDATES_REACHED",
                                warnings=warnings,
                                warning_codes=warning_codes,
                            )
                            search_stopped = True
                            return
                        enumerated_candidates += 1
                        hit_id = _candidate_hit_id(
                            cassette_sequence=candidate.cassette_sequence,
                            target_strand=solve_spec.nick_goal.target_strand,
                            left_variant_id=left_placement.variant.id,
                            right_variant_id=right_placement.variant.id,
                            left_boundary=left_placement.nick_boundary,
                            right_boundary=right_placement.nick_boundary,
                        )
                        explicit_spec = _make_explicit_spec(
                            solve_spec=solve_spec,
                            candidate=candidate,
                            left_placement=left_placement,
                            right_placement=right_placement,
                            catalog_path=catalog_path,
                            name=f"{spec_path.stem}__{hit_id}",
                        )
                        report = build_cassette_report(
                            explicit_spec,
                            spec_path=spec_path,
                            workspace_root=workspace_root,
                            catalog_path=catalog_path,
                            catalog=catalog,
                        )
                        if report.status != "satisfied" or report.candidate is None:
                            return
                        extra_issues = []
                        extra_issues.extend(
                            site_blacklist_issues(
                                candidate=candidate,
                                hit_report=report,
                                catalog=catalog,
                                solve_spec=solve_spec,
                            )
                        )
                        extra_issues.extend(sequence_blacklist_issues(candidate=candidate, solve_spec=solve_spec))
                        extra_issues.extend(sequence_quality_issues(candidate=candidate, solve_spec=solve_spec))
                        if extra_issues:
                            return
                        observed_gc_fraction = gc_fraction(candidate.cassette_sequence)
                        observed_homopolymer_run = max_homopolymer_run(candidate.cassette_sequence)
                        observed_extra_site_count = extra_site_count(
                            candidate=candidate,
                            hit_report=report,
                            catalog=catalog,
                            solve_spec=solve_spec,
                        )
                        breakdown, score_tuple = _score_hit(
                            cassette_sequence=candidate.cassette_sequence,
                            bounded_segment_length=report.candidate.bounded_nicked_segment.length_nt,
                            extra_site_count=observed_extra_site_count,
                            gc_fraction=observed_gc_fraction,
                            homopolymer_run=observed_homopolymer_run,
                            solve_spec=solve_spec,
                        )
                        accepted_candidates += 1
                        accepted_pool.consider(
                            CandidateHitRecord(
                                hit_id=hit_id,
                                left_variant_id=left_placement.variant.id,
                                right_variant_id=right_placement.variant.id,
                                explicit_spec=explicit_spec,
                                report=report,
                                cassette_sequence=candidate.cassette_sequence,
                                stem5p_arm=candidate.stem5p_arm,
                                loop=candidate.loop,
                                gc_fraction=observed_gc_fraction,
                                extra_site_count=observed_extra_site_count,
                                score_breakdown=breakdown,
                                base_penalty_vector=tuple(score_tuple[:-1]),
                                score_tuple=score_tuple,
                                left_nick_boundary=report.candidate.intended_left_nick.boundary,
                                right_nick_boundary=report.candidate.intended_right_nick.boundary,
                                bounded_segment_length=report.candidate.bounded_nicked_segment.length_nt,
                            )
                        )
                        return

                    location, index = variable_order[depth]
                    domain = projected_domains[index] if location == "left" else loop_domains[index]
                    for base in sorted(domain):
                        if location == "left":
                            assigned_left[index] = base
                        else:
                            assigned_loop[index] = base
                        dfs(depth + 1)
                        if location == "left":
                            assigned_left[index] = None
                        else:
                            assigned_loop[index] = None
                        if search_stopped:
                            return

                dfs(0)
                if search_stopped:
                    break
            if search_stopped:
                break
        if saw_admissible_placement:
            admissible_pair_count += 1
        if search_stopped:
            break

    if admissible_pair_count == 0:
        return SolveSearchResult(
            status="invalid_spec",
            issues=[
                ValidationIssue(
                    code="NO_ADMISSIBLE_VARIANT_PLACEMENTS",
                    message=(
                        "No allowed variant pair could place intended sites wholly inside the stem arms "
                        "while matching the requested windows and target strand."
                    ),
                    details={},
                )
            ],
            warnings=warnings,
            warning_codes=warning_codes,
            hits=[],
            selection_summary=None,
            enumerated_candidate_count=enumerated_candidates,
            accepted_candidate_count=0,
            considered_variant_pair_count=considered_variant_pair_count,
            visited_search_node_count=visited_search_nodes,
        )

    search_truncated = any(
        code in {"MAX_SEARCH_NODES_REACHED", "MAX_ENUMERATED_CANDIDATES_REACHED"} for code in warning_codes
    )
    selection_outcome = select_hits(
        accepted_pool=accepted_pool,
        search_settings=solve_spec.search,
        accepted_candidate_count=accepted_candidates,
        search_truncated=search_truncated,
    )
    if selection_outcome.pool_summary.truncated:
        _append_warning(
            code="ACCEPTED_POOL_TRUNCATED",
            warnings=warnings,
            warning_codes=warning_codes,
        )
    non_exhaustive_reason = selection_outcome.summary.selection_pool_non_exhaustive_reason
    if non_exhaustive_reason in {"pool_bounded", "search_bounded_and_pool_bounded"}:
        _append_warning(
            code="SELECTION_RESULTS_POOL_BOUNDED",
            warnings=warnings,
            warning_codes=warning_codes,
        )
    if non_exhaustive_reason in {"search_bounded", "search_bounded_and_pool_bounded"}:
        _append_warning(
            code="SELECTION_RESULTS_SEARCH_BOUNDED",
            warnings=warnings,
            warning_codes=warning_codes,
        )
    if selection_outcome.summary.policy_underfilled:
        _append_warning(
            code="SELECTION_POLICY_LIMITED_HITS",
            warnings=warnings,
            warning_codes=warning_codes,
        )
    if not selection_outcome.selected_hits:
        return SolveSearchResult(
            status="no_hits",
            issues=[],
            warnings=warnings,
            warning_codes=warning_codes,
            hits=[],
            selection_summary=selection_outcome.summary,
            enumerated_candidate_count=enumerated_candidates,
            accepted_candidate_count=accepted_candidates,
            considered_variant_pair_count=considered_variant_pair_count,
            visited_search_node_count=visited_search_nodes,
        )

    return SolveSearchResult(
        status="solved",
        issues=[],
        warnings=warnings,
        warning_codes=warning_codes,
        hits=selection_outcome.selected_hits,
        selection_summary=selection_outcome.summary,
        enumerated_candidate_count=enumerated_candidates,
        accepted_candidate_count=accepted_candidates,
        considered_variant_pair_count=considered_variant_pair_count,
        visited_search_node_count=visited_search_nodes,
    )


def build_solve_report(
    *,
    solve_spec: HairpinCassetteSolveSpec,
    spec_path: Path,
    workspace_root: Path,
    catalog: NickaseCatalog,
    search_result: SolveSearchResult,
) -> SolveReport:
    hits = [
        CandidateHit(
            rank=index,
            solution_id=selected_hit.record.hit_id,
            score=list(selected_hit.record.score_tuple),
            base_penalty_vector=list(selected_hit.record.base_penalty_vector),
            hit_id=selected_hit.record.hit_id,
            cassette_sequence=selected_hit.record.cassette_sequence,
            stem5p_arm=selected_hit.record.stem5p_arm,
            loop=selected_hit.record.loop,
            left_variant_id=selected_hit.record.left_variant_id,
            right_variant_id=selected_hit.record.right_variant_id,
            left_nick_boundary=selected_hit.record.left_nick_boundary,
            right_nick_boundary=selected_hit.record.right_nick_boundary,
            target_strand=solve_spec.nick_goal.target_strand,
            bounded_segment_length=selected_hit.record.bounded_segment_length,
            extra_site_count=selected_hit.record.extra_site_count,
            gc_fraction=selected_hit.record.gc_fraction,
            score_breakdown=selected_hit.record.score_breakdown,
            selection_rank_reason=selected_hit.selection_rank_reason,
            distance_to_previous_selected=(
                float(selected_hit.distance_to_previous_selected)
                if selected_hit.distance_to_previous_selected is not None
                else None
            ),
        )
        for index, selected_hit in enumerate(search_result.hits, start=1)
    ]
    return SolveReport(
        status=search_result.status,
        workspace_root=str(workspace_root),
        spec_path=str(spec_path),
        metadata=SolveReportMetadata(
            catalog_variants=_catalog_variants_for_report(catalog),
            warnings=search_result.warnings,
            warning_codes=search_result.warning_codes,
            enumerated_candidate_count=search_result.enumerated_candidate_count,
            accepted_candidate_count=search_result.accepted_candidate_count,
            considered_variant_pair_count=search_result.considered_variant_pair_count,
            visited_search_node_count=search_result.visited_search_node_count,
            catalog_preset=catalog.preset_id,
            catalog_additional_paths=[],
        ),
        issues=search_result.issues,
        hits=hits,
        selection_summary=search_result.selection_summary,
    )
