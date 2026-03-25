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
from dnadesign.cruncher.cassette.scanning import (
    EvaluatedMatch,
    display_motif_for_orientation,
    enumerate_site_instances,
)
from dnadesign.cruncher.cassette.solve_models import (
    CandidateHit,
    CandidateScoreBreakdown,
    HairpinCassetteSolveSpec,
    SolveReport,
    SolveReportMetadata,
)
from dnadesign.cruncher.utils.hashing import sha256_bytes

_BASE_ORDER = ("A", "C", "G", "T")
_COMPLEMENT = {"A": "T", "C": "G", "G": "C", "T": "A"}


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
class CandidateHitRecord:
    hit_id: str
    left_variant_id: str
    right_variant_id: str
    explicit_spec: HairpinCassetteSpec
    report: object
    cassette_sequence: str
    stem5p_arm: str
    loop: str
    gc_fraction: float
    extra_site_count: int
    score_breakdown: CandidateScoreBreakdown
    score_tuple: tuple[float | int | str, ...]
    left_nick_boundary: int
    right_nick_boundary: int
    bounded_segment_length: int


@dataclass
class SolveSearchResult:
    status: Literal["solved", "no_hits", "invalid_spec"]
    issues: list[ValidationIssue]
    warnings: list[str]
    hits: list[CandidateHitRecord]
    enumerated_candidate_count: int
    accepted_candidate_count: int
    considered_variant_pair_count: int


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


def _gc_fraction(sequence: str) -> float:
    if not sequence:
        return 0.0
    gc = sum(1 for base in sequence if base in {"G", "C"})
    return gc / len(sequence)


def _max_homopolymer_run(sequence: str) -> int:
    if not sequence:
        return 0
    best = 1
    current = 1
    for index in range(1, len(sequence)):
        if sequence[index] == sequence[index - 1]:
            current += 1
            best = max(best, current)
        else:
            current = 1
    return best


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


def _expanded_blacklist_motifs(
    *,
    literals: list[str],
    iupac_motifs: list[str],
    include_reverse_complements: bool,
) -> tuple[str, ...]:
    motifs = list(literals) + list(iupac_motifs)
    if include_reverse_complements:
        motifs.extend(reverse_complement_iupac(motif) for motif in list(motifs))
    return tuple(sorted(set(motifs)))


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


def _scope_filtered_matches(
    *,
    matches: list[EvaluatedMatch],
    solve_spec: HairpinCassetteSolveSpec,
    scope: Literal["cassette_only", "evaluation_context"],
) -> list[EvaluatedMatch]:
    if scope == "evaluation_context":
        return matches
    cassette_start = len(solve_spec.construct_context.left_flank)
    cassette_end = cassette_start + solve_spec.cassette_length_nt
    return [match for match in matches if cassette_start <= match.site.start and match.site.end <= cassette_end]


def _scan_specificity_occurrences(
    *,
    candidate: ConcreteCandidate,
    catalog: NickaseCatalog,
    solve_spec: HairpinCassetteSolveSpec,
    specificity_ids: set[str] | None = None,
    variant_ids: set[str] | None = None,
) -> list[EvaluatedMatch]:
    entries_to_scan: list[NickaseCatalogEntry] = []
    if variant_ids:
        catalog_by_id = catalog.by_id()
        entries_to_scan.extend(
            catalog_by_id[variant_id] for variant_id in sorted(variant_ids) if variant_id in catalog_by_id
        )
    if specificity_ids:
        seen_specificities: set[str] = {entry.specificity_id for entry in entries_to_scan}
        for entry in sorted(catalog.entries, key=lambda item: (item.specificity_id, item.id)):
            if entry.specificity_id in specificity_ids and entry.specificity_id not in seen_specificities:
                entries_to_scan.append(entry)
                seen_specificities.add(entry.specificity_id)
    if not entries_to_scan:
        return []
    all_matches: list[EvaluatedMatch] = []
    cassette_offset = len(solve_spec.construct_context.left_flank)
    for entry in entries_to_scan:
        all_matches.extend(
            enumerate_site_instances(
                candidate.evaluation_primary_sequence,
                cassette_offset=cassette_offset,
                entry=entry,
            )
        )
    return all_matches


def _intended_site_keys(hit_report: object) -> set[tuple[str, int, int]]:
    candidate = getattr(hit_report, "candidate", None)
    if candidate is None:
        return set()
    return {
        (
            candidate.intended_left_site.specificity_id,
            candidate.intended_left_site.start,
            candidate.intended_left_site.end,
        ),
        (
            candidate.intended_right_site.specificity_id,
            candidate.intended_right_site.start,
            candidate.intended_right_site.end,
        ),
    }


def _site_blacklist_issues(
    *,
    candidate: ConcreteCandidate,
    hit_report: object,
    catalog: NickaseCatalog,
    solve_spec: HairpinCassetteSolveSpec,
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    any_specificities = set(solve_spec.site_blacklist.forbidden_any_site_specificity_ids)
    any_variants = set(solve_spec.site_blacklist.forbidden_any_site_variant_ids)
    unintended_specificities = set(solve_spec.site_blacklist.forbidden_unintended_site_specificity_ids)
    scope = solve_spec.site_blacklist.scope

    if any_specificities or any_variants:
        matches = _scope_filtered_matches(
            matches=_scan_specificity_occurrences(
                candidate=candidate,
                catalog=catalog,
                solve_spec=solve_spec,
                specificity_ids=any_specificities,
                variant_ids=any_variants,
            ),
            solve_spec=solve_spec,
            scope=scope,
        )
        if matches:
            issues.append(
                ValidationIssue(
                    code="FORBIDDEN_SITE_OCCURRENCE",
                    message="A forbidden site specificity or variant occurred in the protected scope.",
                    details={
                        "specificity_ids": sorted({match.site.specificity_id for match in matches}),
                        "variant_ids": sorted({match.variant.id for match in matches}),
                        "scope": scope,
                    },
                )
            )

    if unintended_specificities:
        matches = _scope_filtered_matches(
            matches=_scan_specificity_occurrences(
                candidate=candidate,
                catalog=catalog,
                solve_spec=solve_spec,
                specificity_ids=unintended_specificities,
            ),
            solve_spec=solve_spec,
            scope=scope,
        )
        intended_keys = _intended_site_keys(hit_report)
        extra_matches = [
            match
            for match in matches
            if (match.site.specificity_id, match.site.start, match.site.end) not in intended_keys
        ]
        if extra_matches:
            issues.append(
                ValidationIssue(
                    code="FORBIDDEN_UNINTENDED_SITE_OCCURRENCE",
                    message="A forbidden unintended site specificity occurred outside the chosen intended hits.",
                    details={
                        "specificity_ids": sorted({match.site.specificity_id for match in extra_matches}),
                        "scope": scope,
                    },
                )
            )
    return issues


def _sequence_blacklist_issues(
    *,
    candidate: ConcreteCandidate,
    solve_spec: HairpinCassetteSolveSpec,
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    if solve_spec.sequence_blacklist.scope == "cassette_only":
        sequence = candidate.cassette_sequence
    else:
        sequence = candidate.evaluation_primary_sequence
    motifs = _expanded_blacklist_motifs(
        literals=solve_spec.sequence_blacklist.forbidden_literals,
        iupac_motifs=solve_spec.sequence_blacklist.forbidden_iupac_motifs,
        include_reverse_complements=solve_spec.sequence_blacklist.forbid_reverse_complements,
    )
    violations: list[str] = []
    for motif in motifs:
        motif_len = len(motif)
        if any(
            motif_matches(sequence[start : start + motif_len], motif)
            for start in range(0, len(sequence) - motif_len + 1)
        ):
            violations.append(motif)
    if violations:
        issues.append(
            ValidationIssue(
                code="FORBIDDEN_SEQUENCE_MOTIF",
                message="A forbidden literal or IUPAC motif occurred in the protected sequence scope.",
                details={"motifs": sorted(set(violations)), "scope": solve_spec.sequence_blacklist.scope},
            )
        )
    return issues


def _sequence_quality_issues(
    *,
    candidate: ConcreteCandidate,
    solve_spec: HairpinCassetteSolveSpec,
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    gc_fraction = _gc_fraction(candidate.cassette_sequence)
    gc_range = solve_spec.sequence_quality.gc_fraction
    if gc_range is not None and not (gc_range.min <= gc_fraction <= gc_range.max):
        issues.append(
            ValidationIssue(
                code="GC_FRACTION_OUT_OF_RANGE",
                message="Cassette GC fraction fell outside the requested interval.",
                details={"gc_fraction": gc_fraction, "min": gc_range.min, "max": gc_range.max},
            )
        )
    homopolymer_run = _max_homopolymer_run(candidate.cassette_sequence)
    if (
        solve_spec.sequence_quality.max_homopolymer_run is not None
        and homopolymer_run > solve_spec.sequence_quality.max_homopolymer_run
    ):
        issues.append(
            ValidationIssue(
                code="HOMOPOLYMER_RUN_TOO_LONG",
                message="Cassette contains a homopolymer run longer than the requested maximum.",
                details={
                    "observed_max_homopolymer_run": homopolymer_run,
                    "max_homopolymer_run": solve_spec.sequence_quality.max_homopolymer_run,
                },
            )
        )
    return issues


def _extra_site_count(
    *,
    candidate: ConcreteCandidate,
    hit_report: object,
    catalog: NickaseCatalog,
    solve_spec: HairpinCassetteSolveSpec,
) -> int:
    seen_specificities: set[str] = set()
    specificity_entries: list[NickaseCatalogEntry] = []
    for entry in sorted(catalog.entries, key=lambda item: (item.specificity_id, item.id)):
        if entry.specificity_id not in seen_specificities:
            specificity_entries.append(entry)
            seen_specificities.add(entry.specificity_id)
    matches: list[EvaluatedMatch] = []
    cassette_offset = len(solve_spec.construct_context.left_flank)
    for entry in specificity_entries:
        matches.extend(
            enumerate_site_instances(
                candidate.evaluation_primary_sequence,
                cassette_offset=cassette_offset,
                entry=entry,
            )
        )
    matches = _scope_filtered_matches(matches=matches, solve_spec=solve_spec, scope=solve_spec.site_blacklist.scope)
    intended_keys = _intended_site_keys(hit_report)
    unique_occurrences = {
        (match.site.specificity_id, match.site.start, match.site.end, match.site.orientation) for match in matches
    }
    intended_occurrences = {
        occurrence
        for occurrence in unique_occurrences
        if (occurrence[0], occurrence[1], occurrence[2]) in intended_keys
    }
    return len(unique_occurrences - intended_occurrences)


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
            write_render_contract=solve_spec.output.write_render_contract,
        ),
    )


def _candidate_hit_id(cassette_sequence: str) -> str:
    return sha256_bytes(cassette_sequence.encode("utf-8"))[:10]


def _hamming_distance(left: str, right: str) -> int:
    return sum(1 for a, b in zip(left, right, strict=True) if a != b)


def _diversify_hits(
    hits: list[CandidateHitRecord],
    *,
    min_pairwise_hamming_distance: int,
    max_hits: int,
) -> list[CandidateHitRecord]:
    if min_pairwise_hamming_distance <= 0:
        return hits[:max_hits]
    kept: list[CandidateHitRecord] = []
    for hit in hits:
        if all(
            _hamming_distance(hit.cassette_sequence, existing.cassette_sequence) >= min_pairwise_hamming_distance
            for existing in kept
        ):
            kept.append(hit)
        if len(kept) >= max_hits:
            break
    return kept


def _deduplicate_hits(hits: list[CandidateHitRecord]) -> list[CandidateHitRecord]:
    best_by_sequence: dict[str, CandidateHitRecord] = {}
    for hit in hits:
        existing = best_by_sequence.get(hit.cassette_sequence)
        if existing is None or (hit.score_tuple, hit.left_variant_id, hit.right_variant_id) < (
            existing.score_tuple,
            existing.left_variant_id,
            existing.right_variant_id,
        ):
            best_by_sequence[hit.cassette_sequence] = hit
    return list(best_by_sequence.values())


def _recorded_hit_limit(solve_spec: HairpinCassetteSolveSpec) -> int:
    return max(
        64,
        solve_spec.search.max_hits * 8,
        solve_spec.search.materialize_top_k * 8,
    )


def _trim_hit_buffer(
    hits: list[CandidateHitRecord],
    *,
    solve_spec: HairpinCassetteSolveSpec,
    warnings: list[str],
) -> list[CandidateHitRecord]:
    limit = _recorded_hit_limit(solve_spec)
    if len(hits) <= limit * 2:
        return hits
    trimmed = _deduplicate_hits(hits)
    trimmed.sort(key=lambda hit: hit.score_tuple)
    if len(trimmed) > limit:
        warning = "internal hit buffer truncated to keep solve memory bounded."
        if warning not in warnings:
            warnings.append(warning)
        trimmed = trimmed[:limit]
    return trimmed


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
            hits=[],
            enumerated_candidate_count=0,
            accepted_candidate_count=0,
            considered_variant_pair_count=0,
        )

    all_hits: list[CandidateHitRecord] = []
    warnings: list[str] = []
    enumerated_candidates = 0
    accepted_candidates = 0
    visited_search_nodes = 0
    admissible_pair_count = 0
    search_stopped = False
    loop_domains = _build_initial_domains(solve_spec.topology.loop_pattern)
    blacklist_motifs = _expanded_blacklist_motifs(
        literals=solve_spec.sequence_blacklist.forbidden_literals,
        iupac_motifs=solve_spec.sequence_blacklist.forbidden_iupac_motifs,
        include_reverse_complements=solve_spec.sequence_blacklist.forbid_reverse_complements,
    )

    for variant_pair in variant_pairs:
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
                    nonlocal accepted_candidates, all_hits, enumerated_candidates, search_stopped, visited_search_nodes
                    if search_stopped:
                        return
                    visited_search_nodes += 1
                    if visited_search_nodes > solve_spec.search.max_search_nodes:
                        warnings.append("search.max_search_nodes reached before exhausting the solve search tree.")
                        search_stopped = True
                        return
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
                        enumerated_candidates += 1
                        if enumerated_candidates > solve_spec.search.max_enumerated_candidates:
                            warnings.append(
                                "search.max_enumerated_candidates reached before exhausting the solve search space."
                            )
                            search_stopped = True
                            return
                        explicit_spec = _make_explicit_spec(
                            solve_spec=solve_spec,
                            candidate=candidate,
                            left_placement=left_placement,
                            right_placement=right_placement,
                            catalog_path=catalog_path,
                            name=f"{spec_path.stem}__{_candidate_hit_id(candidate.cassette_sequence)}",
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
                            _site_blacklist_issues(
                                candidate=candidate,
                                hit_report=report,
                                catalog=catalog,
                                solve_spec=solve_spec,
                            )
                        )
                        extra_issues.extend(_sequence_blacklist_issues(candidate=candidate, solve_spec=solve_spec))
                        extra_issues.extend(_sequence_quality_issues(candidate=candidate, solve_spec=solve_spec))
                        if extra_issues:
                            return
                        gc_fraction = _gc_fraction(candidate.cassette_sequence)
                        homopolymer_run = _max_homopolymer_run(candidate.cassette_sequence)
                        extra_site_count = _extra_site_count(
                            candidate=candidate,
                            hit_report=report,
                            catalog=catalog,
                            solve_spec=solve_spec,
                        )
                        breakdown, score_tuple = _score_hit(
                            cassette_sequence=candidate.cassette_sequence,
                            bounded_segment_length=report.candidate.bounded_nicked_segment.length_nt,
                            extra_site_count=extra_site_count,
                            gc_fraction=gc_fraction,
                            homopolymer_run=homopolymer_run,
                            solve_spec=solve_spec,
                        )
                        accepted_candidates += 1
                        all_hits.append(
                            CandidateHitRecord(
                                hit_id=_candidate_hit_id(candidate.cassette_sequence),
                                left_variant_id=left_placement.variant.id,
                                right_variant_id=right_placement.variant.id,
                                explicit_spec=explicit_spec,
                                report=report,
                                cassette_sequence=candidate.cassette_sequence,
                                stem5p_arm=candidate.stem5p_arm,
                                loop=candidate.loop,
                                gc_fraction=gc_fraction,
                                extra_site_count=extra_site_count,
                                score_breakdown=breakdown,
                                score_tuple=score_tuple,
                                left_nick_boundary=report.candidate.intended_left_nick.boundary,
                                right_nick_boundary=report.candidate.intended_right_nick.boundary,
                                bounded_segment_length=report.candidate.bounded_nicked_segment.length_nt,
                            )
                        )
                        all_hits = _trim_hit_buffer(all_hits, solve_spec=solve_spec, warnings=warnings)
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
            hits=[],
            enumerated_candidate_count=enumerated_candidates,
            accepted_candidate_count=0,
            considered_variant_pair_count=0,
        )

    unique_hits = _deduplicate_hits(all_hits)
    unique_hits.sort(key=lambda hit: hit.score_tuple)
    ranked_hits = _diversify_hits(
        unique_hits,
        min_pairwise_hamming_distance=solve_spec.search.min_pairwise_hamming_distance,
        max_hits=solve_spec.search.max_hits,
    )
    if not ranked_hits:
        return SolveSearchResult(
            status="no_hits",
            issues=[],
            warnings=warnings,
            hits=[],
            enumerated_candidate_count=enumerated_candidates,
            accepted_candidate_count=accepted_candidates,
            considered_variant_pair_count=admissible_pair_count,
        )

    return SolveSearchResult(
        status="solved",
        issues=[],
        warnings=warnings,
        hits=ranked_hits,
        enumerated_candidate_count=enumerated_candidates,
        accepted_candidate_count=accepted_candidates,
        considered_variant_pair_count=admissible_pair_count,
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
            score=list(hit.score_tuple),
            hit_id=hit.hit_id,
            cassette_sequence=hit.cassette_sequence,
            stem5p_arm=hit.stem5p_arm,
            loop=hit.loop,
            left_variant_id=hit.left_variant_id,
            right_variant_id=hit.right_variant_id,
            left_nick_boundary=hit.left_nick_boundary,
            right_nick_boundary=hit.right_nick_boundary,
            target_strand=solve_spec.nick_goal.target_strand,
            bounded_segment_length=hit.bounded_segment_length,
            extra_site_count=hit.extra_site_count,
            gc_fraction=hit.gc_fraction,
            score_breakdown=hit.score_breakdown,
        )
        for index, hit in enumerate(search_result.hits, start=1)
    ]
    return SolveReport(
        status=search_result.status,
        workspace_root=str(workspace_root),
        spec_path=str(spec_path),
        metadata=SolveReportMetadata(
            catalog_variants=_catalog_variants_for_report(catalog),
            warnings=search_result.warnings,
            enumerated_candidate_count=search_result.enumerated_candidate_count,
            accepted_candidate_count=search_result.accepted_candidate_count,
            considered_variant_pair_count=search_result.considered_variant_pair_count,
            catalog_preset=catalog.preset_id,
            catalog_additional_paths=[],
        ),
        issues=search_result.issues,
        hits=hits,
    )
