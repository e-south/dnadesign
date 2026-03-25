"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/cassette/solve_filters.py

Solve-only blacklist, quality, and occurrence helpers for cassette search.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Literal

from dnadesign.cruncher.cassette.models import (
    NickaseCatalog,
    NickaseCatalogEntry,
    ValidationIssue,
    motif_matches,
    reverse_complement_iupac,
)
from dnadesign.cruncher.cassette.scanning import EvaluatedMatch, enumerate_site_instances
from dnadesign.cruncher.cassette.solve_models import HairpinCassetteSolveSpec


def gc_fraction(sequence: str) -> float:
    if not sequence:
        return 0.0
    gc = sum(1 for base in sequence if base in {"G", "C"})
    return gc / len(sequence)


def max_homopolymer_run(sequence: str) -> int:
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


def expanded_blacklist_motifs(
    *,
    literals: list[str],
    iupac_motifs: list[str],
    include_reverse_complements: bool,
) -> tuple[str, ...]:
    motifs = list(literals) + list(iupac_motifs)
    if include_reverse_complements:
        motifs.extend(reverse_complement_iupac(motif) for motif in list(motifs))
    return tuple(sorted(set(motifs)))


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
    candidate: object,
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
    evaluation_primary_sequence = getattr(candidate, "evaluation_primary_sequence")
    for entry in entries_to_scan:
        all_matches.extend(
            enumerate_site_instances(
                evaluation_primary_sequence,
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


def site_blacklist_issues(
    *,
    candidate: object,
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


def sequence_blacklist_issues(
    *,
    candidate: object,
    solve_spec: HairpinCassetteSolveSpec,
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    if solve_spec.sequence_blacklist.scope == "cassette_only":
        sequence = getattr(candidate, "cassette_sequence")
    else:
        sequence = getattr(candidate, "evaluation_primary_sequence")
    motifs = expanded_blacklist_motifs(
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


def sequence_quality_issues(
    *,
    candidate: object,
    solve_spec: HairpinCassetteSolveSpec,
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    cassette_sequence = getattr(candidate, "cassette_sequence")
    observed_gc_fraction = gc_fraction(cassette_sequence)
    gc_range = solve_spec.sequence_quality.gc_fraction
    if gc_range is not None and not (gc_range.min <= observed_gc_fraction <= gc_range.max):
        issues.append(
            ValidationIssue(
                code="GC_FRACTION_OUT_OF_RANGE",
                message="Cassette GC fraction fell outside the requested interval.",
                details={"gc_fraction": observed_gc_fraction, "min": gc_range.min, "max": gc_range.max},
            )
        )
    homopolymer_run = max_homopolymer_run(cassette_sequence)
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


def extra_site_count(
    *,
    candidate: object,
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
    evaluation_primary_sequence = getattr(candidate, "evaluation_primary_sequence")
    for entry in specificity_entries:
        matches.extend(
            enumerate_site_instances(
                evaluation_primary_sequence,
                cassette_offset=cassette_offset,
                entry=entry,
            )
        )
    matches = _scope_filtered_matches(
        matches=matches,
        solve_spec=solve_spec,
        scope=solve_spec.site_blacklist.scope,
    )
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
