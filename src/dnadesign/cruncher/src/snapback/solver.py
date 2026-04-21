"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/snapback/solver.py

Bounded co-design solve/search for v3 snapback workflows.

Module Author(s): Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
from heapq import nsmallest
from itertools import product
from pathlib import Path

from dnadesign.cruncher.nickases.models import NickaseCatalog, reverse_complement
from dnadesign.cruncher.nickases.scanning import (
    enumerate_site_instances,
    enumerate_site_instances_starting_at_or_after,
    suffix_sensitive_scan_start,
)
from dnadesign.cruncher.nickases.selection import snapback_entry_priority_key
from dnadesign.cruncher.snapback.artifacts import display_workspace_relative
from dnadesign.cruncher.snapback.models import build_catalog_info
from dnadesign.cruncher.snapback.planner import evaluate_snapback_candidate
from dnadesign.cruncher.snapback.solve_models import (
    SingleNickSnapbackSolveSpec,
    SnapbackSolveFrontierRow,
    SnapbackSolveHit,
    SnapbackSolveReport,
    SnapbackSolveReportMetadata,
)
from dnadesign.cruncher.snapback.solve_search import (
    SnapbackCodesignedInput,
    SnapbackSearchFrontier,
    build_ordered_search_frontiers,
    enumerate_frontier_codesigned_inputs,
)


def _warning(code: str, message: str) -> tuple[str, str]:
    return code, message


def _lexical_dna_sequences(length: int):
    if length == 0:
        yield ""
        return
    for bases in product("ACGT", repeat=length):
        yield "".join(bases)


@lru_cache(maxsize=None)
def _lexical_dna_sequence_pool(length: int) -> tuple[str, ...]:
    return tuple(_lexical_dna_sequences(length))


def _enumerate_foldback_arms(retained_homology_sequence: str, *, max_mismatches: int):
    perfect = reverse_complement(retained_homology_sequence)

    def _walk(index: int, mismatches_used: int, prefix: list[str]):
        if index == len(perfect):
            yield "".join(prefix)
            return
        perfect_base = perfect[index]
        prefix.append(perfect_base)
        yield from _walk(index + 1, mismatches_used, prefix)
        prefix.pop()
        if mismatches_used >= max_mismatches:
            return
        for alt_base in "ACGT":
            if alt_base == perfect_base:
                continue
            prefix.append(alt_base)
            yield from _walk(index + 1, mismatches_used + 1, prefix)
            prefix.pop()

    yield from _walk(0, 0, [])


@lru_cache(maxsize=None)
def _foldback_arm_pool(retained_homology_sequence: str, max_mismatches: int) -> tuple[str, ...]:
    return tuple(_enumerate_foldback_arms(retained_homology_sequence, max_mismatches=max_mismatches))


@dataclass(frozen=True)
class _SnapbackSolveTrial:
    codesigned_input: SnapbackCodesignedInput
    designed_sequence: str
    cap_sequence: str
    foldback_arm: str
    invariant_prefix_matches: tuple[object, ...]
    appended_suffix_start: int


def _rank_key(candidate, *, catalog_by_id) -> tuple[object, ...]:
    entry = catalog_by_id[candidate.intended_nick.variant_id]
    return (
        candidate.nick_boundary_from_left,
        candidate.paired_bp,
        candidate.cap_extension_nt,
        candidate.added_nt,
        len(candidate.extra_nick_events),
        candidate.site_mutation_count,
        snapback_entry_priority_key(entry),
        round(candidate.gc_fraction_added, 6),
        candidate.max_homopolymer_run_added,
        candidate.max_uninterrupted_duplex_bp,
        candidate.cap_sequence,
        candidate.foldback_arm,
        candidate.intended_site.matched_span_sequence,
        candidate.input_sequence,
        candidate.intended_nick.variant_id,
        candidate.intended_site.orientation,
    )


def _hit_id(candidate) -> str:
    digest = sha256()
    digest.update(candidate.intended_nick.variant_id.encode("utf-8"))
    digest.update(b"\n")
    digest.update(candidate.designed_sequence.encode("utf-8"))
    return digest.hexdigest()[:12]


def _iter_snapback_trials(
    *,
    frontier: SnapbackSearchFrontier,
    codesigned_inputs: list[SnapbackCodesignedInput],
    max_mismatches: int,
):
    cap_sequences = _lexical_dna_sequence_pool(frontier.cap_extension_nt)
    retained_window = frontier.retained_homology_window
    for codesigned_input in codesigned_inputs:
        invariant_prefix_matches = tuple(
            enumerate_site_instances(
                codesigned_input.input_sequence,
                coordinate_offset=0,
                entry=codesigned_input.entry,
            )
        )
        appended_suffix_start = suffix_sensitive_scan_start(
            codesigned_input.entry,
            prefix_length=len(codesigned_input.input_sequence),
        )
        retained_homology_sequence = codesigned_input.input_sequence[retained_window.start : retained_window.end]
        foldback_arms = _foldback_arm_pool(retained_homology_sequence, max_mismatches)
        for cap_sequence in cap_sequences:
            for foldback_arm in foldback_arms:
                yield _SnapbackSolveTrial(
                    codesigned_input=codesigned_input,
                    designed_sequence=f"{codesigned_input.input_sequence}{cap_sequence}{foldback_arm}",
                    cap_sequence=cap_sequence,
                    foldback_arm=foldback_arm,
                    invariant_prefix_matches=invariant_prefix_matches,
                    appended_suffix_start=appended_suffix_start,
                )


def _build_frontier_rows(
    frontier_stats: dict[tuple[int, int, int], dict[str, int]],
) -> list[SnapbackSolveFrontierRow]:
    return [
        SnapbackSolveFrontierRow(
            nick_boundary_from_left=nick_boundary,
            paired_bp=paired_bp,
            cap_extension_nt=cap_extension_nt,
            codesigned_input_count=counts["codesigned_input_count"],
            enumerated_candidate_count=counts["enumerated_candidate_count"],
            accepted_candidate_count=counts["accepted_candidate_count"],
        )
        for (nick_boundary, paired_bp, cap_extension_nt), counts in sorted(frontier_stats.items())
    ]


def _build_solve_hits(selected, *, catalog_by_id) -> list[SnapbackSolveHit]:
    return [
        SnapbackSolveHit(
            rank=index,
            hit_id=_hit_id(candidate),
            variant_id=candidate.intended_nick.variant_id,
            intended_site_orientation=candidate.intended_site.orientation,
            intended_site_sequence=candidate.intended_site.matched_span_sequence,
            nick_boundary=candidate.nick_boundary,
            nick_boundary_from_left=candidate.nick_boundary_from_left,
            site_mutation_count=candidate.site_mutation_count,
            retained_start_from_nick=candidate.retained_start_from_nick,
            cap_sequence=candidate.cap_sequence,
            foldback_arm=candidate.foldback_arm,
            added_nt=candidate.added_nt,
            cap_nt=candidate.cap_nt,
            cap_extension_nt=candidate.cap_extension_nt,
            paired_bp=candidate.paired_bp,
            mismatch_count=candidate.mismatch_count,
            terminal_ligatable_duplex_bp=candidate.terminal_ligatable_duplex_bp,
            max_uninterrupted_duplex_bp=candidate.max_uninterrupted_duplex_bp,
            extra_nick_event_count=len(candidate.extra_nick_events),
            gc_fraction_added=candidate.gc_fraction_added,
            nickase=build_catalog_info(catalog_by_id[candidate.intended_nick.variant_id]),
            explicit_report=candidate,
        )
        for index, candidate in enumerate(selected, start=1)
    ]


def solve_snapback_search(
    spec: SingleNickSnapbackSolveSpec,
    *,
    spec_path: Path,
    workspace_root: Path,
    catalog: NickaseCatalog,
) -> SnapbackSolveReport:
    catalog_by_id = catalog.by_id()
    warnings: list[str] = []
    warning_codes: list[str] = []
    visited_nodes = 0
    enumerated_candidates = 0
    accepted_candidates = 0
    truncated = False
    accepted = []
    template_sequence = spec.input.canonical_top_strand.sequence
    duplex_window = spec.input.canonical_top_strand.pre_nick_duplex_window
    resolved_search_space = spec.resolved_search_space()
    frontier_stats: dict[tuple[int, int, int], dict[str, int]] = {}

    for frontier in build_ordered_search_frontiers(spec):
        frontier_key = frontier.key()
        frontier_row = frontier_stats.setdefault(
            frontier_key,
            {
                "codesigned_input_count": 0,
                "enumerated_candidate_count": 0,
                "accepted_candidate_count": 0,
            },
        )
        codesigned_inputs = enumerate_frontier_codesigned_inputs(
            template_sequence,
            frontier=frontier,
            catalog_entries=catalog.entries,
            duplex_window=duplex_window,
            normalize_to_top_strand_nick=spec.orientation_policy.normalize_to_top_strand_nick,
        )
        frontier_row["codesigned_input_count"] = len(codesigned_inputs)
        retained_window = frontier.retained_homology_window
        for trial in _iter_snapback_trials(
            frontier=frontier,
            codesigned_inputs=codesigned_inputs,
            max_mismatches=spec.search.max_mismatches,
        ):
            if visited_nodes >= spec.search.max_search_nodes:
                truncated = True
                warning_code, warning = _warning(
                    "MAX_SEARCH_NODES_REACHED",
                    "Search stopped after reaching search.max_search_nodes.",
                )
                warning_codes.append(warning_code)
                warnings.append(warning)
                break
            visited_nodes += 1
            if enumerated_candidates >= spec.search.max_enumerated_candidates:
                truncated = True
                warning_code, warning = _warning(
                    "MAX_ENUMERATED_CANDIDATES_REACHED",
                    "Search stopped after reaching search.max_enumerated_candidates.",
                )
                warning_codes.append(warning_code)
                warnings.append(warning)
                break
            enumerated_candidates += 1
            frontier_row["enumerated_candidate_count"] += 1
            suffix_sensitive_matches = enumerate_site_instances_starting_at_or_after(
                trial.designed_sequence,
                coordinate_offset=0,
                entry=trial.codesigned_input.entry,
                start_min=trial.appended_suffix_start,
            )
            all_matches = [*trial.invariant_prefix_matches, *suffix_sensitive_matches]
            candidate, issues = evaluate_snapback_candidate(
                input_sequence=trial.codesigned_input.input_sequence,
                protected_region=spec.input.canonical_top_strand.protected_region,
                pre_nick_duplex_window=duplex_window,
                retained_homology_window=retained_window,
                cap_sequence=trial.cap_sequence,
                foldback_arm=trial.foldback_arm,
                homology_max_mismatches=spec.search.max_mismatches,
                terminal_ligatable_duplex_min=resolved_search_space.terminal_ligatable_duplex_bp.min,
                terminal_ligatable_duplex_max=resolved_search_space.terminal_ligatable_duplex_bp.max,
                max_uninterrupted_duplex_bp=resolved_search_space.max_uninterrupted_duplex_bp,
                max_added_nt=spec.search.max_added_nt,
                gc_bounds=spec.sequence_quality.gc_fraction,
                max_homopolymer_run_allowed=spec.sequence_quality.max_homopolymer_run,
                intended_match=trial.codesigned_input.intended_match,
                site_mutation_count=trial.codesigned_input.site_mutation_count,
                all_matches=all_matches,
                forbid_additional_target_strand_nicks=spec.constraints.forbid_additional_target_strand_nicks,
                forbid_any_additional_nicks=spec.constraints.forbid_any_additional_nicks,
            )
            if issues or candidate is None:
                continue
            accepted_candidates += 1
            frontier_row["accepted_candidate_count"] += 1
            accepted.append(candidate)
        if truncated:
            break

    selected = nsmallest(
        spec.search.max_hits,
        accepted,
        key=lambda candidate: _rank_key(candidate, catalog_by_id=catalog_by_id),
    )
    frontier = _build_frontier_rows(frontier_stats)
    first_satisfied_frontier = next((row for row in frontier if row.accepted_candidate_count > 0), None)
    hits = _build_solve_hits(selected, catalog_by_id=catalog_by_id)
    status: str
    if truncated:
        status = "search_truncated"
    elif hits:
        status = "satisfied"
    else:
        status = "no_hits"
    return SnapbackSolveReport(
        status=status,
        spec_name=spec.name,
        spec_path=str(spec_path),
        workspace_root=str(workspace_root),
        metadata=SnapbackSolveReportMetadata(
            catalog_preset=spec.catalog.preset,
            catalog_presets=spec.catalog.resolved_preset_ids(),
            catalog_additional_paths=[str(path) for path in spec.catalog.additional_paths],
            resolved_search_space=resolved_search_space,
            visited_search_node_count=visited_nodes,
            enumerated_candidate_count=min(enumerated_candidates, spec.search.max_enumerated_candidates),
            accepted_candidate_count=accepted_candidates,
            materialized_hit_count=0,
            frontier_row_count=len(frontier),
            first_satisfied_frontier=first_satisfied_frontier,
            search_truncated=truncated,
            warnings=warnings,
            warning_codes=warning_codes,
        ),
        hits=hits,
        frontier=frontier,
    )


def render_solve_markdown_report(report: SnapbackSolveReport) -> str:
    lines = [
        "# Snapback Solve Report",
        "",
        f"- status: {report.status}",
        f"- spec_path: {display_workspace_relative(report.spec_path, workspace_root=report.workspace_root)}",
    ]
    if report.solve_id:
        lines.append(f"- solve_id: {report.solve_id}")
    if report.run_dir:
        lines.append(f"- run_dir: {display_workspace_relative(report.run_dir, workspace_root=report.workspace_root)}")
    if report.metadata.catalog_preset:
        lines.append(f"- catalog_preset: {report.metadata.catalog_preset}")
    preset_ids = (
        report.metadata.catalog_presets[1:] if report.metadata.catalog_preset else report.metadata.catalog_presets
    )
    for preset_id in preset_ids:
        lines.append(f"- catalog_preset: {preset_id}")
    for overlay in report.metadata.catalog_additional_paths:
        lines.append(f"- catalog_overlay: {display_workspace_relative(overlay, workspace_root=report.workspace_root)}")
    for code, warning in zip(report.metadata.warning_codes, report.metadata.warnings, strict=False):
        lines.append(f"- warning[{code}]: {warning}")
    lines.extend(
        [
            (
                "- resolved_nick_boundary_window: "
                f"{report.metadata.resolved_search_space.nick_boundary_window.min}"
                f"..{report.metadata.resolved_search_space.nick_boundary_window.max}"
            ),
            (
                "- resolved_retained_homology_length: "
                f"{report.metadata.resolved_search_space.retained_homology_length.min}"
                f"..{report.metadata.resolved_search_space.retained_homology_length.max}"
            ),
            f"- min_paired_bp: {report.metadata.resolved_search_space.min_paired_bp}",
            (
                "- resolved_terminal_ligatable_duplex_bp: "
                f"{report.metadata.resolved_search_space.terminal_ligatable_duplex_bp.min}"
                f"..{report.metadata.resolved_search_space.terminal_ligatable_duplex_bp.max}"
            ),
            (
                "- resolved_max_uninterrupted_duplex_bp: "
                f"{report.metadata.resolved_search_space.max_uninterrupted_duplex_bp}"
            ),
            f"- visited_search_node_count: {report.metadata.visited_search_node_count}",
            f"- enumerated_candidate_count: {report.metadata.enumerated_candidate_count}",
            f"- accepted_candidate_count: {report.metadata.accepted_candidate_count}",
            f"- materialized_hit_count: {report.metadata.materialized_hit_count}",
            f"- frontier_row_count: {report.metadata.frontier_row_count}",
        ]
    )
    if report.metadata.first_satisfied_frontier is not None:
        frontier = report.metadata.first_satisfied_frontier
        lines.append(
            "- first_satisfied_frontier: "
            f"boundary={frontier.nick_boundary_from_left} "
            f"paired_bp={frontier.paired_bp} "
            f"cap_ext_nt={frontier.cap_extension_nt} "
            f"codesigned_inputs={frontier.codesigned_input_count} "
            f"enumerated={frontier.enumerated_candidate_count} "
            f"accepted={frontier.accepted_candidate_count}"
        )
    if report.hits:
        lines.extend(["", "## Hits"])
        for hit in report.hits:
            outside_site = hit.nickase.selection.outside_site if hit.nickase.selection is not None else "unknown"
            snapback_tier = hit.nickase.selection.snapback_tier if hit.nickase.selection is not None else "-"
            lines.append(
                f"- rank {hit.rank}: {hit.hit_id} {hit.variant_id}@{hit.nick_boundary} "
                f"site={hit.intended_site_orientation}:{hit.intended_site_sequence} "
                f"site_mutations={hit.site_mutation_count} "
                f"cap_nt={hit.cap_nt} cap_ext_nt={hit.cap_extension_nt} "
                f"cap_ext_seq={hit.cap_sequence or '-'} added_nt={hit.added_nt}"
            )
            lines.append(
                "  "
                f"nickase={hit.nickase.nicked_strand}:{hit.nickase.active_cut_offset} "
                f"outside_site={outside_site} "
                f"snapback_tier={snapback_tier} "
                f"vendor={hit.nickase.vendor or hit.nickase.source or '-'}"
            )
            if hit.nickase.selection is not None and hit.nickase.selection.warning_codes:
                lines.append(f"  warnings={','.join(hit.nickase.selection.warning_codes)}")
    if report.issues:
        lines.extend(["", "## Issues"])
        for issue in report.issues:
            lines.append(f"- {issue.code}: {issue.message}")
    return "\n".join(lines) + "\n"


__all__ = ["render_solve_markdown_report", "solve_snapback_search"]
