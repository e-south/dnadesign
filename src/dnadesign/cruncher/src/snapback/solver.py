"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/snapback/solver.py

Bounded solve/search for v2 snapback workflows.

Module Author(s): Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from hashlib import sha256
from itertools import product
from pathlib import Path

from dnadesign.cruncher.nickases.models import NickaseCatalog, reverse_complement
from dnadesign.cruncher.nickases.scanning import enumerate_site_instances
from dnadesign.cruncher.snapback.models import CoordinateSpan
from dnadesign.cruncher.snapback.planner import evaluate_snapback_candidate
from dnadesign.cruncher.snapback.solve_models import (
    SingleNickSnapbackSolveSpec,
    SnapbackSolveHit,
    SnapbackSolveReport,
    SnapbackSolveReportMetadata,
)


def _warning(code: str, message: str) -> tuple[str, str]:
    return code, message


def _lexical_dna_sequences(length: int):
    if length == 0:
        yield ""
        return
    for bases in product("ACGT", repeat=length):
        yield "".join(bases)


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


def _rank_key(candidate) -> tuple[object, ...]:
    return (
        candidate.nick_boundary_from_left,
        candidate.retained_start_from_nick,
        candidate.added_nt,
        candidate.max_uninterrupted_duplex_bp,
        len(candidate.extra_nick_events),
        round(candidate.gc_distance, 6),
        candidate.max_homopolymer_run_added,
        candidate.cap_sequence,
        candidate.foldback_arm,
        candidate.intended_nick.variant_id,
        candidate.intended_site.orientation,
    )


def _hit_id(candidate) -> str:
    digest = sha256()
    digest.update(candidate.intended_nick.variant_id.encode("utf-8"))
    digest.update(b"\n")
    digest.update(candidate.designed_sequence.encode("utf-8"))
    return digest.hexdigest()[:12]


def solve_snapback_search(
    spec: SingleNickSnapbackSolveSpec,
    *,
    spec_path: Path,
    workspace_root: Path,
    catalog: NickaseCatalog,
) -> SnapbackSolveReport:
    warnings: list[str] = []
    warning_codes: list[str] = []
    catalog_by_id = catalog.by_id()
    invalid_variant_ids = [
        variant_id for variant_id in spec.nickase_policy.allowed_variant_ids if variant_id not in catalog_by_id
    ]
    if invalid_variant_ids:
        return SnapbackSolveReport(
            status="invalid_catalog",
            spec_name=spec.name,
            spec_path=str(spec_path),
            workspace_root=str(workspace_root),
            metadata=SnapbackSolveReportMetadata(
                catalog_preset=spec.catalog.preset,
                catalog_additional_paths=[str(path) for path in spec.catalog.additional_paths],
                visited_search_node_count=0,
                enumerated_candidate_count=0,
                accepted_candidate_count=0,
                materialized_hit_count=0,
            ),
            issues=[
                {
                    "code": "UNKNOWN_ALLOWED_VARIANT_ID",
                    "message": (
                        "nickase_policy.allowed_variant_ids referenced variants missing from the resolved catalog."
                    ),
                    "details": {"variant_ids": invalid_variant_ids},
                }
            ],
        )

    visited_nodes = 0
    enumerated_candidates = 0
    accepted_candidates = 0
    truncated = False
    accepted = []
    input_sequence = spec.input.canonical_top_strand.sequence
    duplex_window = spec.input.canonical_top_strand.pre_nick_duplex_window

    for variant_id in spec.nickase_policy.allowed_variant_ids:
        entry = catalog_by_id[variant_id]
        intended_matches = enumerate_site_instances(input_sequence, coordinate_offset=0, entry=entry)
        for intended_match in intended_matches:
            if intended_match.nick.boundary < spec.goal.nick_boundary_window.min:
                continue
            if intended_match.nick.boundary > spec.goal.nick_boundary_window.max:
                continue
            if intended_match.site.start < duplex_window.start or intended_match.site.end > duplex_window.end:
                continue
            if spec.nickase_policy.normalize_to_top_strand_nick and intended_match.nick.strand != "primary":
                continue
            for retained_start_delta in range(
                spec.goal.retained_start_from_nick.min,
                spec.goal.retained_start_from_nick.max + 1,
            ):
                retained_start = intended_match.nick.boundary + retained_start_delta
                for retained_length in range(
                    spec.search.retained_homology_length.min,
                    spec.search.retained_homology_length.max + 1,
                ):
                    retained_end = retained_start + retained_length
                    if retained_end > len(input_sequence):
                        continue
                    retained_window = CoordinateSpan(start=retained_start, end=retained_end)
                    retained_homology_sequence = input_sequence[retained_start:retained_end]
                    for cap_nt in range(spec.search.cap_nt.min, spec.search.cap_nt.max + 1):
                        for cap_sequence in _lexical_dna_sequences(cap_nt):
                            for foldback_arm in _enumerate_foldback_arms(
                                retained_homology_sequence,
                                max_mismatches=spec.search.max_mismatches,
                            ):
                                visited_nodes += 1
                                if visited_nodes > spec.search.max_search_nodes:
                                    truncated = True
                                    warning_code, warning = _warning(
                                        "MAX_SEARCH_NODES_REACHED",
                                        "Search stopped after reaching search.max_search_nodes.",
                                    )
                                    warning_codes.append(warning_code)
                                    warnings.append(warning)
                                    break
                                enumerated_candidates += 1
                                if enumerated_candidates > spec.search.max_enumerated_candidates:
                                    truncated = True
                                    warning_code, warning = _warning(
                                        "MAX_ENUMERATED_CANDIDATES_REACHED",
                                        "Search stopped after reaching search.max_enumerated_candidates.",
                                    )
                                    warning_codes.append(warning_code)
                                    warnings.append(warning)
                                    break
                                designed_sequence = f"{input_sequence}{cap_sequence}{foldback_arm}"
                                all_matches = enumerate_site_instances(
                                    designed_sequence,
                                    coordinate_offset=0,
                                    entry=entry,
                                )
                                candidate, issues = evaluate_snapback_candidate(
                                    input_sequence=input_sequence,
                                    protected_region=spec.input.canonical_top_strand.protected_region,
                                    pre_nick_duplex_window=duplex_window,
                                    retained_homology_window=retained_window,
                                    cap_sequence=cap_sequence,
                                    foldback_arm=foldback_arm,
                                    homology_max_mismatches=spec.search.max_mismatches,
                                    terminal_ligatable_duplex_min=spec.constraints.terminal_ligatable_duplex_bp.min,
                                    terminal_ligatable_duplex_max=spec.constraints.terminal_ligatable_duplex_bp.max,
                                    max_uninterrupted_duplex_bp=spec.constraints.max_uninterrupted_duplex_bp,
                                    max_added_nt=spec.search.max_added_nt,
                                    gc_bounds=spec.sequence_quality.gc_fraction,
                                    max_homopolymer_run_allowed=spec.sequence_quality.max_homopolymer_run,
                                    intended_match=intended_match,
                                    all_matches=all_matches,
                                    forbid_additional_target_strand_nicks=spec.constraints.forbid_additional_target_strand_nicks,
                                    forbid_any_additional_nicks=spec.constraints.forbid_any_additional_nicks,
                                )
                                if issues or candidate is None:
                                    continue
                                accepted_candidates += 1
                                accepted.append(candidate)
                            if truncated:
                                break
                        if truncated:
                            break
                    if truncated:
                        break
                if truncated:
                    break
            if truncated:
                break
        if truncated:
            break

    accepted.sort(key=_rank_key)
    selected = accepted[: spec.search.max_hits]
    hits = [
        SnapbackSolveHit(
            rank=index,
            hit_id=_hit_id(candidate),
            variant_id=candidate.intended_nick.variant_id,
            intended_site_orientation=candidate.intended_site.orientation,
            nick_boundary=candidate.nick_boundary,
            nick_boundary_from_left=candidate.nick_boundary_from_left,
            retained_start_from_nick=candidate.retained_start_from_nick,
            cap_sequence=candidate.cap_sequence,
            foldback_arm=candidate.foldback_arm,
            added_nt=candidate.added_nt,
            paired_bp=candidate.paired_bp,
            mismatch_count=candidate.mismatch_count,
            terminal_ligatable_duplex_bp=candidate.terminal_ligatable_duplex_bp,
            max_uninterrupted_duplex_bp=candidate.max_uninterrupted_duplex_bp,
            extra_nick_event_count=len(candidate.extra_nick_events),
            gc_fraction_added=candidate.gc_fraction_added,
            explicit_report=candidate,
        )
        for index, candidate in enumerate(selected, start=1)
    ]
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
            catalog_additional_paths=[str(path) for path in spec.catalog.additional_paths],
            visited_search_node_count=visited_nodes,
            enumerated_candidate_count=min(enumerated_candidates, spec.search.max_enumerated_candidates),
            accepted_candidate_count=accepted_candidates,
            materialized_hit_count=0,
            search_truncated=truncated,
            warnings=warnings,
            warning_codes=warning_codes,
        ),
        hits=hits,
    )


def render_solve_markdown_report(report: SnapbackSolveReport) -> str:
    lines = [
        "# Snapback Solve Report",
        "",
        f"- status: {report.status}",
        f"- spec_path: {report.spec_path}",
    ]
    if report.solve_id:
        lines.append(f"- solve_id: {report.solve_id}")
    if report.run_dir:
        lines.append(f"- run_dir: {report.run_dir}")
    if report.metadata.catalog_preset:
        lines.append(f"- catalog_preset: {report.metadata.catalog_preset}")
    for overlay in report.metadata.catalog_additional_paths:
        lines.append(f"- catalog_overlay: {overlay}")
    for code, warning in zip(report.metadata.warning_codes, report.metadata.warnings, strict=False):
        lines.append(f"- warning[{code}]: {warning}")
    lines.extend(
        [
            f"- visited_search_node_count: {report.metadata.visited_search_node_count}",
            f"- enumerated_candidate_count: {report.metadata.enumerated_candidate_count}",
            f"- accepted_candidate_count: {report.metadata.accepted_candidate_count}",
            f"- materialized_hit_count: {report.metadata.materialized_hit_count}",
        ]
    )
    if report.hits:
        lines.extend(["", "## Hits"])
        for hit in report.hits:
            lines.append(
                f"- rank {hit.rank}: {hit.hit_id} {hit.variant_id}@{hit.nick_boundary} "
                f"cap={hit.cap_sequence} added_nt={hit.added_nt}"
            )
    if report.issues:
        lines.extend(["", "## Issues"])
        for issue in report.issues:
            lines.append(f"- {issue.code}: {issue.message}")
    return "\n".join(lines) + "\n"


__all__ = ["render_solve_markdown_report", "solve_snapback_search"]
