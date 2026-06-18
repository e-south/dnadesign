"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cruncher/src/snapback/planner.py

Deterministic validation and reporting for v2 explicit single-nick snapback.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from dnadesign.cruncher.nickases.models import NickaseCatalog
from dnadesign.cruncher.nickases.scanning import EvaluatedMatch, enumerate_site_instances
from dnadesign.cruncher.snapback.artifacts import display_workspace_relative
from dnadesign.cruncher.snapback.models import (
    EFFECTIVE_CAP_LOOP_NT,
    CoordinateSpan,
    PairContract,
    SingleNickSnapbackSpec,
    SnapbackCandidateDesign,
    SnapbackEvaluationReport,
    SnapbackIssue,
    SnapbackReportMetadata,
    build_catalog_info,
    build_post_nick_sequence,
    gc_distance_for_range,
    gc_fraction,
    max_homopolymer_run,
)

_COMPLEMENT_BASE = {
    "A": "T",
    "C": "G",
    "G": "C",
    "T": "A",
}


def _issue(code: str, message: str, **details: object) -> SnapbackIssue:
    return SnapbackIssue(code=code, message=message, details=details)


def _site_inside_duplex_window(match: EvaluatedMatch, duplex_window: CoordinateSpan) -> bool:
    return duplex_window.start <= match.site.start and match.site.end <= duplex_window.end


def _select_intended_match(
    *,
    matches: list[EvaluatedMatch],
    duplex_window: CoordinateSpan,
    boundary_min: int,
    boundary_max: int,
    normalize_to_top_strand_nick: bool,
) -> tuple[EvaluatedMatch | None, list[SnapbackIssue]]:
    in_boundary = [match for match in matches if boundary_min <= match.nick.boundary <= boundary_max]
    in_duplex = [match for match in in_boundary if _site_inside_duplex_window(match, duplex_window)]
    if normalize_to_top_strand_nick:
        normalized = [match for match in in_duplex if match.nick.strand == "primary"]
        if not normalized and any(match.nick.strand != "primary" for match in in_duplex):
            return (
                None,
                [
                    _issue(
                        "INTENDED_NICK_NOT_ON_TOP_STRAND",
                        "The requested nick window only resolved nicks on the "
                        "non-top strand after orientation normalization.",
                        window_min=boundary_min,
                        window_max=boundary_max,
                    )
                ],
            )
        in_duplex = normalized
    if not in_duplex:
        return (
            None,
            [
                _issue(
                    "INTENDED_NICK_WINDOW_NO_MATCH",
                    "No intended nick matched the requested nick boundary window inside pre_nick_duplex_window.",
                    window_min=boundary_min,
                    window_max=boundary_max,
                    duplex_window=duplex_window.model_dump(mode="json"),
                )
            ],
        )
    if len(in_duplex) > 1:
        return (
            None,
            [
                _issue(
                    "INTENDED_NICK_WINDOW_AMBIGUOUS",
                    "Multiple intended nick candidates matched the requested nick boundary window.",
                    count=len(in_duplex),
                )
            ],
        )
    return in_duplex[0], []


def _pairing_summary(*, retained_homology_sequence: str, foldback_arm: str) -> tuple[list[int], int, int, list[bool]]:
    mismatch_positions: list[int] = []
    matched_mask: list[bool] = []
    for index, (retained_base, arm_base) in enumerate(
        zip(retained_homology_sequence, reversed(foldback_arm), strict=True)
    ):
        matched = _COMPLEMENT_BASE[arm_base] == retained_base
        matched_mask.append(matched)
        if not matched:
            mismatch_positions.append(index)
    terminal_run = 0
    for matched in matched_mask:
        if not matched:
            break
        terminal_run += 1
    longest = 0
    current = 0
    for matched in matched_mask:
        if matched:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return mismatch_positions, terminal_run, longest, matched_mask


def _build_pair_map(
    *,
    retained_span: CoordinateSpan,
    foldback_span: CoordinateSpan,
    matched_mask: list[bool],
) -> list[PairContract]:
    pairs: list[PairContract] = []
    for index, matched in enumerate(matched_mask):
        if not matched:
            continue
        pairs.append(
            PairContract(
                left=retained_span.start + index,
                right=foldback_span.end - 1 - index,
            )
        )
    return pairs


def _protected_overlap_local_span(
    *,
    protected_region: CoordinateSpan,
    retained_homology_window: CoordinateSpan,
) -> tuple[int, int] | None:
    overlap_start = max(protected_region.start, retained_homology_window.start)
    overlap_end = min(protected_region.end, retained_homology_window.end)
    if overlap_end <= overlap_start:
        return None
    return overlap_start - retained_homology_window.start, overlap_end - retained_homology_window.start


def evaluate_snapback_candidate(
    *,
    input_sequence: str,
    protected_region: CoordinateSpan,
    pre_nick_duplex_window: CoordinateSpan,
    retained_homology_window: CoordinateSpan,
    cap_sequence: str,
    foldback_arm: str,
    homology_max_mismatches: int,
    terminal_ligatable_duplex_min: int,
    terminal_ligatable_duplex_max: int,
    max_uninterrupted_duplex_bp: int,
    max_added_nt: int,
    gc_bounds,
    max_homopolymer_run_allowed: int | None,
    intended_match: EvaluatedMatch,
    site_mutation_count: int,
    all_matches: list[EvaluatedMatch],
    forbid_additional_target_strand_nicks: bool,
    forbid_any_additional_nicks: bool,
) -> tuple[SnapbackCandidateDesign | None, list[SnapbackIssue]]:
    issues: list[SnapbackIssue] = []
    nick_boundary = intended_match.nick.boundary
    if retained_homology_window.start != nick_boundary:
        issues.append(
            _issue(
                "RETAINED_HOMOLOGY_MUST_START_AT_NICK",
                "retained_homology_window.start must equal the resolved nick boundary.",
                retained_homology_start=retained_homology_window.start,
                nick_boundary=nick_boundary,
            )
        )
        return None, issues

    retained_homology_sequence = input_sequence[retained_homology_window.start : retained_homology_window.end]
    released_prefix_sequence = ""
    source_cap_window = CoordinateSpan(start=retained_homology_window.end, end=len(input_sequence))
    source_cap_sequence = input_sequence[source_cap_window.start : source_cap_window.end]
    effective_cap_sequence = f"{source_cap_sequence}{cap_sequence}"
    added_sequence = f"{cap_sequence}{foldback_arm}"
    added_nt = len(added_sequence)
    if added_nt > max_added_nt:
        issues.append(
            _issue(
                "ADDED_NT_EXCEEDS_MAX",
                "The authored addition exceeds constraints.max_added_nt.",
                observed_added_nt=added_nt,
                max_added_nt=max_added_nt,
            )
        )
    mismatch_positions, terminal_run, longest_run, matched_mask = _pairing_summary(
        retained_homology_sequence=retained_homology_sequence,
        foldback_arm=foldback_arm,
    )
    mismatch_count = len(mismatch_positions)
    if mismatch_count > homology_max_mismatches:
        issues.append(
            _issue(
                "HOMOLOGY_MISMATCH_LIMIT_EXCEEDED",
                "foldback_arm exceeded topology.homology_policy.max_mismatches.",
                mismatch_count=mismatch_count,
                max_mismatches=homology_max_mismatches,
            )
        )
    protected_overlap = _protected_overlap_local_span(
        protected_region=protected_region,
        retained_homology_window=retained_homology_window,
    )
    if protected_overlap is not None:
        overlap_start, overlap_end = protected_overlap
        protected_overlap_mismatches = [
            position for position in mismatch_positions if overlap_start <= position < overlap_end
        ]
        if protected_overlap_mismatches:
            issues.append(
                _issue(
                    "PROTECTED_REGION_MISMATCH_OVERLAP",
                    "Mismatch positions must not fall inside the retained-homology overlap with protected_region.",
                    protected_region=protected_region.model_dump(mode="json"),
                    retained_homology_window=retained_homology_window.model_dump(mode="json"),
                    protected_overlap_mismatch_positions=protected_overlap_mismatches,
                )
            )
    if not terminal_ligatable_duplex_min <= terminal_run <= terminal_ligatable_duplex_max:
        issues.append(
            _issue(
                "TERMINAL_LIGATABLE_DUPLEX_BP_OUT_OF_RANGE",
                "The ligation-adjacent paired run fell outside constraints.terminal_ligatable_duplex_bp.",
                observed_terminal_ligatable_duplex_bp=terminal_run,
                min=terminal_ligatable_duplex_min,
                max=terminal_ligatable_duplex_max,
            )
        )
    if longest_run > max_uninterrupted_duplex_bp:
        issues.append(
            _issue(
                "MAX_UNINTERRUPTED_DUPLEX_BP_EXCEEDED",
                "The post-nick duplex exceeded constraints.max_uninterrupted_duplex_bp.",
                observed_max_uninterrupted_duplex_bp=longest_run,
                max_uninterrupted_duplex_bp=max_uninterrupted_duplex_bp,
            )
        )
    if len(effective_cap_sequence) != EFFECTIVE_CAP_LOOP_NT:
        issues.append(
            _issue(
                "EFFECTIVE_CAP_NT_OUT_OF_RANGE",
                "The effective snapback cap loop must be exactly 3 nt.",
                observed_cap_nt=len(effective_cap_sequence),
                required_cap_nt=EFFECTIVE_CAP_LOOP_NT,
                source_cap_nt=len(source_cap_sequence),
                cap_extension_nt=len(cap_sequence),
            )
        )
    gc_added = gc_fraction(added_sequence)
    gc_distance = gc_distance_for_range(added_sequence, gc_bounds)
    if gc_bounds is not None and not gc_bounds.min <= gc_added <= gc_bounds.max:
        issues.append(
            _issue(
                "GC_FRACTION_OUT_OF_RANGE",
                "The authored addition falls outside sequence_quality.gc_fraction.",
                observed_gc_fraction=gc_added,
                min=gc_bounds.min,
                max=gc_bounds.max,
            )
        )
    homopolymer_run = max_homopolymer_run(added_sequence)
    if max_homopolymer_run_allowed is not None and homopolymer_run > max_homopolymer_run_allowed:
        issues.append(
            _issue(
                "HOMOPOLYMER_RUN_EXCEEDED",
                "The authored addition exceeds sequence_quality.max_homopolymer_run.",
                observed_run=homopolymer_run,
                max_homopolymer_run=max_homopolymer_run_allowed,
            )
        )

    selected_key = intended_match.key()
    extra_nick_events = [match.nick for match in all_matches if match.key() != selected_key]
    extra_target_strand_nicks = [event for event in extra_nick_events if event.strand == intended_match.nick.strand]
    if forbid_additional_target_strand_nicks and extra_target_strand_nicks:
        issues.append(
            _issue(
                "EXTRA_TARGET_STRAND_NICKS_FOUND",
                "Additional nick events were detected on the intended nick strand.",
                count=len(extra_target_strand_nicks),
            )
        )
    if forbid_any_additional_nicks and extra_nick_events:
        issues.append(
            _issue(
                "EXTRA_NICKS_FOUND",
                "Additional nick events were detected for the selected nickase variant.",
                count=len(extra_nick_events),
            )
        )
    if issues:
        return None, issues

    input_length = len(input_sequence)
    cap_span = CoordinateSpan(start=input_length, end=input_length + len(cap_sequence))
    foldback_arm_span = CoordinateSpan(start=cap_span.end, end=cap_span.end + len(foldback_arm))
    post_nick_sequence = build_post_nick_sequence(
        released_prefix_sequence=released_prefix_sequence,
        retained_homology_sequence=retained_homology_sequence,
        source_cap_sequence=source_cap_sequence,
        cap_sequence=cap_sequence,
        foldback_arm=foldback_arm,
    )
    post_nick_released_prefix_span = CoordinateSpan(start=0, end=len(released_prefix_sequence))
    post_nick_retained_homology_span = CoordinateSpan(
        start=post_nick_released_prefix_span.end,
        end=post_nick_released_prefix_span.end + len(retained_homology_sequence),
    )
    post_nick_source_cap_span = CoordinateSpan(
        start=post_nick_retained_homology_span.end,
        end=post_nick_retained_homology_span.end + len(source_cap_sequence),
    )
    post_nick_cap_extension_span = CoordinateSpan(
        start=post_nick_source_cap_span.end,
        end=post_nick_source_cap_span.end + len(cap_sequence),
    )
    post_nick_cap_span = CoordinateSpan(
        start=post_nick_retained_homology_span.end,
        end=post_nick_cap_extension_span.end,
    )
    post_nick_foldback_arm_span = CoordinateSpan(
        start=post_nick_cap_span.end,
        end=post_nick_cap_span.end + len(foldback_arm),
    )
    pair_map = _build_pair_map(
        retained_span=post_nick_retained_homology_span,
        foldback_span=post_nick_foldback_arm_span,
        matched_mask=matched_mask,
    )
    return (
        SnapbackCandidateDesign(
            designed_sequence=f"{input_sequence}{cap_sequence}{foldback_arm}",
            input_sequence=input_sequence,
            protected_region=protected_region,
            pre_nick_duplex_window=pre_nick_duplex_window,
            retained_homology_window=retained_homology_window,
            source_cap_window=source_cap_window,
            cap_span=cap_span,
            foldback_arm_span=foldback_arm_span,
            retained_homology_sequence=retained_homology_sequence,
            released_prefix_sequence=released_prefix_sequence,
            source_cap_sequence=source_cap_sequence,
            effective_cap_sequence=effective_cap_sequence,
            cap_sequence=cap_sequence,
            foldback_arm=foldback_arm,
            intended_site=intended_match.site,
            intended_nick=intended_match.nick,
            nick_boundary=nick_boundary,
            nick_boundary_from_left=nick_boundary,
            site_mutation_count=site_mutation_count,
            released_prefix_nt=0,
            retained_start_from_nick=0,
            cap_nt=len(effective_cap_sequence),
            cap_extension_nt=len(cap_sequence),
            paired_bp=len(foldback_arm),
            mismatch_count=mismatch_count,
            mismatch_positions=mismatch_positions,
            terminal_ligatable_duplex_bp=terminal_run,
            max_uninterrupted_duplex_bp=longest_run,
            added_nt=added_nt,
            extra_nick_event_count=len(extra_nick_events),
            gc_fraction_added=gc_added,
            gc_distance=gc_distance,
            max_homopolymer_run_added=homopolymer_run,
            extra_target_strand_nicks=extra_target_strand_nicks,
            extra_nick_events=extra_nick_events,
            post_nick_sequence=post_nick_sequence,
            post_nick_released_prefix_span=post_nick_released_prefix_span,
            post_nick_retained_homology_span=post_nick_retained_homology_span,
            post_nick_source_cap_span=post_nick_source_cap_span,
            post_nick_cap_extension_span=post_nick_cap_extension_span,
            post_nick_cap_span=post_nick_cap_span,
            post_nick_foldback_arm_span=post_nick_foldback_arm_span,
            pair_map=pair_map,
        ),
        [],
    )


def _markdown_report(report: SnapbackEvaluationReport) -> str:
    lines = [
        f"# Snapback Report: {report.spec_name}",
        "",
        f"- status: {report.status}",
        f"- spec_path: {display_workspace_relative(report.spec_path, workspace_root=report.workspace_root)}",
        f"- catalog_source: {display_workspace_relative(report.catalog_source, workspace_root=report.workspace_root)}",
        f"- coordinate_semantics: {report.metadata.coordinate_semantics}",
        f"- boundary_semantics: {report.metadata.boundary_semantics}",
        f"- input_length_nt: {report.metadata.input_length_nt}",
        f"- added_nt: {report.metadata.added_nt}",
    ]
    for preset_id in report.metadata.catalog_presets:
        lines.append(f"- catalog_preset: {preset_id}")
    if report.run_dir:
        lines.append(f"- run_dir: {display_workspace_relative(report.run_dir, workspace_root=report.workspace_root)}")
    for code, warning in zip(report.metadata.warning_codes, report.metadata.warnings, strict=False):
        lines.append(f"- warning[{code}]: {warning}")
    if report.metadata.catalog_variants:
        catalog_entry = report.metadata.catalog_variants[0]
        lines.extend(
            [
                "",
                "## Nickase",
                f"- variant_id: {catalog_entry.variant_id}",
                f"- nicked_strand: {catalog_entry.nicked_strand}",
                f"- active_cut_offset: {catalog_entry.active_cut_offset}",
                (
                    "- outside_site: "
                    f"{catalog_entry.selection.outside_site if catalog_entry.selection is not None else 'unknown'}"
                ),
                (
                    "- snapback_tier: "
                    f"{catalog_entry.selection.snapback_tier if catalog_entry.selection is not None else '-'}"
                ),
                f"- vendor: {catalog_entry.vendor or catalog_entry.source or '-'}",
            ]
        )
        if catalog_entry.selection is not None and catalog_entry.selection.warning_codes:
            lines.append(f"- warnings: {', '.join(catalog_entry.selection.warning_codes)}")
    if report.candidate is not None:
        candidate = report.candidate
        lines.extend(
            [
                "",
                "## Candidate",
                f"- designed_sequence: `{candidate.designed_sequence}`",
                f"- intended_nick: {candidate.intended_nick.variant_id}@{candidate.nick_boundary}",
                f"- nick_boundary_from_left: {candidate.nick_boundary_from_left}",
                f"- site_mutation_count: {candidate.site_mutation_count}",
                f"- released_prefix_nt: {candidate.released_prefix_nt}",
                f"- retained_start_from_nick: {candidate.retained_start_from_nick}",
                f"- cap_nt: {candidate.cap_nt}",
                f"- cap_extension_nt: {candidate.cap_extension_nt}",
                f"- paired_bp: {candidate.paired_bp}",
                f"- mismatch_count: {candidate.mismatch_count}",
                f"- terminal_ligatable_duplex_bp: {candidate.terminal_ligatable_duplex_bp}",
                f"- max_uninterrupted_duplex_bp: {candidate.max_uninterrupted_duplex_bp}",
                f"- extra_nick_event_count: {candidate.extra_nick_event_count}",
            ]
        )
    if report.issues:
        lines.extend(["", "## Issues"])
        for issue in report.issues:
            lines.append(f"- {issue.code}: {issue.message}")
    return "\n".join(lines) + "\n"


def build_invalid_catalog_report(
    spec: SingleNickSnapbackSpec,
    *,
    spec_path: Path,
    workspace_root: Path,
    catalog_source: str,
    code: str,
    message: str,
    details: dict[str, object] | None = None,
) -> SnapbackEvaluationReport:
    return SnapbackEvaluationReport(
        status="invalid_catalog",
        spec_name=spec.name,
        workspace_root=str(workspace_root),
        spec_path=str(spec_path),
        catalog_source=catalog_source,
        metadata=SnapbackReportMetadata(
            input_length_nt=len(spec.input_sequence),
            added_nt=spec.added_nt,
            designed_length_nt=len(spec.designed_sequence),
            catalog_source=catalog_source,
            catalog_presets=spec.design.nickase.catalog.resolved_preset_ids(),
            catalog_variants=[],
        ),
        issues=[_issue(code, message, **(details or {}))],
    )


def build_snapback_report(
    spec: SingleNickSnapbackSpec,
    *,
    spec_path: Path,
    workspace_root: Path,
    catalog: NickaseCatalog,
    catalog_source: str,
) -> SnapbackEvaluationReport:
    catalog_by_id = catalog.by_id()
    variant_id = spec.design.nickase.variant_id
    if variant_id not in catalog_by_id:
        return build_invalid_catalog_report(
            spec,
            spec_path=spec_path,
            workspace_root=workspace_root,
            catalog_source=catalog_source,
            code="UNKNOWN_VARIANT_ID",
            message="design.nickase.variant_id was not found in the resolved nickase catalog.",
            details={"variant_id": variant_id},
        )
    entry = catalog_by_id[variant_id]
    designed_sequence = spec.designed_sequence
    matches = enumerate_site_instances(designed_sequence, coordinate_offset=0, entry=entry)
    intended_match, issues = _select_intended_match(
        matches=matches,
        duplex_window=spec.input.canonical_top_strand.pre_nick_duplex_window,
        boundary_min=spec.design.single_nick_goal.nick_boundary_window.min,
        boundary_max=spec.design.single_nick_goal.nick_boundary_window.max,
        normalize_to_top_strand_nick=spec.design.orientation_policy.normalize_to_top_strand_nick,
    )
    candidate = None
    if intended_match is not None and not issues:
        candidate, issues = evaluate_snapback_candidate(
            input_sequence=spec.input_sequence,
            protected_region=spec.input.canonical_top_strand.protected_region,
            pre_nick_duplex_window=spec.input.canonical_top_strand.pre_nick_duplex_window,
            retained_homology_window=spec.design.topology.retained_homology_window,
            cap_sequence=spec.design.topology.cap_sequence,
            foldback_arm=spec.design.topology.foldback_arm,
            homology_max_mismatches=spec.design.topology.homology_policy.max_mismatches,
            terminal_ligatable_duplex_min=spec.design.constraints.terminal_ligatable_duplex_bp.min,
            terminal_ligatable_duplex_max=spec.design.constraints.terminal_ligatable_duplex_bp.max,
            max_uninterrupted_duplex_bp=spec.design.constraints.max_uninterrupted_duplex_bp,
            max_added_nt=spec.design.constraints.max_added_nt,
            gc_bounds=spec.design.sequence_quality.gc_fraction,
            max_homopolymer_run_allowed=spec.design.sequence_quality.max_homopolymer_run,
            intended_match=intended_match,
            site_mutation_count=0,
            all_matches=matches,
            forbid_additional_target_strand_nicks=spec.design.constraints.forbid_additional_target_strand_nicks,
            forbid_any_additional_nicks=spec.design.constraints.forbid_any_additional_nicks,
        )

    status = "satisfied" if candidate is not None and not issues else "unsatisfied"
    return SnapbackEvaluationReport(
        status=status,
        spec_name=spec.name,
        workspace_root=str(workspace_root),
        spec_path=str(spec_path),
        catalog_source=catalog_source,
        metadata=SnapbackReportMetadata(
            input_length_nt=len(spec.input_sequence),
            added_nt=spec.added_nt,
            designed_length_nt=len(designed_sequence),
            catalog_source=catalog_source,
            catalog_presets=spec.design.nickase.catalog.resolved_preset_ids(),
            catalog_variants=[build_catalog_info(entry)],
        ),
        issues=issues,
        candidate=candidate,
    )


def render_markdown_report(report: SnapbackEvaluationReport) -> str:
    return _markdown_report(report)


__all__ = [
    "build_snapback_report",
    "build_invalid_catalog_report",
    "evaluate_snapback_candidate",
    "render_markdown_report",
]
