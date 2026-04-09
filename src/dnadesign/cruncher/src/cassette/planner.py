"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/cassette/planner.py

Deterministic validation/report planning for dual-context cassette specs.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from dnadesign.cruncher.cassette.errors import CassettePlanningError
from dnadesign.cruncher.cassette.models import (
    BoundedNickedSegment,
    CassetteCandidateDesign,
    CassetteEvaluationReport,
    CassetteReportMetadata,
    CatalogNormalizationInfo,
    HairpinCassetteSpec,
    NickaseCatalog,
    NickaseCatalogEntry,
    NickEvent,
    NormalizedCassetteSpec,
    SpanContract,
    ValidationIssue,
    reverse_complement_iupac,
)
from dnadesign.cruncher.cassette.scanning import EvaluatedMatch, enumerate_site_instances


def _window_limit(spec: NormalizedCassetteSpec) -> int:
    return spec.topology.cassette_length_nt - 1 if spec.schema_version == 1 else spec.topology.cassette_length_nt


def _validate_window_bounds(spec: NormalizedCassetteSpec) -> None:
    max_boundary = _window_limit(spec)
    for label, request in (("left", spec.nicking.left), ("right", spec.nicking.right)):
        if request.window_end > max_boundary:
            raise CassettePlanningError(
                f"{label}.nick_window.end={request.window_end} "
                f"exceeds cassette length {spec.topology.cassette_length_nt}."
            )
        if request.window_start > max_boundary:
            raise CassettePlanningError(
                f"{label}.nick_window.start={request.window_start} "
                f"exceeds cassette length {spec.topology.cassette_length_nt}."
            )


def _is_within_left_stem(match: EvaluatedMatch, *, spec: NormalizedCassetteSpec) -> bool:
    if match.site.cassette_start is None or match.site.cassette_end is None:
        return False
    return 0 <= match.site.cassette_start and match.site.cassette_end <= spec.topology.stem_length_nt


def _is_within_right_stem(match: EvaluatedMatch, *, spec: NormalizedCassetteSpec) -> bool:
    if match.site.cassette_start is None or match.site.cassette_end is None:
        return False
    right_stem_start = spec.topology.stem_length_nt + spec.topology.loop_length_nt
    return right_stem_start <= match.site.cassette_start and match.site.cassette_end <= spec.topology.cassette_length_nt


def _filter_window(matches: list[EvaluatedMatch], *, window_start: int, window_end: int) -> list[EvaluatedMatch]:
    return [match for match in matches if window_start <= match.nick.boundary <= window_end]


def _filter_target_strand(matches: list[EvaluatedMatch], *, target_strand: str) -> list[EvaluatedMatch]:
    return [match for match in matches if match.nick.strand == target_strand]


def _mirror_coupled(left_match: EvaluatedMatch, right_match: EvaluatedMatch, *, cassette_length: int) -> bool:
    if left_match.site.cassette_start is None or left_match.site.cassette_end is None:
        return False
    if right_match.site.cassette_start is None or right_match.site.cassette_end is None:
        return False
    return (
        right_match.site.cassette_start == cassette_length - left_match.site.cassette_end
        and right_match.site.cassette_end == cassette_length - left_match.site.cassette_start
    )


def _preflight_symmetry_issues(
    spec: NormalizedCassetteSpec,
    *,
    catalog_by_id: dict[str, NickaseCatalogEntry],
    left_matches: list[EvaluatedMatch],
    right_matches: list[EvaluatedMatch],
) -> list[ValidationIssue]:
    if spec.nicking.left.variant_id != spec.nicking.right.variant_id:
        return []
    entry = catalog_by_id[spec.nicking.left.variant_id]
    if reverse_complement_iupac(entry.motif_top_5to3) == entry.motif_top_5to3:
        return []
    left_candidates = _filter_target_strand(
        _filter_window(
            [match for match in left_matches if _is_within_left_stem(match, spec=spec)],
            window_start=spec.nicking.left.window_start,
            window_end=spec.nicking.left.window_end,
        ),
        target_strand=spec.nicking.target_strand,
    )
    right_wrong_strand = [
        match
        for match in _filter_window(
            [match for match in right_matches if _is_within_right_stem(match, spec=spec)],
            window_start=spec.nicking.right.window_start,
            window_end=spec.nicking.right.window_end,
        )
        if match.nick.strand != spec.nicking.target_strand
    ]
    for left_match in left_candidates:
        for right_match in right_wrong_strand:
            if _mirror_coupled(left_match, right_match, cassette_length=spec.topology.cassette_length_nt):
                return [
                    ValidationIssue(
                        code="UNSAT_BY_MIRROR_SYMMETRY",
                        message=(
                            "Mirror-coupled placement with the same non-palindromic nickase variant forces the "
                            "opposite duplex strand on the mirrored arm."
                        ),
                        details={
                            "variant_id": entry.id,
                            "target_strand": spec.nicking.target_strand,
                        },
                    )
                ]
    return []


def _select_intended_matches(
    *,
    side: str,
    spec: NormalizedCassetteSpec,
    matches: list[EvaluatedMatch],
) -> tuple[EvaluatedMatch | None, list[ValidationIssue]]:
    if side == "left":
        request = spec.nicking.left

        def stem_filter(match: EvaluatedMatch) -> bool:
            return _is_within_left_stem(match, spec=spec)

        stem_issue_code = "LEFT_SITE_NOT_IN_LEFT_STEM"
        window_issue_code = "LEFT_WINDOW_NO_MATCH"
        ambiguous_issue_code = "LEFT_WINDOW_AMBIGUOUS"
    else:
        request = spec.nicking.right

        def stem_filter(match: EvaluatedMatch) -> bool:
            return _is_within_right_stem(match, spec=spec)

        stem_issue_code = "RIGHT_SITE_NOT_IN_RIGHT_STEM"
        window_issue_code = "RIGHT_WINDOW_NO_MATCH"
        ambiguous_issue_code = "RIGHT_WINDOW_AMBIGUOUS"

    window_matches = _filter_window(matches, window_start=request.window_start, window_end=request.window_end)
    if not window_matches:
        return (
            None,
            [
                ValidationIssue(
                    code=window_issue_code,
                    message=f"No nick boundary matched the {side} window.",
                    details={"variant_id": request.variant_id},
                )
            ],
        )

    stem_matches = [match for match in window_matches if stem_filter(match)]
    if not stem_matches:
        return (
            None,
            [
                ValidationIssue(
                    code=stem_issue_code,
                    message=f"The intended {side} site must lie wholly inside the {side} stem arm.",
                    details={"variant_id": request.variant_id},
                )
            ],
        )

    target_matches = _filter_target_strand(stem_matches, target_strand=spec.nicking.target_strand)
    if not target_matches:
        return (
            None,
            [
                ValidationIssue(
                    code="TARGET_STRAND_MISMATCH",
                    message=f"The {side} nick window only matched the opposite duplex strand.",
                    details={
                        "side": side,
                        "variant_id": request.variant_id,
                        "target_strand": spec.nicking.target_strand,
                        "available_strands": sorted({match.nick.strand for match in stem_matches}),
                    },
                )
            ],
        )

    if len(target_matches) > 1:
        return (
            None,
            [
                ValidationIssue(
                    code=ambiguous_issue_code,
                    message=f"Multiple intended {side} nick candidates matched the requested window.",
                    details={"variant_id": request.variant_id, "count": len(target_matches)},
                )
            ],
        )
    return target_matches[0], []


def _collect_extra_designated_strand_nicks(
    *,
    spec: NormalizedCassetteSpec,
    all_matches: list[EvaluatedMatch],
    selected_matches: list[EvaluatedMatch],
) -> list[NickEvent]:
    selected_keys = {match.key() for match in selected_matches}
    return [
        match.nick
        for match in all_matches
        if match.nick.strand == spec.nicking.target_strand and match.key() not in selected_keys
    ]


def _catalog_variants_for_report(
    *,
    spec: NormalizedCassetteSpec,
    catalog_by_id: dict[str, NickaseCatalogEntry],
) -> list[CatalogNormalizationInfo]:
    variant_ids = {spec.nicking.left.variant_id, spec.nicking.right.variant_id}
    return [
        CatalogNormalizationInfo(
            variant_id=variant.id,
            specificity_id=variant.specificity_id,
            motif_top_5to3=variant.motif_top_5to3,
            motif_len=variant.motif_len or len(variant.motif_top_5to3),
            top_cut_offset=variant.top_cut_offset,
            bottom_cut_offset=variant.bottom_cut_offset,
            source=variant.source,
            raw_cut_notation=variant.raw_cut_notation,
            metadata=variant.metadata,
        )
        for variant_id in sorted(variant_ids)
        for variant in [catalog_by_id[variant_id]]
    ]


def _markdown_report(report: CassetteEvaluationReport) -> str:
    lines = [
        f"# Cassette Report: {report.spec_name}",
        "",
        f"- status: {report.status}",
        f"- target_strand: {report.target_strand}",
        f"- spec_path: {report.spec_path}",
        f"- catalog_path: {report.catalog_path}",
        f"- coordinate_semantics: {report.metadata.coordinate_semantics}",
        f"- bounded_segment_statement: {report.metadata.bounded_segment_statement}",
    ]
    if report.run_dir:
        lines.append(f"- run_dir: {report.run_dir}")
    for warning in report.metadata.warnings:
        lines.append(f"- warning: {warning}")
    if report.candidate is not None:
        candidate = report.candidate
        lines.extend(
            [
                "",
                "## Candidate",
                f"- cassette_sequence: `{candidate.cassette_sequence}`",
                (
                    "- intended_nicks: "
                    f"{candidate.intended_left_nick.variant_id}@{candidate.intended_left_nick.boundary}, "
                    f"{candidate.intended_right_nick.variant_id}@{candidate.intended_right_nick.boundary}"
                ),
                (
                    "- bounded_nicked_segment: "
                    f"{candidate.bounded_nicked_segment.start_boundary}.."
                    f"{candidate.bounded_nicked_segment.end_boundary} "
                    f"(length={candidate.bounded_nicked_segment.length_nt})"
                ),
            ]
        )
        if candidate.extra_designated_strand_nicks:
            lines.append("- extra_designated_strand_nicks:")
            for extra in candidate.extra_designated_strand_nicks:
                lines.append(
                    f"  - {extra.variant_id} {extra.source_site_orientation} "
                    f"site=[{extra.source_site_start},{extra.source_site_end}) boundary={extra.boundary}"
                )
    if report.issues:
        lines.extend(["", "## Issues"])
        for issue in report.issues:
            lines.append(f"- {issue.code}: {issue.message}")
    return "\n".join(lines) + "\n"


def build_cassette_report(
    spec: HairpinCassetteSpec,
    *,
    spec_path: Path,
    workspace_root: Path,
    catalog_path: Path,
    catalog: NickaseCatalog,
) -> CassetteEvaluationReport:
    normalized = spec.normalize()
    _validate_window_bounds(normalized)

    catalog_by_id = catalog.by_id()
    missing = [
        variant_id
        for variant_id in {normalized.nicking.left.variant_id, normalized.nicking.right.variant_id}
        if variant_id not in catalog_by_id
    ]
    if missing:
        raise CassettePlanningError(f"Nickase ids not found in catalog: {', '.join(sorted(missing))}")

    scan_variant_ids = (
        sorted(catalog_by_id)
        if normalized.site_policy.scan_scope == "catalog"
        else sorted({normalized.nicking.left.variant_id, normalized.nicking.right.variant_id})
    )
    all_matches: list[EvaluatedMatch] = []
    for variant_id in scan_variant_ids:
        all_matches.extend(
            enumerate_site_instances(
                normalized.construct_context.evaluation_primary_sequence,
                cassette_offset=normalized.construct_context.cassette_start_offset,
                entry=catalog_by_id[variant_id],
            )
        )

    left_matches = [match for match in all_matches if match.variant.id == normalized.nicking.left.variant_id]
    right_matches = [match for match in all_matches if match.variant.id == normalized.nicking.right.variant_id]

    issues = _preflight_symmetry_issues(
        normalized,
        catalog_by_id=catalog_by_id,
        left_matches=left_matches,
        right_matches=right_matches,
    )

    left_match: EvaluatedMatch | None = None
    right_match: EvaluatedMatch | None = None
    if not issues:
        left_match, left_issues = _select_intended_matches(side="left", spec=normalized, matches=left_matches)
        right_match, right_issues = _select_intended_matches(side="right", spec=normalized, matches=right_matches)
        issues.extend(left_issues)
        issues.extend(right_issues)

    candidate: CassetteCandidateDesign | None = None
    if not issues and left_match is not None and right_match is not None:
        if left_match.nick.boundary >= right_match.nick.boundary:
            issues.append(
                ValidationIssue(
                    code="LEFT_NOT_BEFORE_RIGHT",
                    message="Left boundary must be strictly less than right boundary.",
                    details={
                        "left_boundary": left_match.nick.boundary,
                        "right_boundary": right_match.nick.boundary,
                    },
                )
            )
        else:
            bounded_nicked_segment = BoundedNickedSegment(
                strand=normalized.nicking.target_strand,
                start_boundary=left_match.nick.boundary,
                end_boundary=right_match.nick.boundary,
                length_nt=right_match.nick.boundary - left_match.nick.boundary,
            )
            if normalized.nicking.bounded_segment_length is not None:
                allowed = normalized.nicking.bounded_segment_length
                if not (allowed.min <= bounded_nicked_segment.length_nt <= allowed.max):
                    issues.append(
                        ValidationIssue(
                            code="BOUNDED_SEGMENT_LENGTH_OUT_OF_RANGE",
                            message="Bounded nicked segment length fell outside the requested interval.",
                            details={
                                "observed_length": bounded_nicked_segment.length_nt,
                                "min": allowed.min,
                                "max": allowed.max,
                            },
                        )
                    )
            extras = _collect_extra_designated_strand_nicks(
                spec=normalized,
                all_matches=all_matches,
                selected_matches=[left_match, right_match],
            )
            if normalized.site_policy.forbid_additional_designated_strand_nicks and extras:
                issues.append(
                    ValidationIssue(
                        code="EXTRA_DESIGNATED_STRAND_NICKS_FOUND",
                        message="Additional designated-strand nick events were detected under the active scan scope.",
                        details={"count": len(extras), "scan_scope": normalized.site_policy.scan_scope},
                    )
                )
            else:
                candidate = CassetteCandidateDesign(
                    cassette_sequence=normalized.topology.cassette_sequence,
                    stem5p_arm=normalized.topology.stem5p_arm,
                    loop=normalized.topology.loop,
                    stem3p_arm=normalized.topology.stem3p_arm,
                    target_strand=normalized.nicking.target_strand,
                    intended_left_site=left_match.site,
                    intended_right_site=right_match.site,
                    intended_left_nick=left_match.nick,
                    intended_right_nick=right_match.nick,
                    bounded_nicked_segment=bounded_nicked_segment,
                    extra_designated_strand_nicks=extras,
                    evaluation_primary_sequence=normalized.construct_context.evaluation_primary_sequence,
                    evaluation_complement_sequence=normalized.construct_context.evaluation_complement_sequence,
                    cassette_length_nt=normalized.topology.cassette_length_nt,
                    context_offset=normalized.construct_context.cassette_start_offset,
                    stem5p_span=SpanContract(start=0, end=normalized.topology.stem_length_nt),
                    loop_span=SpanContract(
                        start=normalized.topology.stem_length_nt,
                        end=normalized.topology.stem_length_nt + normalized.topology.loop_length_nt,
                    ),
                    stem3p_span=SpanContract(
                        start=normalized.topology.stem_length_nt + normalized.topology.loop_length_nt,
                        end=normalized.topology.cassette_length_nt,
                    ),
                    pair_map=normalized.topology.pair_map,
                )

    warnings: list[str] = []
    if normalized.schema_version == 1:
        warnings.append("schema_version 1 uses legacy coordinate semantics; schema_version 2 is recommended.")

    status = "satisfied" if not issues and candidate is not None else "unsatisfied"
    return CassetteEvaluationReport(
        status=status,
        spec_name=spec.name,
        target_strand=normalized.nicking.target_strand,
        workspace_root=str(workspace_root),
        spec_path=str(spec_path),
        catalog_path=str(catalog_path),
        metadata=CassetteReportMetadata(
            spec_schema_version=normalized.schema_version,
            coordinate_semantics=normalized.coordinate_semantics,
            left_flank_length=len(normalized.construct_context.left_flank),
            right_flank_length=len(normalized.construct_context.right_flank),
            evaluation_primary_length=len(normalized.construct_context.evaluation_primary_sequence),
            catalog_variants=_catalog_variants_for_report(spec=normalized, catalog_by_id=catalog_by_id),
            warnings=warnings,
        ),
        issues=issues,
        candidate=candidate,
    )


def render_markdown_report(report: CassetteEvaluationReport) -> str:
    return _markdown_report(report)
