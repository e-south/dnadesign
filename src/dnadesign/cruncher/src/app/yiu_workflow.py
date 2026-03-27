"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/app/yiu_workflow.py

Explicit YIU workflow validation, trace, and deterministic artifact materialization.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dnadesign.cruncher.bio import (
    derive_cut_geometry,
    iupac_bases_for_symbol,
    iupac_symbols_compatible,
    longest_reverse_complement_overlap,
    motif_matches,
    normalize_iupac,
    reverse_complement_iupac,
    sequence_contains_iupac,
)
from dnadesign.cruncher.yiu.artifacts import (
    STATE_VIEW_SCHEMA_VERSION,
    annotations_path,
    build_run_dir,
    catalog_fingerprint,
    design_id,
    fragments_path,
    input_fingerprint,
    parts_path,
    prepare_run_dir,
    published_views_dir,
    published_views_manifest_path,
    report_path,
    resolve_code_revision,
    state_view_path,
    status_path,
    trace_path,
    write_csv,
    write_manifest,
    write_published_views_manifest,
    write_report,
    write_status,
    write_trace,
    write_trace_manifest,
)
from dnadesign.cruncher.yiu.catalog import LoadedYiuCatalogs, load_yiu_catalogs
from dnadesign.cruncher.yiu.load import load_yiu_spec
from dnadesign.cruncher.yiu.models import (
    EnzymeSiteSpec,
    ProjectedRegion,
    ProjectedRegionPart,
    RegionSpec,
    RegionSpecV2,
    YiuOligoPartCatalogEntry,
    YiuPatternEvidenceSummary,
    YiuProcessSpec,
    YiuProcessSpecV2,
    YiuReportMetadata,
    YiuStateRecord,
    YiuValidationIssue,
    YiuValidationReport,
)


def _region_lookup(spec: YiuProcessSpec) -> dict[str, RegionSpec]:
    regions = (
        spec.source_oligo.payload_windows
        + spec.source_oligo.homology_windows
        + spec.source_oligo.retained_regions
        + spec.source_oligo.sacrificial_regions
    )
    return {region.id: region for region in regions}


def _primer_lookup(spec: YiuProcessSpec) -> dict[str, RegionSpec]:
    return {site.id: site for site in spec.source_oligo.primer_sites}


def _restriction_lookup(spec: YiuProcessSpec) -> dict[str, EnzymeSiteSpec]:
    return {site.id: site for site in spec.source_oligo.restriction_sites}


def _nickase_lookup(spec: YiuProcessSpec) -> dict[str, EnzymeSiteSpec]:
    return {site.id: site for site in spec.source_oligo.nickase_sites}


@dataclass(frozen=True)
class _StateSegment:
    segment_id: str
    source_start: int
    source_end: int
    state_start: int
    state_end: int


@dataclass(frozen=True)
class _StickyEndMatch:
    paired_nt: int
    left_start: int
    left_end: int
    right_start: int
    right_end: int
    unpaired_tail_nt: int
    bulge_nt: int = 0
    bulge_side: str | None = None


def _annotation_collections(
    spec: YiuProcessSpec | YiuProcessSpecV2,
) -> tuple[tuple[str, list[RegionSpec | EnzymeSiteSpec]], ...]:
    if isinstance(spec, YiuProcessSpecV2):
        return (
            ("primer_binding_core", list(spec.source_oligo.annotations.primer_binding_cores)),
            ("nickase_site", list(spec.source_oligo.annotations.nickase_sites)),
            ("payload_window", list(spec.source_oligo.annotations.payload_windows)),
            ("homology_window", list(spec.source_oligo.annotations.homology_windows)),
            ("retained_region", list(spec.source_oligo.annotations.retained_regions)),
            ("sacrificial_region", list(spec.source_oligo.annotations.sacrificial_regions)),
        )
    return (
        ("primer_site", spec.source_oligo.primer_sites),
        ("restriction_site", spec.source_oligo.restriction_sites),
        ("nickase_site", spec.source_oligo.nickase_sites),
        ("payload_window", spec.source_oligo.payload_windows),
        ("homology_window", spec.source_oligo.homology_windows),
        ("retained_region", spec.source_oligo.retained_regions),
        ("sacrificial_region", spec.source_oligo.sacrificial_regions),
    )


def _item_end(item: RegionSpec | EnzymeSiteSpec) -> int:
    return item.end if isinstance(item, RegionSpec) else item.end


def _projected_annotations(spec: YiuProcessSpec, *, interval_start: int, interval_end: int) -> list[dict[str, Any]]:
    projected: list[dict[str, Any]] = []
    for category, collection in _annotation_collections(spec):
        for item in collection:
            end = _item_end(item)
            if item.start < interval_start or end > interval_end:
                continue
            row: dict[str, Any] = {
                "category": category,
                "id": item.id,
                "start": item.start - interval_start,
                "end": end - interval_start,
                "source_start": item.start,
                "source_end": end,
                "label": getattr(item, "enzyme", item.id),
            }
            if isinstance(item, RegionSpec):
                row["strand"] = item.strand
            if isinstance(item, EnzymeSiteSpec):
                row["orientation"] = item.orientation
            projected.append(row)
    return projected


def _sequence_for_region(sequence: str, region: RegionSpec) -> str:
    return sequence[region.start : region.end]


def _ranges_overlap(left_start: int, left_end: int, right_start: int, right_end: int) -> bool:
    return left_start < right_end and right_start < left_end


def _overlap(left: RegionSpec, right: RegionSpec) -> bool:
    return _ranges_overlap(left.start, left.end, right.start, right.end)


def _issue(code: str, message: str, *, step_id: str | None = None, state_id: str | None = None) -> YiuValidationIssue:
    return YiuValidationIssue(code=code, message=message, step_id=step_id, state_id=state_id)


def _resolve_region(
    regions: dict[str, RegionSpec],
    region_id: str,
    *,
    code: str,
    label: str,
    step_id: str,
    issues: list[YiuValidationIssue],
) -> RegionSpec | None:
    region = regions.get(str(region_id))
    if region is None:
        issues.append(_issue(code, f"{label} references unknown region {region_id!r}", step_id=step_id))
    return region


def _resolve_region_list(
    regions: dict[str, RegionSpec],
    region_ids: list[str],
    *,
    code: str,
    label: str,
    step_id: str,
    issues: list[YiuValidationIssue],
) -> list[RegionSpec]:
    resolved: list[RegionSpec] = []
    for region_id in region_ids:
        region = _resolve_region(
            regions,
            region_id,
            code=code,
            label=label,
            step_id=step_id,
            issues=issues,
        )
        if region is not None:
            resolved.append(region)
    return resolved


def _project_region_to_state(
    region: RegionSpec,
    *,
    interval_start: int,
    interval_end: int,
    code: str,
    label: str,
    step_id: str,
    issues: list[YiuValidationIssue],
) -> RegionSpec | None:
    if region.start < interval_start or region.end > interval_end:
        issues.append(
            _issue(
                code,
                f"{label} {region.id} falls outside current state interval {interval_start}:{interval_end}",
                step_id=step_id,
            )
        )
        return None
    return RegionSpec(
        id=region.id,
        start=region.start - interval_start,
        end=region.end - interval_start,
        strand=region.strand,
    )


def _segments_for_source_regions(regions: list[RegionSpec]) -> list[_StateSegment]:
    cursor = 0
    segments: list[_StateSegment] = []
    for region in regions:
        length = region.end - region.start
        if length <= 0:
            continue
        segments.append(
            _StateSegment(
                segment_id=region.id,
                source_start=region.start,
                source_end=region.end,
                state_start=cursor,
                state_end=cursor + length,
            )
        )
        cursor += length
    return segments


def _project_region_to_segments(
    region: RegionSpec,
    segments: list[_StateSegment],
    *,
    code: str,
    label: str,
    state_id: str,
    step_id: str,
    issues: list[YiuValidationIssue],
) -> ProjectedRegion | None:
    parts: list[ProjectedRegionPart] = []
    for segment in segments:
        overlap_start = max(region.start, segment.source_start)
        overlap_end = min(region.end, segment.source_end)
        if overlap_start >= overlap_end:
            continue
        parts.append(
            ProjectedRegionPart(
                segment_id=segment.segment_id,
                start=segment.state_start + (overlap_start - segment.source_start),
                end=segment.state_start + (overlap_end - segment.source_start),
            )
        )
    if not parts:
        issues.append(
            _issue(
                code,
                f"{label} {region.id} is not preserved within the current state segments",
                step_id=step_id,
            )
        )
        return None
    return ProjectedRegion(
        id=f"{state_id}:{region.id}",
        source_region_id=region.id,
        state_id=state_id,
        spans_junction=len(parts) > 1,
        parts=parts,
    )


def _projected_region_sequence(sequence: str, projected: ProjectedRegion) -> str:
    return "".join(sequence[part.start : part.end] for part in projected.parts)


def _projected_region_payload(
    projected: ProjectedRegion,
    *,
    source_region: RegionSpec,
    sequence: str,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": source_region.id,
        "source_start": source_region.start,
        "source_end": source_region.end,
        "sequence": _projected_region_sequence(sequence, projected),
        "spans_junction": projected.spans_junction,
        "parts": [part.model_dump(mode="json") for part in projected.parts],
    }
    if len(projected.parts) == 1:
        payload["state_start"] = projected.parts[0].start
        payload["state_end"] = projected.parts[0].end
    return payload


def _projected_region_overlaps_interval(
    projected: ProjectedRegion,
    *,
    start: int,
    end: int,
) -> bool:
    return any(_ranges_overlap(part.start, part.end, start, end) for part in projected.parts)


def _segments_for_projected_regions(
    regions: list[RegionSpec],
    *,
    projected_by_id: dict[str, ProjectedRegion],
) -> list[_StateSegment]:
    cursor = 0
    segments: list[_StateSegment] = []
    for region in regions:
        projected = projected_by_id.get(region.id)
        if projected is None:
            continue
        length = sum(part.end - part.start for part in projected.parts)
        if length <= 0:
            continue
        segments.append(
            _StateSegment(
                segment_id=region.id,
                source_start=region.start,
                source_end=region.end,
                state_start=cursor,
                state_end=cursor + length,
            )
        )
        cursor += length
    return segments


def _joined_region_segments(regions: list[RegionSpec], *, sequence: str) -> list[dict[str, Any]]:
    cursor = 0
    rows: list[dict[str, Any]] = []
    for region in regions:
        segment_sequence = _sequence_for_region(sequence, region)
        rows.append(
            {
                "id": region.id,
                "source_start": region.start,
                "source_end": region.end,
                "payload_start": cursor,
                "payload_end": cursor + len(segment_sequence),
                "sequence": segment_sequence,
            }
        )
        cursor += len(segment_sequence)
    return rows


def _branched_state_arms(*, retained_product: str, adapter_sequence: str) -> list[dict[str, Any]]:
    payload_length = len(retained_product)
    return [
        {
            "id": "retained_product",
            "role": "payload",
            "state_start": 0,
            "state_end": payload_length,
            "sequence": retained_product,
        },
        {
            "id": "y_adapter",
            "role": "adapter",
            "state_start": payload_length + 1,
            "state_end": payload_length + 1 + len(adapter_sequence),
            "sequence": adapter_sequence,
        },
    ]


_AMBIGUOUS_IUPAC_SYMBOLS = frozenset("RYSWKMBDHVN")


def _sequence_mode_for_values(*sequences: str | None, pattern_label: str = "iupac_pattern") -> str:
    for sequence in sequences:
        if not sequence:
            continue
        for symbol in sequence.upper():
            if not symbol.isalpha():
                continue
            if symbol in _AMBIGUOUS_IUPAC_SYMBOLS:
                return pattern_label
    return "concrete"


def _state_topology(state: YiuStateRecord) -> str:
    if state.topology_kind:
        return state.topology_kind
    topology = state.metadata.get("topology")
    if isinstance(topology, str) and topology:
        return topology
    return "linear_dsdna" if state.complement_sequence is not None else "linear_ssdna"


def _state(
    *,
    state_id: str,
    step_id: str,
    kind: str,
    status: str,
    primary_sequence: str | None,
    complement_sequence: str | None = None,
    metadata: dict[str, Any] | None = None,
    state_kind: str | None = None,
    topology_kind: str | None = None,
    view_contract_version: int | None = None,
    segments: list[dict[str, Any]] | None = None,
    annotations: list[dict[str, Any]] | None = None,
    cuts: list[dict[str, Any]] | None = None,
    junctions: list[dict[str, Any]] | None = None,
    fragments: list[dict[str, Any]] | None = None,
    pattern_evidence_summary: YiuPatternEvidenceSummary | None = None,
    pattern_label: str = "iupac_pattern",
) -> YiuStateRecord:
    return YiuStateRecord(
        state_id=state_id,
        step_id=step_id,
        kind=kind,
        status=status,  # type: ignore[arg-type]
        sequence_mode=_sequence_mode_for_values(
            primary_sequence,
            complement_sequence,
            pattern_label=pattern_label,
        ),  # type: ignore[arg-type]
        view_contract_version=view_contract_version,
        state_kind=state_kind or kind,
        topology_kind=topology_kind,
        primary_sequence=primary_sequence,
        complement_sequence=complement_sequence,
        segments=segments or [],
        annotations=annotations or [],
        cuts=cuts or [],
        junctions=junctions or [],
        fragments=fragments or [],
        pattern_evidence_summary=pattern_evidence_summary or YiuPatternEvidenceSummary(),
        metadata=metadata or {},
    )


def _compatible_sequence(left: str, right: str) -> bool:
    if len(left) != len(right):
        return False
    return all(iupac_symbols_compatible(lhs, rhs) for lhs, rhs in zip(left, right, strict=True))


def _match_sort_key(match: _StickyEndMatch) -> tuple[int, int, int, int, int, int]:
    return (
        -match.paired_nt,
        match.unpaired_tail_nt,
        match.bulge_nt,
        match.left_start,
        match.right_start,
        0 if match.bulge_side is None else (1 if match.bulge_side == "left" else 2),
    )


def _best_contiguous_sticky_end_match(left: str, right: str) -> _StickyEndMatch | None:
    candidates: list[_StickyEndMatch] = []
    for shift in range(-len(right) + 1, len(left)):
        left_start = max(0, shift)
        right_start = max(0, -shift)
        paired_nt = min(len(left) - left_start, len(right) - right_start)
        if paired_nt <= 0:
            continue
        if not _compatible_sequence(
            left[left_start : left_start + paired_nt],
            right[right_start : right_start + paired_nt],
        ):
            continue
        candidates.append(
            _StickyEndMatch(
                paired_nt=paired_nt,
                left_start=left_start,
                left_end=left_start + paired_nt,
                right_start=right_start,
                right_end=right_start + paired_nt,
                unpaired_tail_nt=(len(left) + len(right)) - (2 * paired_nt),
            )
        )
    if not candidates:
        return None
    return sorted(candidates, key=_match_sort_key)[0]


def _best_bulged_sticky_end_match(left: str, right: str, *, max_bulge_nt: int) -> _StickyEndMatch | None:
    candidates: list[_StickyEndMatch] = []
    for bulge_side in ("left", "right"):
        gapped_text = left if bulge_side == "left" else right
        if len(gapped_text) < 3:
            continue
        for bulge_nt in range(1, min(max_bulge_nt, len(gapped_text) - 2) + 1):
            for bulge_start in range(1, len(gapped_text) - bulge_nt):
                if bulge_side == "left":
                    compressed_left = left[:bulge_start] + left[bulge_start + bulge_nt :]
                    match = _best_contiguous_sticky_end_match(compressed_left, right)
                    if match is None or not (match.left_start < bulge_start < match.left_end):
                        continue
                    left_end = match.left_end + bulge_nt if match.left_end > bulge_start else match.left_end
                    right_end = match.right_end
                else:
                    compressed_right = right[:bulge_start] + right[bulge_start + bulge_nt :]
                    match = _best_contiguous_sticky_end_match(left, compressed_right)
                    if match is None or not (match.right_start < bulge_start < match.right_end):
                        continue
                    left_end = match.left_end
                    right_end = match.right_end + bulge_nt if match.right_end > bulge_start else match.right_end
                total_unpaired_nt = (len(left) + len(right)) - (2 * match.paired_nt)
                candidates.append(
                    _StickyEndMatch(
                        paired_nt=match.paired_nt,
                        left_start=match.left_start,
                        left_end=left_end,
                        right_start=match.right_start,
                        right_end=right_end,
                        unpaired_tail_nt=max(0, total_unpaired_nt - bulge_nt),
                        bulge_nt=bulge_nt,
                        bulge_side=bulge_side,
                    )
                )
    if not candidates:
        return None
    return sorted(candidates, key=_match_sort_key)[0]


def _iupac_match_status(sequence: str, motif: str) -> str:
    sequence_text = normalize_iupac(sequence)
    motif_text = normalize_iupac(motif)
    if len(sequence_text) != len(motif_text):
        return "impossible"
    saw_possible = False
    for observed, required in zip(sequence_text, motif_text, strict=True):
        observed_set = iupac_bases_for_symbol(observed)
        required_set = iupac_bases_for_symbol(required)
        if not observed_set & required_set:
            return "impossible"
        if not observed_set <= required_set:
            saw_possible = True
    return "possible" if saw_possible else "guaranteed"


def _sequence_contains_status(sequence: str, motif: str) -> str:
    sequence_text = normalize_iupac(sequence)
    motif_text = normalize_iupac(motif)
    window = len(motif_text)
    if window == 0 or window > len(sequence_text):
        return "impossible"
    saw_possible = False
    for idx in range(0, len(sequence_text) - window + 1):
        status = _iupac_match_status(sequence_text[idx : idx + window], motif_text)
        if status == "guaranteed":
            return status
        if status == "possible":
            saw_possible = True
    return "possible" if saw_possible else "impossible"


def _pattern_summary(statuses: list[str]) -> YiuPatternEvidenceSummary:
    counts = {"guaranteed": 0, "possible": 0, "impossible": 0}
    for status in statuses:
        counts[str(status)] += 1
    return YiuPatternEvidenceSummary(
        guaranteed_checks=counts["guaranteed"],
        possible_checks=counts["possible"],
        impossible_checks=counts["impossible"],
    )


def _pattern_policy_issue(
    *,
    status: str,
    policy: str,
    label: str,
    step_id: str,
    issues: list[YiuValidationIssue],
) -> None:
    if status == "guaranteed":
        return
    if status == "possible" and policy == "allow_possible_with_warning":
        issues.append(
            YiuValidationIssue(
                code="PATTERN_CHECK_NOT_GUARANTEED",
                message=f"{label} is only possible under the current IUPAC pattern",
                step_id=step_id,
                severity="warning",
            )
        )
        return
    if status == "possible":
        issues.append(
            _issue(
                "PATTERN_CHECK_NOT_GUARANTEED",
                f"{label} is only possible under the current IUPAC pattern",
                step_id=step_id,
            )
        )
        return
    issues.append(
        _issue(
            "PATTERN_CHECK_IMPOSSIBLE",
            f"{label} is impossible under the current IUPAC pattern",
            step_id=step_id,
        )
    )


def _terminal_tails(match: _StickyEndMatch, *, left_length: int, right_length: int) -> tuple[int, int]:
    return match.left_start + match.right_start, (left_length - match.left_end) + (right_length - match.right_end)


def _evaluate_ligation_compatibility(
    left: str,
    right: str,
    *,
    mode: str,
    partial_rule: dict[str, Any] | None = None,
    bulged_rule: dict[str, Any] | None = None,
) -> _StickyEndMatch | None:
    aligned_right = reverse_complement_iupac(right) if right else ""
    if mode == "exact_complement":
        if left and aligned_right and len(left) == len(aligned_right) and _compatible_sequence(left, aligned_right):
            return _StickyEndMatch(
                paired_nt=len(left),
                left_start=0,
                left_end=len(left),
                right_start=0,
                right_end=len(aligned_right),
                unpaired_tail_nt=0,
            )
        return None
    if mode == "partial_complement":
        rule = partial_rule or {}
        candidate = _best_contiguous_sticky_end_match(left, aligned_right)
        if candidate is None:
            return None
        left_tail_nt, right_tail_nt = _terminal_tails(candidate, left_length=len(left), right_length=len(aligned_right))
        if candidate.bulge_nt != 0:
            return None
        if candidate.paired_nt < int(rule.get("min_paired_nt", 1)):
            return None
        if left_tail_nt and not bool(rule.get("allow_left_tail", True)):
            return None
        if right_tail_nt and not bool(rule.get("allow_right_tail", True)):
            return None
        return candidate
    rule = bulged_rule or {}
    max_bulge_nt = int(rule.get("max_bulge_nt", 1))
    candidates = [
        candidate
        for candidate in (
            _best_bulged_sticky_end_match(left, aligned_right, max_bulge_nt=max_bulge_nt),
            _best_contiguous_sticky_end_match(left, aligned_right),
        )
        if candidate is not None
    ]
    for candidate in sorted(candidates, key=_match_sort_key):
        left_tail_nt, right_tail_nt = _terminal_tails(candidate, left_length=len(left), right_length=len(aligned_right))
        if candidate.bulge_nt > max_bulge_nt:
            continue
        if candidate.bulge_nt > 0 and (
            candidate.left_start + candidate.paired_nt < int(rule.get("min_left_paired_nt", 1))
        ):
            continue
        if candidate.bulge_nt > 0 and (
            candidate.right_start + candidate.paired_nt < int(rule.get("min_right_paired_nt", 1))
        ):
            continue
        if not bool(rule.get("allow_terminal_tails", True)) and (left_tail_nt or right_tail_nt):
            continue
        return candidate
    return None


def _v2_primer_core_lookup(spec: YiuProcessSpecV2) -> dict[str, RegionSpec]:
    return {site.id: site for site in spec.source_oligo.annotations.primer_binding_cores}


def _v2_region_lookup(spec: YiuProcessSpecV2) -> dict[str, RegionSpecV2]:
    regions = (
        spec.source_oligo.annotations.payload_windows
        + spec.source_oligo.annotations.homology_windows
        + spec.source_oligo.annotations.retained_regions
        + spec.source_oligo.annotations.sacrificial_regions
    )
    return {region.id: region for region in regions}


def _v2_nickase_lookup(spec: YiuProcessSpecV2) -> dict[str, EnzymeSiteSpec]:
    return {site.id: site for site in spec.source_oligo.annotations.nickase_sites}


def _v2_part(catalogs: LoadedYiuCatalogs, part_id: str, *, label: str) -> YiuOligoPartCatalogEntry:
    part = catalogs.oligo_parts.get(part_id)
    if part is None:
        raise ValueError(f"{label} {part_id!r} is not present in catalogs.oligo_parts")
    return part


def _v2_overlap_issues(spec: YiuProcessSpecV2) -> list[YiuValidationIssue]:
    issues: list[YiuValidationIssue] = []
    annotations = spec.source_oligo.annotations
    source_state_id = "source_oligo_ssdna"

    for retained in annotations.retained_regions:
        for sacrificial in annotations.sacrificial_regions:
            if _overlap(retained, sacrificial):
                issues.append(
                    _issue(
                        "RETAINED_SACRIFICIAL_OVERLAP",
                        f"retained region {retained.id} overlaps sacrificial region {sacrificial.id}",
                        state_id=source_state_id,
                    )
                )
    for payload_window in annotations.payload_windows:
        for sacrificial in annotations.sacrificial_regions:
            if _overlap(payload_window, sacrificial):
                issues.append(
                    _issue(
                        "PAYLOAD_SACRIFICIAL_OVERLAP",
                        f"payload window {payload_window.id} overlaps sacrificial region {sacrificial.id}",
                        state_id=source_state_id,
                    )
                )
    for primer_core in annotations.primer_binding_cores:
        for site in (*annotations.restriction_sites, *annotations.nickase_sites):
            if _ranges_overlap(primer_core.start, primer_core.end, site.start, site.end):
                issues.append(
                    _issue(
                        "PRIMER_CORE_SITE_OVERLAP",
                        f"primer binding core {primer_core.id} overlaps site {site.id}",
                        state_id=source_state_id,
                    )
                )
    return issues


def _validate_annotation_overlaps(spec: YiuProcessSpec) -> list[YiuValidationIssue]:
    issues: list[YiuValidationIssue] = []
    source_state_id = "source_oligo_ssdna"

    generic_overlap_collections = (
        spec.source_oligo.primer_sites
        + spec.source_oligo.payload_windows
        + spec.source_oligo.restriction_sites
        + spec.source_oligo.nickase_sites
    )
    for index, left in enumerate(generic_overlap_collections):
        left_region = RegionSpec(id=left.id, start=left.start, end=_item_end(left))
        for right in generic_overlap_collections[index + 1 :]:
            right_region = RegionSpec(id=right.id, start=right.start, end=_item_end(right))
            if _overlap(left_region, right_region):
                issues.append(
                    _issue(
                        "ANNOTATION_OVERLAP",
                        f"annotations {left.id} and {right.id} overlap on the source oligo",
                        state_id=source_state_id,
                    )
                )

    for retained in spec.source_oligo.retained_regions:
        for sacrificial in spec.source_oligo.sacrificial_regions:
            if _overlap(retained, sacrificial):
                issues.append(
                    _issue(
                        "RETAINED_SACRIFICIAL_OVERLAP",
                        f"retained region {retained.id} overlaps sacrificial region {sacrificial.id}",
                        state_id=source_state_id,
                    )
                )

    for collection, code, label in (
        (spec.source_oligo.retained_regions, "RETAINED_REGION_PARTIAL_OVERLAP", "retained region"),
        (spec.source_oligo.sacrificial_regions, "SACRIFICIAL_REGION_PARTIAL_OVERLAP", "sacrificial region"),
    ):
        for index, left in enumerate(collection):
            for right in collection[index + 1 :]:
                if not _overlap(left, right):
                    continue
                if left.start == right.start and left.end == right.end:
                    continue
                issues.append(
                    _issue(
                        code,
                        f"{label} {left.id} partially overlaps {right.id}",
                        state_id=source_state_id,
                    )
                )
    return issues


def _catalog_site_issue(
    *,
    site: EnzymeSiteSpec,
    catalog_entry: Any | None,
    missing_code: str,
    mismatch_code: str,
    step_id: str,
    issues: list[YiuValidationIssue],
) -> None:
    if catalog_entry is None:
        issues.append(
            _issue(
                missing_code,
                f"enzyme {site.enzyme!r} for site {site.id} is not present in the referenced catalog",
                step_id=step_id,
            )
        )
        return
    if catalog_entry.recognition_sequence != site.recognition_sequence:
        issues.append(
            _issue(
                mismatch_code,
                f"site {site.id} recognition sequence {site.recognition_sequence!r} does not match catalog value "
                f"{catalog_entry.recognition_sequence!r} for enzyme {site.enzyme!r}",
                step_id=step_id,
            )
        )
    for field_name in ("top_cut_offset", "bottom_cut_offset"):
        catalog_value = getattr(catalog_entry, field_name)
        site_value = getattr(site, field_name)
        if catalog_value is not None and site_value is not None and catalog_value != site_value:
            issues.append(
                _issue(
                    mismatch_code,
                    f"site {site.id} {field_name}={site_value} does not match catalog value {catalog_value} "
                    f"for enzyme {site.enzyme!r}",
                    step_id=step_id,
                )
            )


def _resolve_adapter_sequence(
    spec: YiuProcessSpec,
    step_adapter_sequence: str | None,
    *,
    step_id: str,
    catalogs: LoadedYiuCatalogs,
    issues: list[YiuValidationIssue],
) -> str:
    inline_policy_sequence = spec.adapter_policy.adapter_sequence
    inline_step_sequence = step_adapter_sequence
    resolved_sequence = str(inline_step_sequence or inline_policy_sequence or "")
    adapter_id = spec.adapter_policy.y_adapter_id

    if adapter_id is None:
        if inline_policy_sequence and inline_step_sequence and inline_policy_sequence != inline_step_sequence:
            issues.append(
                _issue(
                    "ADAPTER_SEQUENCE_MISMATCH",
                    f"step adapter sequence {inline_step_sequence!r} does not match adapter_policy.adapter_sequence "
                    f"{inline_policy_sequence!r}",
                    step_id=step_id,
                )
            )
        return resolved_sequence

    if spec.catalogs.adapters is None:
        issues.append(
            _issue(
                "ADAPTER_CATALOG_REQUIRED",
                f"adapter_policy.y_adapter_id {adapter_id!r} requires catalogs.adapters",
                step_id=step_id,
            )
        )
        return resolved_sequence

    catalog_entry = catalogs.adapters.get(adapter_id)
    if catalog_entry is None:
        issues.append(
            _issue(
                "ADAPTER_CATALOG_ENTRY_MISSING",
                f"adapter id {adapter_id!r} is not present in the referenced adapter catalog",
                step_id=step_id,
            )
        )
        return resolved_sequence

    catalog_sequence = catalog_entry.sequence
    for source_label, candidate in (
        ("adapter_policy.adapter_sequence", inline_policy_sequence),
        ("step.adapter_sequence", inline_step_sequence),
    ):
        if candidate is not None and candidate != catalog_sequence:
            issues.append(
                _issue(
                    "ADAPTER_CATALOG_MISMATCH",
                    f"{source_label} {candidate!r} does not match adapter catalog sequence {catalog_sequence!r} "
                    f"for adapter {adapter_id!r}",
                    step_id=step_id,
                )
            )
    return catalog_sequence


def _build_yiu_report_v1(spec: YiuProcessSpec, *, catalogs: LoadedYiuCatalogs | None = None) -> YiuValidationReport:
    catalogs = catalogs or LoadedYiuCatalogs(restriction_enzymes={}, nickases={}, adapters={}, paths=())
    issues: list[YiuValidationIssue] = []
    states: list[YiuStateRecord] = []
    source_sequence = spec.source_oligo.sequence
    regions = _region_lookup(spec)
    primers = _primer_lookup(spec)
    restriction_sites = _restriction_lookup(spec)
    nickase_sites = _nickase_lookup(spec)
    issues.extend(_validate_annotation_overlaps(spec))

    states.append(
        _state(
            state_id="source_oligo_ssdna",
            step_id="source_oligo_ssdna",
            kind="source_oligo_ssdna",
            status="unsatisfied" if issues else "satisfied",
            primary_sequence=source_sequence,
            metadata={"length_nt": len(source_sequence)},
        )
    )

    current_primary = source_sequence
    current_complement: str | None = None
    current_interval_start = 0
    current_interval_end = len(source_sequence)
    current_segments = _segments_for_source_regions([RegionSpec(id="source", start=0, end=len(source_sequence))])
    assembled_payload = ""
    retained_product = ""
    adapter_sequence = spec.adapter_policy.adapter_sequence or ""
    digest_left_overhang = ""
    digest_right_overhang = ""
    fragment_lengths: list[int] = []

    for step in spec.step_graph.steps:
        step_issues: list[YiuValidationIssue] = []
        metadata: dict[str, Any] = {}
        state_primary: str | None = None
        state_complement: str | None = None
        if step.kind == "pcr":
            forward = primers.get(str(step.forward_primer_site))
            reverse = primers.get(str(step.reverse_primer_site))
            amplicon_start = forward.start if forward is not None else None
            amplicon_end = reverse.end if reverse is not None else None
            if forward is None or reverse is None:
                step_issues.append(
                    _issue("PCR_PRIMER_SITE_MISSING", "PCR primer site reference is missing", step_id=step.id)
                )
            elif forward.start >= reverse.start:
                step_issues.append(
                    _issue("PCR_BOUNDARY_INVALID", "forward primer must start before reverse primer", step_id=step.id)
                )
            if amplicon_start is not None and amplicon_end is not None:
                current_primary = source_sequence[amplicon_start:amplicon_end]
                current_complement = reverse_complement_iupac(current_primary)
                current_interval_start = amplicon_start
                current_interval_end = amplicon_end
                current_segments = _segments_for_source_regions(
                    [RegionSpec(id="pcr_interval", start=amplicon_start, end=amplicon_end)]
                )
                state_primary = current_primary
                state_complement = current_complement
                for category, collection in (
                    ("restriction_site", spec.source_oligo.restriction_sites),
                    ("nickase_site", spec.source_oligo.nickase_sites),
                    ("payload_window", spec.source_oligo.payload_windows),
                    ("homology_window", spec.source_oligo.homology_windows),
                    ("retained_region", spec.source_oligo.retained_regions),
                    ("sacrificial_region", spec.source_oligo.sacrificial_regions),
                ):
                    for item in collection:
                        end = item.end if hasattr(item, "end") else item.start + len(item.recognition_sequence)
                        if item.start < amplicon_start or end > amplicon_end:
                            step_issues.append(
                                _issue(
                                    "PCR_AMPLICON_EXCLUDES_ANNOTATION",
                                    f"{category} {item.id} falls outside PCR amplicon {amplicon_start}:{amplicon_end}",
                                    step_id=step.id,
                                )
                            )
            metadata = {
                "forward_primer_site": step.forward_primer_site,
                "reverse_primer_site": step.reverse_primer_site,
                "amplicon_start": amplicon_start,
                "amplicon_end": amplicon_end,
                "amplicon_length_nt": (
                    len(current_primary) if amplicon_start is not None and amplicon_end is not None else 0
                ),
                "projected_annotations": (
                    _projected_annotations(spec, interval_start=amplicon_start, interval_end=amplicon_end)
                    if amplicon_start is not None and amplicon_end is not None
                    else []
                ),
            }
        elif step.kind == "restriction_digest":
            digest_input_primary = current_primary
            left_site = restriction_sites.get(str(step.left_site))
            right_site = restriction_sites.get(str(step.right_site))
            left_geometry = right_geometry = None
            if left_site is None or right_site is None:
                step_issues.append(
                    _issue("DIGEST_SITE_MISSING", "restriction digest site reference is missing", step_id=step.id)
                )
            else:
                for site_id, site in (("left", left_site), ("right", right_site)):
                    site_start = site.start - current_interval_start
                    if site_start < 0 or site.end > current_interval_end:
                        step_issues.append(
                            _issue(
                                "RESTRICTION_SITE_EXCLUDED_FROM_CURRENT_STATE",
                                f"{site_id} restriction site {site.id} falls outside current PCR state "
                                f"{current_interval_start}:{current_interval_end}",
                                step_id=step.id,
                            )
                        )
                        continue
                    if spec.catalogs.restriction_enzymes is not None:
                        _catalog_site_issue(
                            site=site,
                            catalog_entry=catalogs.restriction_enzymes.get(site.enzyme),
                            missing_code="RESTRICTION_CATALOG_ENTRY_MISSING",
                            mismatch_code="RESTRICTION_CATALOG_MISMATCH",
                            step_id=step.id,
                            issues=step_issues,
                        )
                    try:
                        geometry = derive_cut_geometry(
                            digest_input_primary,
                            start=site_start,
                            recognition_sequence=site.recognition_sequence,
                            orientation=site.orientation,
                            top_cut_offset=site.top_cut_offset,
                            bottom_cut_offset=site.bottom_cut_offset,
                        )
                    except ValueError as exc:
                        step_issues.append(
                            _issue(
                                "RESTRICTION_SITE_MISMATCH",
                                f"{site_id} restriction site {site.id} is invalid: {exc}",
                                step_id=step.id,
                            )
                        )
                        continue
                    if site_id == "left":
                        left_geometry = geometry
                    else:
                        right_geometry = geometry
                    if site_id == "left":
                        digest_left_overhang = geometry.overhang_sequence
                        expected = step.expected_left_overhang
                    else:
                        digest_right_overhang = geometry.overhang_sequence
                        expected = step.expected_right_overhang
                    if expected is not None and geometry.overhang_sequence != expected:
                        step_issues.append(
                            _issue(
                                "DIGEST_OVERHANG_MISMATCH",
                                f"{site_id} digest overhang {geometry.overhang_sequence!r} != expected {expected!r}",
                                step_id=step.id,
                            )
                        )
                if left_geometry is not None and right_geometry is not None:
                    left_primary_boundary = (
                        left_geometry.top_boundary
                        if left_geometry.top_boundary is not None
                        else left_geometry.bottom_boundary
                    )
                    right_primary_boundary = (
                        right_geometry.top_boundary
                        if right_geometry.top_boundary is not None
                        else right_geometry.bottom_boundary
                    )
                    left_complement_boundary = (
                        left_geometry.bottom_boundary
                        if left_geometry.bottom_boundary is not None
                        else left_geometry.top_boundary
                    )
                    right_complement_boundary = (
                        right_geometry.bottom_boundary
                        if right_geometry.bottom_boundary is not None
                        else right_geometry.top_boundary
                    )
                    if (
                        left_primary_boundary is None
                        or right_primary_boundary is None
                        or left_primary_boundary >= right_primary_boundary
                    ):
                        step_issues.append(
                            _issue(
                                "DIGEST_PRIMARY_BOUNDARY_INVALID",
                                "restriction digest did not yield an ordered primary-strand retained interval",
                                step_id=step.id,
                            )
                        )
                    else:
                        state_primary = digest_input_primary[left_primary_boundary:right_primary_boundary]
                        current_primary = state_primary
                        current_interval_start += left_primary_boundary
                        current_interval_end = current_interval_start + len(state_primary)
                        current_segments = _segments_for_source_regions(
                            [RegionSpec(id="digest_interval", start=current_interval_start, end=current_interval_end)]
                        )
                    if (
                        left_complement_boundary is None
                        or right_complement_boundary is None
                        or left_complement_boundary >= right_complement_boundary
                    ):
                        step_issues.append(
                            _issue(
                                "DIGEST_COMPLEMENT_BOUNDARY_INVALID",
                                "restriction digest did not yield an ordered complement-strand retained interval",
                                step_id=step.id,
                            )
                        )
                    else:
                        state_complement = reverse_complement_iupac(
                            digest_input_primary[left_complement_boundary:right_complement_boundary]
                        )
                        current_complement = state_complement
                    metadata = {
                        "left_overhang": digest_left_overhang,
                        "right_overhang": digest_right_overhang,
                        "left_primary_cut_boundary": left_primary_boundary,
                        "right_primary_cut_boundary": right_primary_boundary,
                        "left_complement_cut_boundary": left_complement_boundary,
                        "right_complement_cut_boundary": right_complement_boundary,
                        "removed_primary_flanks": (
                            []
                            if left_primary_boundary is None or right_primary_boundary is None
                            else [
                                {
                                    "start": start,
                                    "end": end,
                                    "length_nt": end - start,
                                }
                                for start, end in (
                                    (0, left_primary_boundary),
                                    (right_primary_boundary, len(digest_input_primary)),
                                )
                                if end > start
                            ]
                        ),
                        "projected_annotations": _projected_annotations(
                            spec,
                            interval_start=current_interval_start,
                            interval_end=current_interval_end,
                        )
                        if state_primary is not None
                        else [],
                    }
                else:
                    metadata = {
                        "left_overhang": digest_left_overhang,
                        "right_overhang": digest_right_overhang,
                    }
        elif step.kind == "circularization":
            paired_threshold = step.min_paired_nt if step.min_paired_nt is not None else 1
            max_unpaired_tail_nt = (
                step.max_unpaired_tail_nt
                if step.max_unpaired_tail_nt is not None
                else len(digest_left_overhang) + len(digest_right_overhang)
            )
            max_bulge_nt = step.max_bulge_nt if step.max_bulge_nt is not None else 1
            aligned_right_overhang = reverse_complement_iupac(digest_right_overhang) if digest_right_overhang else ""
            circularization_match: _StickyEndMatch | None = None
            if step.compatibility == "exact_complement":
                if (
                    digest_left_overhang
                    and aligned_right_overhang
                    and len(digest_left_overhang) == len(aligned_right_overhang)
                    and _compatible_sequence(digest_left_overhang, aligned_right_overhang)
                ):
                    circularization_match = _StickyEndMatch(
                        paired_nt=len(digest_left_overhang),
                        left_start=0,
                        left_end=len(digest_left_overhang),
                        right_start=0,
                        right_end=len(aligned_right_overhang),
                        unpaired_tail_nt=0,
                    )
                else:
                    step_issues.append(
                        _issue(
                            "CIRCULARIZATION_COMPATIBILITY_FAIL",
                            "left and right sticky ends are not exact reverse complements",
                            step_id=step.id,
                        )
                    )
            elif step.compatibility == "partial_complement":
                candidate = _best_contiguous_sticky_end_match(digest_left_overhang, aligned_right_overhang)
                if (
                    candidate is None
                    or candidate.paired_nt < paired_threshold
                    or candidate.unpaired_tail_nt > max_unpaired_tail_nt
                ):
                    step_issues.append(
                        _issue(
                            "CIRCULARIZATION_COMPATIBILITY_FAIL",
                            "partial-complement sticky ends do not satisfy the paired-core and tail-slack bounds",
                            step_id=step.id,
                        )
                    )
                else:
                    circularization_match = candidate
            else:
                valid_candidates = [
                    candidate
                    for candidate in (
                        _best_contiguous_sticky_end_match(digest_left_overhang, aligned_right_overhang),
                        _best_bulged_sticky_end_match(
                            digest_left_overhang,
                            aligned_right_overhang,
                            max_bulge_nt=max_bulge_nt,
                        ),
                    )
                    if candidate is not None
                    and candidate.paired_nt >= paired_threshold
                    and candidate.unpaired_tail_nt <= max_unpaired_tail_nt
                    and candidate.bulge_nt <= max_bulge_nt
                ]
                if not valid_candidates:
                    step_issues.append(
                        _issue(
                            "CIRCULARIZATION_COMPATIBILITY_FAIL",
                            "bulged sticky ends do not satisfy the paired-core, bulge, and tail-slack bounds",
                            step_id=step.id,
                        )
                    )
                else:
                    circularization_match = sorted(valid_candidates, key=_match_sort_key)[0]
            left_half = _resolve_region(
                regions,
                str(spec.payload_goal.left_half_ref),
                code="PAYLOAD_REGION_MISSING",
                label="payload_goal.left_half_ref",
                step_id=step.id,
                issues=step_issues,
            )
            right_half = _resolve_region(
                regions,
                str(spec.payload_goal.right_half_ref),
                code="PAYLOAD_REGION_MISSING",
                label="payload_goal.right_half_ref",
                step_id=step.id,
                issues=step_issues,
            )
            if left_half is not None and right_half is not None:
                assembled_payload = _sequence_for_region(source_sequence, left_half) + _sequence_for_region(
                    source_sequence, right_half
                )
                if not motif_matches(assembled_payload, spec.payload_goal.assembled_payload):
                    payload_goal = spec.payload_goal.assembled_payload
                    step_issues.append(
                        _issue(
                            "PAYLOAD_ASSEMBLY_MISMATCH",
                            f"assembled payload {assembled_payload!r} does not satisfy goal {payload_goal!r}",
                            step_id=step.id,
                        )
                    )
            payload_segments = _joined_region_segments(
                [region for region in (left_half, right_half) if region is not None],
                sequence=source_sequence,
            )
            state_primary = current_primary
            state_complement = current_complement
            metadata = {
                "assembled_payload": assembled_payload,
                "sticky_end_overlap": circularization_match.paired_nt if circularization_match is not None else 0,
                "compatibility": step.compatibility,
                "paired_nt": circularization_match.paired_nt if circularization_match is not None else 0,
                "min_paired_nt": paired_threshold,
                "max_unpaired_tail_nt": max_unpaired_tail_nt,
                "unpaired_tail_nt": (
                    circularization_match.unpaired_tail_nt if circularization_match is not None else None
                ),
                "max_bulge_nt": max_bulge_nt if step.compatibility == "bulged" else None,
                "bulge_nt": circularization_match.bulge_nt if circularization_match is not None else 0,
                "bulge_side": circularization_match.bulge_side if circularization_match is not None else None,
                "left_core_start": circularization_match.left_start if circularization_match is not None else None,
                "left_core_end": circularization_match.left_end if circularization_match is not None else None,
                "right_core_start": circularization_match.right_start if circularization_match is not None else None,
                "right_core_end": circularization_match.right_end if circularization_match is not None else None,
                "aligned_right_overhang": aligned_right_overhang,
                "topology": "circular_dsDNA",
                "payload_junction_segments": payload_segments,
                "payload_junction": {
                    "left_region_id": spec.payload_goal.left_half_ref,
                    "right_region_id": spec.payload_goal.right_half_ref,
                    "payload_join_index": payload_segments[0]["payload_end"] if len(payload_segments) >= 1 else 0,
                    "junction_rule": spec.payload_goal.junction_rule,
                },
            }
        elif step.kind == "exonuclease_selection":
            state_primary = current_primary
            state_complement = current_complement
            if spec.cleanup_policy.linear_depletion.enabled and not spec.cleanup_policy.linear_depletion.enzyme:
                step_issues.append(
                    _issue(
                        "LINEAR_DEPLETION_ENZYME_MISSING",
                        "cleanup_policy.linear_depletion.enzyme is required when linear depletion is enabled",
                        step_id=step.id,
                    )
                )
            metadata = {
                "linear_depletion_enabled": spec.cleanup_policy.linear_depletion.enabled,
                "enzyme": spec.cleanup_policy.linear_depletion.enzyme,
                "retained_topology": "circular_dsDNA",
            }
        elif step.kind == "nickase_digest":
            boundaries: list[int] = []
            retained_regions = _resolve_region_list(
                regions,
                step.retained_region_ids,
                code="RETAINED_REGION_MISSING",
                label="retained_region_ids",
                step_id=step.id,
                issues=step_issues,
            )
            projected_retained_regions = {
                region.id: projected
                for region in retained_regions
                if (
                    projected := _project_region_to_segments(
                        region,
                        current_segments,
                        code="RETAINED_REGION_EXCLUDED_FROM_CURRENT_STATE",
                        label="retained region",
                        state_id=step.id,
                        step_id=step.id,
                        issues=step_issues,
                    )
                )
                is not None
            }
            for site_id in step.site_ids:
                site = nickase_sites.get(site_id)
                if site is None:
                    step_issues.append(
                        _issue("NICKASE_SITE_MISSING", f"nickase site {site_id} is missing", step_id=step.id)
                    )
                    continue
                site_start = site.start - current_interval_start
                if site_start < 0 or site.end > current_interval_end:
                    step_issues.append(
                        _issue(
                            "NICKASE_SITE_EXCLUDED_FROM_CURRENT_STATE",
                            f"nickase site {site.id} falls outside current state interval "
                            f"{current_interval_start}:{current_interval_end}",
                            step_id=step.id,
                        )
                    )
                    continue
                if spec.catalogs.nickases is not None:
                    _catalog_site_issue(
                        site=site,
                        catalog_entry=catalogs.nickases.get(site.enzyme),
                        missing_code="NICKASE_CATALOG_ENTRY_MISSING",
                        mismatch_code="NICKASE_CATALOG_MISMATCH",
                        step_id=step.id,
                        issues=step_issues,
                    )
                try:
                    geometry = derive_cut_geometry(
                        current_primary,
                        start=site_start,
                        recognition_sequence=site.recognition_sequence,
                        orientation=site.orientation,
                        top_cut_offset=site.top_cut_offset,
                        bottom_cut_offset=site.bottom_cut_offset,
                    )
                except ValueError as exc:
                    step_issues.append(
                        _issue("NICKASE_SITE_INVALID", f"nickase site {site.id} is invalid: {exc}", step_id=step.id)
                    )
                    continue
                boundary = geometry.top_boundary if geometry.top_boundary is not None else geometry.bottom_boundary
                if boundary is None:
                    continue
                boundaries.append(boundary)
                site_end = site_start + len(site.recognition_sequence)
                for retained_region in projected_retained_regions.values():
                    if _projected_region_overlaps_interval(retained_region, start=site_start, end=site_end):
                        step_issues.append(
                            _issue(
                                "NICKASE_RETAINED_REGION_CONFLICT",
                                f"nickase site {site.id} overlaps retained region {retained_region.source_region_id}",
                                step_id=step.id,
                            )
                        )
            fragment_lengths = []
            sacrificial_regions = _resolve_region_list(
                regions,
                step.sacrificial_region_ids,
                code="SACRIFICIAL_REGION_MISSING",
                label="sacrificial_region_ids",
                step_id=step.id,
                issues=step_issues,
            )
            for region in sacrificial_regions:
                projected_region = _project_region_to_state(
                    region,
                    interval_start=current_interval_start,
                    interval_end=current_interval_end,
                    code="SACRIFICIAL_REGION_EXCLUDED_FROM_CURRENT_STATE",
                    label="sacrificial region",
                    step_id=step.id,
                    issues=step_issues,
                )
                if projected_region is None:
                    continue
                internal_boundaries = sorted(
                    boundary for boundary in boundaries if projected_region.start <= boundary <= projected_region.end
                )
                if not internal_boundaries:
                    step_issues.append(
                        _issue(
                            "NICKASE_SACRIFICIAL_REGION_UNCUT",
                            f"sacrificial region {region.id} is not cut by the configured nickase sites",
                            step_id=step.id,
                        )
                    )
                cuts = [
                    projected_region.start,
                    *internal_boundaries,
                    projected_region.end,
                ]
                fragment_lengths.extend(
                    cuts[idx + 1] - cuts[idx] for idx in range(0, len(cuts) - 1) if cuts[idx + 1] > cuts[idx]
                )
            retained_source_regions = [region for region in retained_regions if region.id in projected_retained_regions]
            retained_component_segments = _segments_for_projected_regions(
                retained_source_regions,
                projected_by_id=projected_retained_regions,
            )
            retained_components = [
                {
                    "id": region.id,
                    "source_start": segment.source_start,
                    "source_end": segment.source_end,
                    "state_start": segment.state_start,
                    "state_end": segment.state_end,
                    "sequence": _projected_region_sequence(current_primary, projected_retained_regions[region.id]),
                }
                for region, segment in zip(retained_source_regions, retained_component_segments, strict=True)
            ]
            retained_product = "".join(
                _projected_region_sequence(current_primary, projected_retained_regions[region_id])
                for region_id in step.retained_region_ids
                if region_id in projected_retained_regions
            )
            current_primary = retained_product
            current_complement = reverse_complement_iupac(retained_product) if retained_product else None
            current_segments = retained_component_segments
            current_interval_start = 0
            current_interval_end = len(current_primary)
            metadata = {
                "fragment_lengths": fragment_lengths,
                "retained_product": retained_product,
                "retained_components": retained_components,
            }
        elif step.kind == "size_selection":
            min_removed = spec.cleanup_policy.size_selection.min_removed_fragment_nt
            if min_removed is not None and any(length < min_removed for length in fragment_lengths):
                step_issues.append(
                    _issue(
                        "SIZE_SELECTION_FRAGMENT_TOO_SHORT_TO_REMOVE",
                        f"sacrificial fragments {fragment_lengths} fall below min removed threshold {min_removed}",
                        step_id=step.id,
                    )
                )
            max_sacrificial = spec.cleanup_policy.size_selection.max_retained_sacrificial_fragment_nt
            if max_sacrificial is not None and any(length > max_sacrificial for length in fragment_lengths):
                step_issues.append(
                    _issue(
                        "SIZE_SELECTION_FRAGMENT_TOO_LARGE",
                        f"sacrificial fragments {fragment_lengths} exceed max {max_sacrificial}",
                        step_id=step.id,
                    )
                )
            min_retained = spec.cleanup_policy.size_selection.min_retained_product_nt
            if min_retained is not None and len(retained_product) < min_retained:
                step_issues.append(
                    _issue(
                        "SIZE_SELECTION_RETAINED_PRODUCT_TOO_SHORT",
                        f"retained product length {len(retained_product)} is below min {min_retained}",
                        step_id=step.id,
                    )
                )
            metadata = {
                "fragment_lengths": fragment_lengths,
                "retained_product_length": len(retained_product),
                "min_removed_fragment_nt": min_removed,
                "max_retained_sacrificial_fragment_nt": max_sacrificial,
                "min_retained_product_nt": min_retained,
            }
        elif step.kind == "foldback":
            left_region = _resolve_region(
                regions,
                str(step.left_homology_window),
                code="HOMOLOGY_WINDOW_MISSING",
                label="left_homology_window",
                step_id=step.id,
                issues=step_issues,
            )
            right_region = _resolve_region(
                regions,
                str(step.right_homology_window),
                code="HOMOLOGY_WINDOW_MISSING",
                label="right_homology_window",
                step_id=step.id,
                issues=step_issues,
            )
            left = right = ""
            overlap = 0
            required = int(step.min_complementary_bases or 0)
            projected_windows: list[dict[str, Any]] = []
            left_projected = right_projected = None
            has_junction_spanning_window = False
            if left_region is not None and right_region is not None:
                left_projected = _project_region_to_segments(
                    left_region,
                    current_segments,
                    code="HOMOLOGY_WINDOW_EXCLUDED_FROM_CURRENT_STATE",
                    label="homology window",
                    state_id=step.id,
                    step_id=step.id,
                    issues=step_issues,
                )
                right_projected = _project_region_to_segments(
                    right_region,
                    current_segments,
                    code="HOMOLOGY_WINDOW_EXCLUDED_FROM_CURRENT_STATE",
                    label="homology window",
                    state_id=step.id,
                    step_id=step.id,
                    issues=step_issues,
                )
                if left_projected is not None:
                    left = _projected_region_sequence(current_primary, left_projected)
                    projected_windows.append(
                        _projected_region_payload(
                            left_projected,
                            source_region=left_region,
                            sequence=current_primary,
                        )
                    )
                    if left_projected.spans_junction:
                        has_junction_spanning_window = True
                        step_issues.append(
                            _issue(
                                "HOMOLOGY_WINDOW_SPANS_JUNCTION",
                                f"homology window {left_region.id} spans multiple retained-product segments",
                                step_id=step.id,
                            )
                        )
                if right_projected is not None:
                    right = _projected_region_sequence(current_primary, right_projected)
                    projected_windows.append(
                        _projected_region_payload(
                            right_projected,
                            source_region=right_region,
                            sequence=current_primary,
                        )
                    )
                    if right_projected.spans_junction:
                        has_junction_spanning_window = True
                        step_issues.append(
                            _issue(
                                "HOMOLOGY_WINDOW_SPANS_JUNCTION",
                                f"homology window {right_region.id} spans multiple retained-product segments",
                                step_id=step.id,
                            )
                        )
                if left_projected is not None and right_projected is not None:
                    overlap = longest_reverse_complement_overlap(left, right)
                    if overlap < required:
                        step_issues.append(
                            _issue(
                                "FOLDBACK_HOMOLOGY_INSUFFICIENT",
                                f"foldback homology overlap {overlap} is below required {required}",
                                step_id=step.id,
                            )
                        )
            overlap_start = len(left) - overlap if overlap > 0 else None
            overlap_end = len(left) if overlap > 0 else None
            topology_compatibility = (
                left_projected is not None
                and right_projected is not None
                and not has_junction_spanning_window
                and overlap >= required
            )
            metadata = {
                "left_homology": left,
                "right_homology": right,
                "complementary_bases": overlap,
                "paired_nt": overlap,
                "overlap_start": overlap_start,
                "overlap_end": overlap_end,
                "sequence_mode": _sequence_mode_for_values(left, right),
                "topology_compatibility": topology_compatibility,
                "projected_homology_windows": projected_windows,
            }
        elif step.kind == "adapter_ligation":
            adapter_sequence = _resolve_adapter_sequence(
                spec,
                step.adapter_sequence,
                step_id=step.id,
                catalogs=catalogs,
                issues=step_issues,
            )
            current_primary = f"{retained_product}|{adapter_sequence}"
            current_complement = None
            arms = _branched_state_arms(retained_product=retained_product, adapter_sequence=adapter_sequence)
            metadata = {
                "adapter_sequence": adapter_sequence,
                "y_adapter_id": spec.adapter_policy.y_adapter_id,
                "topology": "branched_y",
                "arms": arms,
                "branch_junction": {
                    "payload_arm_id": arms[0]["id"],
                    "payload_state_index": arms[0]["state_end"],
                    "adapter_arm_id": arms[1]["id"],
                    "adapter_state_index": arms[1]["state_start"],
                    "separator": "|",
                },
            }
        elif step.kind == "amplification":
            current_primary = f"{retained_product}{adapter_sequence}"
            current_complement = reverse_complement_iupac(current_primary)
            if assembled_payload and not sequence_contains_iupac(current_primary, assembled_payload):
                step_issues.append(
                    _issue(
                        "AMPLIFICATION_PAYLOAD_MISSING",
                        f"final product does not preserve assembled payload {assembled_payload!r}",
                        step_id=step.id,
                    )
                )
            requirements = [
                *(requirement.sequence for requirement in spec.adapter_policy.primer_binding_requirements),
                str(step.forward_primer_requirement),
                str(step.reverse_primer_requirement),
            ]
            for requirement in requirements:
                if requirement and not sequence_contains_iupac(current_primary, requirement):
                    step_issues.append(
                        _issue(
                            "AMPLIFICATION_PRIMER_MISSING",
                            f"final product does not contain primer requirement {requirement!r}",
                            step_id=step.id,
                        )
                    )
            metadata = {"final_product_length": len(current_primary)}

        issues.extend(step_issues)
        state_status = "unsatisfied" if step_issues else "satisfied"
        states.append(
            _state(
                state_id=step.id,
                step_id=step.id,
                kind=step.kind,
                status=state_status,
                primary_sequence=current_primary if state_primary is None else state_primary,
                complement_sequence=current_complement if state_complement is None else state_complement,
                metadata=metadata,
            )
        )

    report_sequence_mode = (
        "iupac_pattern" if any(state.sequence_mode == "iupac_pattern" for state in states) else "concrete"
    )
    report_validation_mode = (
        "pattern_compatibility" if report_sequence_mode == "iupac_pattern" else "concrete_realization"
    )
    states = [state.model_copy(update={"validation_mode": report_validation_mode}) for state in states]

    return YiuValidationReport(
        spec_name=spec.name,
        status="unsatisfied" if issues else "satisfied",
        sequence_mode=report_sequence_mode,
        validation_mode=report_validation_mode,
        metadata=YiuReportMetadata(
            spec_schema_version=spec.schema_version,
            step_count=len(spec.step_graph.steps),
            state_count=len(states),
            emitted_view_count=len(states) if spec.output.emit_view_contracts else 0,
            catalog_paths=[str(path) for path in catalogs.paths],
        ),
        states=states,
        issues=issues,
    )


def _segment_rows(segments: list[_StateSegment]) -> list[dict[str, Any]]:
    return [
        {
            "segment_id": segment.segment_id,
            "source_start": segment.source_start,
            "source_end": segment.source_end,
            "state_start": segment.state_start,
            "state_end": segment.state_end,
        }
        for segment in segments
    ]


def _compound_annotation(
    *,
    annotation_id: str,
    pieces: list[_StateSegment],
    assembled_coordinate_space: str,
) -> dict[str, Any]:
    return {
        "id": annotation_id,
        "projection_kind": "compound",
        "assembled_coordinate_space": assembled_coordinate_space,
        "pieces": [
            {
                "segment_id": piece.segment_id,
                "start": piece.state_start,
                "end": piece.state_end,
            }
            for piece in pieces
        ],
    }


def _v2_report_sequence_mode(states: list[YiuStateRecord]) -> str:
    return "pattern" if any(state.sequence_mode == "pattern" for state in states) else "concrete"


def _has_error_issue(issues: list[YiuValidationIssue]) -> bool:
    return any(issue.severity != "warning" for issue in issues)


def _build_yiu_report_v2(spec: YiuProcessSpecV2, *, catalogs: LoadedYiuCatalogs | None = None) -> YiuValidationReport:
    catalogs = catalogs or LoadedYiuCatalogs()
    issues: list[YiuValidationIssue] = []
    states: list[YiuStateRecord] = []
    sequence = spec.source_oligo.sequence
    annotations = spec.source_oligo.annotations
    primer_cores = _v2_primer_core_lookup(spec)
    regions = _v2_region_lookup(spec)
    issues.extend(_v2_overlap_issues(spec))

    source_segments = _segments_for_source_regions([RegionSpec(id="source", start=0, end=len(sequence))])
    source_annotations = [
        {"id": core.id, "annotation_class": "PrimerBindingCore", "start": core.start, "end": core.end}
        for core in annotations.primer_binding_cores
    ]
    source_annotations.extend(
        {"id": tail.id, "annotation_class": "PrimerTail", "primer_binding_core_id": tail.primer_binding_core_id}
        for tail in annotations.primer_tails
    )
    source_annotations.extend(
        {"id": region.id, "annotation_class": "PayloadWindow", "start": region.start, "end": region.end}
        for region in annotations.payload_windows
    )
    states.append(
        _state(
            state_id="source_oligo_ssdna",
            step_id="source_oligo",
            kind="source_oligo_ssdna",
            state_kind="source_oligo_ssdna",
            topology_kind="linear_ssdna",
            status="unsatisfied" if _has_error_issue(issues) else "satisfied",
            primary_sequence=sequence,
            metadata={"length_nt": len(sequence)},
            view_contract_version=2,
            segments=_segment_rows(source_segments),
            annotations=source_annotations,
            pattern_label="pattern",
        )
    )

    step_issues: list[YiuValidationIssue] = []
    forward_core = primer_cores.get(spec.steps.source_pcr.forward_primer_id.replace("oES790", "source_fwd_core"))
    reverse_core = primer_cores.get(spec.steps.source_pcr.reverse_primer_id.replace("oES791", "source_rev_core"))
    if forward_core is None:
        forward_core = primer_cores.get("source_fwd_core")
    if reverse_core is None:
        reverse_core = primer_cores.get("source_rev_core")
    if forward_core is None or reverse_core is None:
        step_issues.append(
            _issue(
                "PCR_PRIMER_SITE_MISSING",
                "source_pcr primer binding core is missing",
                step_id="source_pcr",
            )
        )
        amplicon_start = 0
        amplicon_end = len(sequence)
    else:
        amplicon_start = forward_core.start
        amplicon_end = reverse_core.end
    current_primary = sequence[amplicon_start:amplicon_end]
    current_complement = reverse_complement_iupac(current_primary)
    current_segments = _segments_for_source_regions(
        [RegionSpec(id="source_amplicon", start=amplicon_start, end=amplicon_end)]
    )
    pcr_state = _state(
        state_id="source_amplicon_dsdna",
        step_id="source_pcr",
        kind="source_amplicon_dsdna",
        state_kind="source_amplicon_dsdna",
        topology_kind="linear_dsdna",
        status="unsatisfied" if _has_error_issue(step_issues) else "satisfied",
        primary_sequence=current_primary,
        complement_sequence=current_complement,
        metadata={
            "amplicon_start": amplicon_start,
            "amplicon_end": amplicon_end,
            "amplicon_length_nt": len(current_primary),
        },
        view_contract_version=2,
        segments=_segment_rows(current_segments),
        annotations=[
            {
                "id": core.id,
                "annotation_class": "PrimerBindingCore",
                "start": core.start - amplicon_start,
                "end": core.end - amplicon_start,
            }
            for core in annotations.primer_binding_cores
            if core.start >= amplicon_start and core.end <= amplicon_end
        ],
        pattern_label="pattern",
    )
    issues.extend(step_issues)
    states.append(pcr_state)

    digest_statuses: list[str] = []
    digest_issues: list[YiuValidationIssue] = []
    digest_cuts: list[dict[str, Any]] = []
    for site in annotations.nickase_sites:
        if site.enzyme not in spec.steps.double_nicking_digest.enzymes:
            continue
        site_start = site.start - amplicon_start
        if site_start < 0 or site.end > amplicon_end:
            digest_issues.append(
                _issue(
                    "NICKASE_SITE_EXCLUDED_FROM_CURRENT_STATE",
                    f"nickase site {site.id} falls outside source amplicon {amplicon_start}:{amplicon_end}",
                    step_id="double_nicking_digest",
                )
            )
            continue
        site_sequence = current_primary[site_start : site_start + len(site.recognition_sequence)]
        status = _iupac_match_status(site_sequence, site.recognition_sequence)
        digest_statuses.append(status)
        _pattern_policy_issue(
            status=status,
            policy=spec.payload_goal.evidence_policy,
            label=f"nickase site {site.id}",
            step_id="double_nicking_digest",
            issues=digest_issues,
        )
        if spec.catalogs.enzymes is not None:
            catalog_entry = catalogs.enzymes.get(site.enzyme)
            if catalog_entry is None:
                digest_issues.append(
                    _issue(
                        "ENZYME_CATALOG_ENTRY_MISSING",
                        f"enzyme {site.enzyme!r} for site {site.id} is not present in catalogs.enzymes",
                        step_id="double_nicking_digest",
                    )
                )
            elif catalog_entry.recognition_sequence != site.recognition_sequence:
                digest_issues.append(
                    _issue(
                        "ENZYME_CATALOG_MISMATCH",
                        f"site {site.id} recognition sequence {site.recognition_sequence!r} "
                        "does not match catalog value "
                        f"{catalog_entry.recognition_sequence!r} for enzyme {site.enzyme!r}",
                        step_id="double_nicking_digest",
                    )
                )
        try:
            geometry = derive_cut_geometry(
                current_primary,
                start=site_start,
                recognition_sequence=site.recognition_sequence,
                orientation=site.orientation,
                top_cut_offset=site.top_cut_offset,
                bottom_cut_offset=site.bottom_cut_offset,
            )
            digest_cuts.append(
                {
                    "site_id": site.id,
                    "top_boundary": geometry.top_boundary,
                    "bottom_boundary": geometry.bottom_boundary,
                }
            )
        except ValueError as exc:
            digest_issues.append(
                _issue(
                    "NICKASE_SITE_INVALID",
                    f"nickase site {site.id} is invalid: {exc}",
                    step_id="double_nicking_digest",
                )
            )

    retained_regions = [regions["retained_left"], regions["retained_right"]]
    pool_fragments = [
        {"fragment_id": "amplicon_prefix", "source_start": 0, "source_end": 14, "retained": False},
        {"fragment_id": "retained_left", "source_start": 14, "source_end": 18, "retained": True},
        {"fragment_id": "sacrificial_center", "source_start": 18, "source_end": 22, "retained": False},
        {"fragment_id": "retained_right", "source_start": 22, "source_end": 26, "retained": True},
        {"fragment_id": "amplicon_suffix", "source_start": 26, "source_end": len(current_primary), "retained": False},
    ]
    digest_state = _state(
        state_id="post_double_nicking_fragment_pool",
        step_id="double_nicking_digest",
        kind="post_double_nicking_fragment_pool",
        state_kind="post_double_nicking_fragment_pool",
        topology_kind="fragment_pool",
        status="unsatisfied" if _has_error_issue(digest_issues) else "satisfied",
        primary_sequence=current_primary,
        metadata={"selected_enzymes": spec.steps.double_nicking_digest.enzymes},
        view_contract_version=2,
        fragments=pool_fragments,
        cuts=digest_cuts,
        pattern_evidence_summary=_pattern_summary(digest_statuses),
        pattern_label="pattern",
    )
    issues.extend(digest_issues)
    states.append(digest_state)

    retained_segments = _segments_for_source_regions(retained_regions)
    retained_product = "".join(_sequence_for_region(sequence, region) for region in retained_regions)
    cleanup_issues: list[YiuValidationIssue] = []
    if spec.steps.heat_cleanup.enabled and spec.steps.heat_cleanup.min_retained_nt is not None:
        if len(retained_product) < spec.steps.heat_cleanup.min_retained_nt:
            cleanup_issues.append(
                _issue(
                    "HEAT_CLEANUP_RETAINED_PRODUCT_TOO_SHORT",
                    f"retained product length {len(retained_product)} is below min_retained_nt "
                    f"{spec.steps.heat_cleanup.min_retained_nt}",
                    step_id="heat_cleanup",
                )
            )
    cleanup_state = _state(
        state_id="post_heat_cleanup_fragment_pool",
        step_id="heat_cleanup",
        kind="post_heat_cleanup_fragment_pool",
        state_kind="post_heat_cleanup_fragment_pool",
        topology_kind="fragment_pool",
        status="unsatisfied" if _has_error_issue(cleanup_issues) else "satisfied",
        primary_sequence=retained_product,
        metadata={
            "discarded_fragments": [fragment["fragment_id"] for fragment in pool_fragments if not fragment["retained"]],
            "retained_fragment_ids": ["retained_left", "retained_right"],
        },
        view_contract_version=2,
        segments=_segment_rows(retained_segments),
        fragments=[
            {
                "fragment_id": "retained_left",
                "state_start": 0,
                "state_end": 4,
                "sequence": retained_product[:4],
            },
            {
                "fragment_id": "retained_right",
                "state_start": 4,
                "state_end": 8,
                "sequence": retained_product[4:],
            },
        ],
        pattern_label="pattern",
    )
    issues.extend(cleanup_issues)
    states.append(cleanup_state)

    adapter_issues: list[YiuValidationIssue] = []
    adapter_part = _v2_part(catalogs, spec.steps.adapter_anneal.adapter_id, label="adapter_anneal.adapter_id")
    adapter_match = _evaluate_ligation_compatibility(
        retained_product,
        adapter_part.sequence,
        mode=spec.steps.adapter_anneal.compatibility_mode,
        partial_rule=(
            spec.steps.adapter_anneal.partial_complement.model_dump(mode="json")
            if spec.steps.adapter_anneal.partial_complement is not None
            else None
        ),
        bulged_rule=(
            spec.steps.adapter_anneal.bulged.model_dump(mode="json")
            if spec.steps.adapter_anneal.bulged is not None
            else None
        ),
    )
    if adapter_match is None:
        adapter_issues.append(
            _issue(
                "ADAPTER_ANNEAL_COMPATIBILITY_FAIL",
                "adapter-source annealing does not satisfy the configured compatibility mode",
                step_id="adapter_anneal",
            )
        )
    adapter_segments = [
        _StateSegment("retained_left", 14, 18, 0, 4),
        _StateSegment("retained_right", 22, 26, 4, 8),
        _StateSegment("y_adapter", -1, -1, 8, 8 + len(adapter_part.sequence)),
    ]
    adapter_state = _state(
        state_id="adapter_annealed_complex",
        step_id="adapter_anneal",
        kind="adapter_annealed_complex",
        state_kind="adapter_annealed_complex",
        topology_kind="annealed_complex",
        status="unsatisfied" if _has_error_issue(adapter_issues) else "satisfied",
        primary_sequence=f"{retained_product}|{adapter_part.sequence}",
        metadata={
            "adapter_id": adapter_part.id,
            "compatibility_mode": spec.steps.adapter_anneal.compatibility_mode,
            "paired_nt": adapter_match.paired_nt if adapter_match is not None else 0,
        },
        view_contract_version=2,
        segments=_segment_rows(adapter_segments),
        junctions=[
            {
                "id": "adapter_anneal_overlap",
                "paired_nt": adapter_match.paired_nt if adapter_match is not None else 0,
                "compatibility_mode": spec.steps.adapter_anneal.compatibility_mode,
            }
        ],
        pattern_label="pattern",
    )
    issues.extend(adapter_issues)
    states.append(adapter_state)

    ligation_issues: list[YiuValidationIssue] = []
    if spec.steps.hairpin_ligation.require_5p_phosphate and not adapter_part.phosphorylated_5p:
        ligation_issues.append(
            _issue(
                "LIGATION_5P_PHOSPHATE_REQUIRED",
                "hairpin_ligation requires a 5' phosphorylated adapter part",
                step_id="hairpin_ligation",
            )
        )
    ligation_match = _evaluate_ligation_compatibility(
        retained_product,
        adapter_part.sequence,
        mode=spec.steps.hairpin_ligation.compatibility_mode,
        partial_rule=(
            spec.steps.hairpin_ligation.partial_complement.model_dump(mode="json")
            if spec.steps.hairpin_ligation.partial_complement is not None
            else None
        ),
        bulged_rule=(
            spec.steps.hairpin_ligation.bulged.model_dump(mode="json")
            if spec.steps.hairpin_ligation.bulged is not None
            else None
        ),
    )
    if ligation_match is None:
        ligation_issues.append(
            _issue(
                "HAIRPIN_LIGATION_COMPATIBILITY_FAIL",
                "hairpin ligation does not satisfy the configured compatibility mode",
                step_id="hairpin_ligation",
            )
        )
    hairpin_sequence = f"{retained_product}{adapter_part.sequence}"
    ligation_state = _state(
        state_id="ligated_ssdna_hairpin",
        step_id="hairpin_ligation",
        kind="ligated_ssdna_hairpin",
        state_kind="ligated_ssdna_hairpin",
        topology_kind="hairpin_ssdna",
        status="unsatisfied" if _has_error_issue(ligation_issues) else "satisfied",
        primary_sequence=hairpin_sequence,
        metadata={
            "ligase": spec.steps.hairpin_ligation.ligase,
            "require_5p_phosphate": spec.steps.hairpin_ligation.require_5p_phosphate,
            "paired_nt": ligation_match.paired_nt if ligation_match is not None else 0,
        },
        view_contract_version=2,
        junctions=[
            {
                "id": "hairpin_ligation_junction",
                "paired_nt": ligation_match.paired_nt if ligation_match is not None else 0,
                "compatibility_mode": spec.steps.hairpin_ligation.compatibility_mode,
            }
        ],
        pattern_label="pattern",
    )
    issues.extend(ligation_issues)
    states.append(ligation_state)

    hairpin_pcr_issues: list[YiuValidationIssue] = []
    assembled_payload_status = _iupac_match_status(retained_product, spec.payload_goal.assembled_payload_pattern)
    _pattern_policy_issue(
        status=assembled_payload_status,
        policy=spec.payload_goal.evidence_policy,
        label="assembled payload pattern",
        step_id="hairpin_pcr",
        issues=hairpin_pcr_issues,
    )
    insert_primary = retained_product + reverse_complement_iupac(retained_product)
    insert_complement = reverse_complement_iupac(insert_primary)
    insert_segments = [
        _StateSegment("retained_left", 14, 18, 0, 4),
        _StateSegment("retained_right", 22, 26, 4, 8),
        _StateSegment("inverted_right", 22, 26, 8, 12),
        _StateSegment("inverted_left", 14, 18, 12, 16),
    ]
    insert_annotations = [
        _compound_annotation(
            annotation_id="assembled_payload",
            pieces=insert_segments[:2],
            assembled_coordinate_space=spec.payload_goal.assembly_space,
        )
    ]
    hairpin_pcr_state = _state(
        state_id="hairpin_pcr_linear_insert",
        step_id="hairpin_pcr",
        kind="hairpin_pcr_linear_insert",
        state_kind="hairpin_pcr_linear_insert",
        topology_kind="linear_dsdna",
        status="unsatisfied" if _has_error_issue(hairpin_pcr_issues) else "satisfied",
        primary_sequence=insert_primary,
        complement_sequence=insert_complement,
        metadata={
            "single_primer_precycles": spec.steps.hairpin_pcr.single_primer_precycles.model_dump(mode="json"),
            "x_structure_resolution_cycle": spec.steps.hairpin_pcr.x_structure_resolution_cycle.model_dump(mode="json"),
        },
        view_contract_version=2,
        segments=_segment_rows(insert_segments),
        annotations=insert_annotations,
        junctions=[
            {
                "id": "payload_assembly_junction",
                "assembly_space": spec.payload_goal.assembly_space,
                "join_index": 4,
            }
        ],
        pattern_evidence_summary=_pattern_summary([assembled_payload_status]),
        pattern_label="pattern",
    )
    issues.extend(hairpin_pcr_issues)
    states.append(hairpin_pcr_state)
    current_insert_primary = insert_primary
    current_insert_complement = insert_complement
    current_insert_segments = insert_segments

    if spec.steps.insert_cleanup.enabled:
        insert_cleanup_state = _state(
            state_id="post_insert_cleanup_linear_insert",
            step_id="insert_cleanup",
            kind="post_insert_cleanup_linear_insert",
            state_kind="post_insert_cleanup_linear_insert",
            topology_kind="linear_dsdna",
            status="satisfied",
            primary_sequence=current_insert_primary,
            complement_sequence=current_insert_complement,
            metadata={"enabled": True, "method": spec.steps.insert_cleanup.method},
            view_contract_version=2,
            segments=_segment_rows(current_insert_segments),
            annotations=insert_annotations,
            junctions=hairpin_pcr_state.junctions,
            pattern_label="pattern",
        )
        states.append(insert_cleanup_state)

    if spec.workflow_scope == "insert_plus_backbone_cloning" and spec.steps.backbone_pcr.enabled:
        backbone_issues: list[YiuValidationIssue] = []
        backbone_sequence: str | None = None
        if spec.steps.backbone_pcr.backbone_id is None:
            backbone_issues.append(
                _issue(
                    "BACKBONE_ID_REQUIRED",
                    "backbone_pcr.backbone_id is required when backbone_pcr is enabled",
                    step_id="backbone_pcr",
                )
            )
        else:
            backbone_entry = catalogs.backbones.get(spec.steps.backbone_pcr.backbone_id)
            if backbone_entry is None:
                backbone_issues.append(
                    _issue(
                        "BACKBONE_CATALOG_ENTRY_MISSING",
                        f"backbone {spec.steps.backbone_pcr.backbone_id!r} is not present in catalogs.backbones",
                        step_id="backbone_pcr",
                    )
                )
            else:
                backbone_sequence = backbone_entry.sequence or ""
        backbone_primary = backbone_sequence or ""
        backbone_state = _state(
            state_id="backbone_amplicon",
            step_id="backbone_pcr",
            kind="backbone_amplicon",
            state_kind="backbone_amplicon",
            topology_kind="linear_dsdna",
            status="unsatisfied" if _has_error_issue(backbone_issues) else "satisfied",
            primary_sequence=backbone_primary,
            complement_sequence=reverse_complement_iupac(backbone_primary) if backbone_primary else None,
            metadata={"backbone_id": spec.steps.backbone_pcr.backbone_id},
            view_contract_version=2,
            segments=(
                [
                    {
                        "segment_id": spec.steps.backbone_pcr.backbone_id or "backbone",
                        "source_start": 0,
                        "source_end": len(backbone_primary),
                        "state_start": 0,
                        "state_end": len(backbone_primary),
                    }
                ]
                if backbone_primary
                else []
            ),
            pattern_label="pattern",
        )
        issues.extend(backbone_issues)
        states.append(backbone_state)

        if spec.steps.golden_gate_assembly.enabled:
            assembly_primary = (
                f"{current_insert_primary}|{backbone_primary}" if backbone_primary else current_insert_primary
            )
            assembly_state = _state(
                state_id="assembly_reaction",
                step_id="golden_gate_assembly",
                kind="assembly_reaction",
                state_kind="assembly_reaction",
                topology_kind="assembly_reaction",
                status="satisfied",
                primary_sequence=assembly_primary,
                metadata={
                    "enzyme": spec.steps.golden_gate_assembly.enzyme,
                    "backbone_id": spec.steps.golden_gate_assembly.backbone_id,
                },
                view_contract_version=2,
                junctions=[{"id": "golden_gate_join", "enzyme": spec.steps.golden_gate_assembly.enzyme}],
                pattern_label="pattern",
            )
            plasmid_primary = f"{current_insert_primary}{backbone_primary}"
            plasmid_state = _state(
                state_id="assembled_plasmid_candidate",
                step_id="golden_gate_assembly",
                kind="assembled_plasmid_candidate",
                state_kind="assembled_plasmid_candidate",
                topology_kind="circular_dsdna_candidate",
                status="satisfied",
                primary_sequence=plasmid_primary,
                complement_sequence=reverse_complement_iupac(plasmid_primary) if plasmid_primary else None,
                metadata={"backbone_id": spec.steps.golden_gate_assembly.backbone_id},
                view_contract_version=2,
                pattern_label="pattern",
            )
            states.extend([assembly_state, plasmid_state])

    report_sequence_mode = _v2_report_sequence_mode(states)
    report_validation_mode = "pattern_compatibility" if report_sequence_mode == "pattern" else "concrete_realization"
    states = [state.model_copy(update={"validation_mode": report_validation_mode}) for state in states]
    return YiuValidationReport(
        protocol=spec.protocol_template,
        protocol_template=spec.protocol_template,
        workflow_scope=spec.workflow_scope,
        spec_name=spec.name,
        status="unsatisfied" if _has_error_issue(issues) else "satisfied",
        sequence_mode=report_sequence_mode,
        validation_mode=report_validation_mode,
        metadata=YiuReportMetadata(
            spec_schema_version=spec.schema_version,
            step_count=len(states) - 1,
            state_count=len(states),
            emitted_view_count=len(states) if spec.output.emit_view_contracts else 0,
            view_contract_version=spec.output.publish_contract_version,
            catalog_paths=[str(path) for path in catalogs.paths],
        ),
        states=states,
        issues=issues,
    )


def _build_yiu_report(
    spec: YiuProcessSpec | YiuProcessSpecV2,
    *,
    catalogs: LoadedYiuCatalogs | None = None,
) -> YiuValidationReport:
    if isinstance(spec, YiuProcessSpecV2):
        return _build_yiu_report_v2(spec, catalogs=catalogs)
    return _build_yiu_report_v1(spec, catalogs=catalogs)


def validate_yiu_spec(path: str | Path) -> YiuValidationReport:
    spec, _spec_path, workspace_root = load_yiu_spec(path)
    catalogs = load_yiu_catalogs(spec, workspace_root=workspace_root)
    report = _build_yiu_report(spec, catalogs=catalogs)
    return report.model_copy(
        update={
            "metadata": report.metadata.model_copy(
                update={
                    # `validate` returns a report only; it does not materialize state-view files.
                    "emitted_view_count": 0,
                }
            )
        }
    )


def _catalog_bytes(catalog_paths: list[Path]) -> bytes:
    if not catalog_paths:
        return b""
    return b"\n".join(path.read_bytes() for path in catalog_paths if path.exists())


def _annotation_rows(spec: YiuProcessSpec) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for category, collection in _annotation_collections(spec):
        for item in collection:
            rows.append(
                {
                    "category": category,
                    "id": item.id,
                    "start": item.start,
                    "end": _item_end(item),
                    "label": getattr(item, "enzyme", item.id),
                }
            )
    return rows


def _parts_rows(report: YiuValidationReport) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for state in report.states:
        if state.primary_sequence:
            rows.append(
                {
                    "state_id": state.state_id,
                    "part_id": f"{state.state_id}_primary",
                    "role": state.kind,
                    "sequence": state.primary_sequence,
                }
            )
        if state.complement_sequence:
            rows.append(
                {
                    "state_id": state.state_id,
                    "part_id": f"{state.state_id}_complement",
                    "role": f"{state.kind}_complement",
                    "sequence": state.complement_sequence,
                }
            )
    return rows


def _fragment_rows(report: YiuValidationReport) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for state in report.states:
        for index, length in enumerate(state.metadata.get("fragment_lengths", []), start=1):
            rows.append({"state_id": state.state_id, "fragment_id": f"{state.state_id}_{index}", "length_nt": length})
    return rows


def _publish_views(run_dir: Path, report: YiuValidationReport) -> None:
    published_views_dir(run_dir).mkdir(parents=True, exist_ok=True)
    view_contract_version = report.metadata.view_contract_version or STATE_VIEW_SCHEMA_VERSION
    for state in report.states:
        payload = {
            "schema_version": view_contract_version,
            "view_contract_version": view_contract_version,
            "family": "yiu",
            "workflow": "yiu",
            "protocol": report.protocol,
            "protocol_template": report.protocol_template,
            "state_id": state.state_id,
            "state_kind": state.state_kind or state.kind,
            "kind": state.kind,
            "status": state.status,
            "molecule_topology": _state_topology(state),
            "topology_kind": state.topology_kind or _state_topology(state),
            "sequence_mode": state.sequence_mode,
            "validation_mode": state.validation_mode,
            "primary_sequence": state.primary_sequence,
            "complement_sequence": state.complement_sequence,
            "segments": state.segments,
            "annotations": state.annotations,
            "cuts": state.cuts,
            "junctions": state.junctions,
            "fragments": state.fragments,
            "meta": state.metadata,
        }
        state_view_path(run_dir, state.state_id).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _materialize_yiu_bundle(spec_path: str | Path, *, force_overwrite: bool) -> tuple[Path, YiuValidationReport]:
    spec, resolved_spec_path, workspace_root = load_yiu_spec(spec_path)
    catalogs = load_yiu_catalogs(spec, workspace_root=workspace_root)
    report = _build_yiu_report(spec, catalogs=catalogs)
    catalog_paths = list(catalogs.paths)
    spec_bytes = resolved_spec_path.read_bytes()
    catalog_bytes = _catalog_bytes(catalog_paths)
    run_id = design_id(spec_bytes=spec_bytes, catalog_bytes=catalog_bytes)
    input_fingerprint_value = input_fingerprint(spec_bytes=spec_bytes, catalog_bytes=catalog_bytes)
    catalog_fingerprint_value = catalog_fingerprint(catalog_bytes=catalog_bytes)
    code_revision = resolve_code_revision(workspace_root)
    run_dir = build_run_dir(
        workspace_root=workspace_root,
        run_root=spec.output.run_dir,
        spec_name=spec.name,
        run_id=run_id,
    )
    prepare_run_dir(run_dir, force_overwrite=force_overwrite)
    report = report.model_copy(update={"run_dir": str(run_dir.resolve())})
    write_report(run_dir, report)
    write_status(
        run_dir,
        report,
        input_fingerprint_value=input_fingerprint_value,
        catalog_fingerprint_value=catalog_fingerprint_value,
        code_revision=code_revision,
    )
    write_manifest(
        run_dir,
        workspace_root=workspace_root,
        spec_path=resolved_spec_path,
        report=report,
        input_fingerprint_value=input_fingerprint_value,
        catalog_fingerprint_value=catalog_fingerprint_value,
        code_revision=code_revision,
        catalog_paths=catalog_paths,
    )
    write_trace(run_dir, report.states)
    write_trace_manifest(run_dir, report)
    write_csv(
        parts_path(run_dir),
        fieldnames=["state_id", "part_id", "role", "sequence"],
        rows=_parts_rows(report),
    )
    write_csv(
        annotations_path(run_dir),
        fieldnames=["category", "id", "start", "end", "label"],
        rows=_annotation_rows(spec),
    )
    write_csv(
        fragments_path(run_dir),
        fieldnames=["state_id", "fragment_id", "length_nt"],
        rows=_fragment_rows(report),
    )
    if spec.output.emit_view_contracts:
        _publish_views(run_dir, report)
        write_published_views_manifest(run_dir, report)
    return run_dir, report


def run_yiu_design(spec_path: str | Path, *, force_overwrite: bool = False) -> tuple[Path, YiuValidationReport]:
    return _materialize_yiu_bundle(spec_path, force_overwrite=force_overwrite)


def run_yiu_trace(spec_path: str | Path, *, force_overwrite: bool = False) -> tuple[Path, YiuValidationReport]:
    return _materialize_yiu_bundle(spec_path, force_overwrite=force_overwrite)


def yiu_show_payload(run_dir: str | Path) -> dict[str, object]:
    resolved = Path(run_dir).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"YIU run directory not found: {resolved}")
    manifest = report = status = None
    if report_path(resolved).exists():
        report = json.loads(report_path(resolved).read_text(encoding="utf-8"))
    if status_path(resolved).exists():
        status = json.loads(status_path(resolved).read_text(encoding="utf-8"))
    if (resolved / "yiu_manifest.json").exists():
        manifest = json.loads((resolved / "yiu_manifest.json").read_text(encoding="utf-8"))
    if report is None or status is None or manifest is None:
        raise ValueError(f"Run directory does not contain a complete YIU bundle: {resolved}")
    return {
        "spec_name": report["spec_name"],
        "run_dir": str(resolved),
        "status": status["status"],
        "status_message": status["status_message"],
        "protocol": status.get("protocol"),
        "protocol_template": status.get("protocol_template"),
        "view_contract_version": status.get("view_contract_version"),
        "manifest_path": str((resolved / "yiu_manifest.json").resolve()),
        "status_path": str(status_path(resolved).resolve()),
        "report_path": str(report_path(resolved).resolve()),
        "trace_path": str(trace_path(resolved).resolve()),
        "trace_manifest_path": str((resolved / "yiu_trace_manifest.json").resolve()),
        "published_views_manifest_path": str(published_views_manifest_path(resolved).resolve()),
        "published_views_dir": str(published_views_dir(resolved).resolve()),
    }
