"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/app/yiu_workflow/helpers.py

Shared helpers for YIU report construction and bundle publication.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from dnadesign.cruncher.bio import (
    iupac_bases_for_symbol,
    iupac_symbols_compatible,
    normalize_iupac,
    reverse_complement_iupac,
)
from dnadesign.cruncher.yiu.catalog import LoadedYiuCatalogs
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
    YiuStateRecord,
    YiuTemplateBindingsV2,
    YiuValidationIssue,
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
        + spec.source_oligo.annotations.named_regions
    )
    return {region.id: region for region in regions}


def _v2_nickase_lookup(spec: YiuProcessSpecV2) -> dict[str, EnzymeSiteSpec]:
    return {site.id: site for site in spec.source_oligo.annotations.nickase_sites}


def _v2_site_lookup(spec: YiuProcessSpecV2) -> dict[str, EnzymeSiteSpec]:
    return {
        site.id: site
        for site in [*spec.source_oligo.annotations.restriction_sites, *spec.source_oligo.annotations.nickase_sites]
    }


def _v2_part(catalogs: LoadedYiuCatalogs, part_id: str, *, label: str) -> YiuOligoPartCatalogEntry:
    part = catalogs.oligo_parts.get(part_id)
    if part is None:
        raise ValueError(f"{label} {part_id!r} is not present in catalogs.oligo_parts")
    return part


def _split_template_bindings(spec: YiuProcessSpecV2) -> YiuTemplateBindingsV2:
    if spec.template_bindings is None:
        raise ValueError(
            "YIU_TEMPLATE_BINDING_MISSING: template_bindings is required for "
            "protocol_template=yiu_circularized_payload_v1"
        )
    return spec.template_bindings


def _overlap_policy_key(left_kind: str, right_kind: str) -> tuple[str, str]:
    return tuple(sorted((left_kind, right_kind)))


_YIU_OVERLAP_POLICY_MATRIX: dict[tuple[str, str], str] = {
    _overlap_policy_key("payload_window", "payload_window"): "forbidden",
    _overlap_policy_key("primer_binding_core", "primer_binding_core"): "allowed_if_identical_span",
    _overlap_policy_key("restriction_site", "restriction_site"): "allowed_if_identical_span",
    _overlap_policy_key("nickase_site", "nickase_site"): "allowed_if_identical_span",
    _overlap_policy_key("homology_window", "homology_window"): "allowed_if_identical_span",
    _overlap_policy_key("retained_region", "retained_region"): "allowed_if_identical_span",
    _overlap_policy_key("sacrificial_region", "sacrificial_region"): "allowed_if_identical_span",
    _overlap_policy_key("payload_window", "retained_region"): "allowed_if_identical_span",
    _overlap_policy_key("payload_window", "homology_window"): "allowed_if_identical_span",
    _overlap_policy_key("retained_region", "homology_window"): "allowed_if_identical_span",
    _overlap_policy_key("primer_binding_core", "restriction_site"): "forbidden",
    _overlap_policy_key("primer_binding_core", "nickase_site"): "forbidden",
    _overlap_policy_key("retained_region", "sacrificial_region"): "forbidden",
    _overlap_policy_key("payload_window", "sacrificial_region"): "forbidden",
    _overlap_policy_key("snapback_seed", "sacrificial_region"): "forbidden",
    _overlap_policy_key("named_region", "payload_window"): "allowed_with_override",
    _overlap_policy_key("named_region", "retained_region"): "allowed_with_override",
}


def _v2_annotation_rows(spec: YiuProcessSpecV2) -> list[dict[str, Any]]:
    annotations = spec.source_oligo.annotations
    rows: list[dict[str, Any]] = []
    collection_specs = (
        ("primer_binding_core", annotations.primer_binding_cores),
        ("restriction_site", annotations.restriction_sites),
        ("nickase_site", annotations.nickase_sites),
        ("payload_window", annotations.payload_windows),
        ("homology_window", annotations.homology_windows),
        ("retained_region", annotations.retained_regions),
        ("sacrificial_region", annotations.sacrificial_regions),
    )
    for annotation_kind, collection in collection_specs:
        for item in collection:
            rows.append(
                {
                    "id": item.id,
                    "kind": annotation_kind,
                    "start": item.start,
                    "end": _item_end(item),
                }
            )
    for region in annotations.named_regions:
        rows.append(
            {
                "id": region.id,
                "kind": str(region.annotation_class or "named_region"),
                "start": region.start,
                "end": region.end,
            }
        )
    return rows


def _overlap_relation(left: dict[str, Any], right: dict[str, Any]) -> str:
    if int(left["start"]) == int(right["start"]) and int(left["end"]) == int(right["end"]):
        return "identical"
    left_contains_right = int(left["start"]) <= int(right["start"]) and int(left["end"]) >= int(right["end"])
    right_contains_left = int(right["start"]) <= int(left["start"]) and int(right["end"]) >= int(left["end"])
    if left_contains_right or right_contains_left:
        return "nested"
    return "partial"


def _override_allows_relation(override_mode: str, relation: str) -> bool:
    if override_mode == "allow_partial":
        return relation in {"partial", "nested", "identical"}
    if override_mode == "allow_nested":
        return relation in {"nested", "identical"}
    if override_mode == "allow_equal":
        return relation == "identical"
    return False


def _v2_overlap_issues(spec: YiuProcessSpecV2) -> list[YiuValidationIssue]:
    issues: list[YiuValidationIssue] = []
    source_state_id = "source_oligo_ssdna"
    overrides = {
        frozenset({override.left_annotation_id, override.right_annotation_id}): override
        for override in spec.source_oligo.annotations.overlap_overrides
    }
    annotation_rows = _v2_annotation_rows(spec)
    for index, left in enumerate(annotation_rows):
        for right in annotation_rows[index + 1 :]:
            if not _ranges_overlap(int(left["start"]), int(left["end"]), int(right["start"]), int(right["end"])):
                continue
            pair_key = _overlap_policy_key(str(left["kind"]), str(right["kind"]))
            policy = _YIU_OVERLAP_POLICY_MATRIX.get(pair_key, "allowed")
            override = overrides.get(frozenset({str(left["id"]), str(right["id"])}))
            relation = _overlap_relation(left, right)
            if override is not None and policy != "allowed_with_override":
                issues.append(
                    _issue(
                        "YIU_OVERLAP_OVERRIDE_NOT_ALLOWED",
                        f"overlap override for {left['id']} and {right['id']} is not allowed by the declared "
                        f"policy for {left['kind']} x {right['kind']}",
                        state_id=source_state_id,
                    )
                )
                continue
            if policy == "allowed":
                continue
            if policy == "forbidden":
                issues.append(
                    _issue(
                        "YIU_OVERLAP_FORBIDDEN",
                        f"annotations {left['id']} ({left['kind']}) and {right['id']} ({right['kind']}) overlap",
                        state_id=source_state_id,
                    )
                )
                continue
            if policy == "allowed_if_identical_span" and relation != "identical":
                issues.append(
                    _issue(
                        "YIU_OVERLAP_POLICY_VIOLATION",
                        f"annotations {left['id']} and {right['id']} overlap as {relation}; "
                        "identical spans are required",
                        state_id=source_state_id,
                    )
                )
                continue
            if policy == "allowed_if_nested" and relation not in {"nested", "identical"}:
                issues.append(
                    _issue(
                        "YIU_OVERLAP_POLICY_VIOLATION",
                        f"annotations {left['id']} and {right['id']} overlap as {relation}; nested spans are required",
                        state_id=source_state_id,
                    )
                )
                continue
            if policy == "allowed_with_override":
                if override is None:
                    issues.append(
                        _issue(
                            "YIU_OVERLAP_OVERRIDE_REQUIRED",
                            f"annotations {left['id']} and {right['id']} require overlap_overrides to overlap",
                            state_id=source_state_id,
                        )
                    )
                    continue
                if not _override_allows_relation(override.mode, relation):
                    issues.append(
                        _issue(
                            "YIU_OVERLAP_OVERRIDE_RELAXATION_INVALID",
                            f"overlap override {override.mode} does not permit the {relation} overlap between "
                            f"{left['id']} and {right['id']}",
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
