"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/app/yiu_workflow/split_support.py

Split-template invariant helpers shared by v1 and v2 report builders.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any

from dnadesign.cruncher.app.yiu_workflow.helpers import (
    _evaluate_ligation_compatibility,
    _issue,
    _iupac_match_status,
    _projected_region_sequence,
    _sequence_contains_status,
    _sequence_for_region,
    _split_template_bindings,
    _StateSegment,
    _StickyEndMatch,
    _terminal_tails,
    _v2_primer_core_lookup,
)
from dnadesign.cruncher.bio import derive_cut_geometry, reverse_complement_iupac
from dnadesign.cruncher.yiu.catalog import LoadedYiuCatalogs
from dnadesign.cruncher.yiu.models import (
    CompoundRegionRef,
    EnzymeSiteSpec,
    LigationRuleSpec,
    ProjectedRegion,
    ProjectedRegionPart,
    RegionSpec,
    RegionSpecV2,
    YiuHardInvariant,
    YiuOligoPartCatalogEntry,
    YiuProcessSpec,
    YiuProcessSpecV2,
    YiuValidationIssue,
)


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


def _ligation_rule_match(left: str, right: str, rule: LigationRuleSpec) -> _StickyEndMatch | None:
    if rule.mode == "exact_complement":
        return _evaluate_ligation_compatibility(left, right, mode="exact_complement")
    if rule.mode == "partial_complement":
        candidate = _evaluate_ligation_compatibility(
            left,
            right,
            mode="partial_complement",
            partial_rule={
                "min_paired_nt": rule.min_contiguous_core_bp,
                "allow_left_tail": True,
                "allow_right_tail": True,
            },
        )
    else:
        candidate = _evaluate_ligation_compatibility(
            left,
            right,
            mode="bulged",
            bulged_rule={
                "min_left_paired_nt": max(1, rule.min_left_flank_bp or rule.min_contiguous_core_bp),
                "min_right_paired_nt": max(1, rule.min_right_flank_bp or rule.min_contiguous_core_bp),
                "max_bulge_nt": rule.max_bulge_nt,
                "allow_terminal_tails": True,
            },
        )
    if candidate is None:
        return None
    left_tail_nt, right_tail_nt = _terminal_tails(candidate, left_length=len(left), right_length=len(right))
    if left_tail_nt > rule.max_left_tail_nt or right_tail_nt > rule.max_right_tail_nt:
        return None
    if candidate.bulge_nt > rule.max_bulge_nt:
        return None
    return candidate


def _project_region_to_segments_optional(
    region: RegionSpec,
    segments: list[_StateSegment],
    *,
    state_id: str,
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
        return None
    return ProjectedRegion(
        id=f"{state_id}:{region.id}",
        source_region_id=region.id,
        state_id=state_id,
        spans_junction=len(parts) > 1,
        parts=parts,
    )


def _resolve_region_sequence_in_state(
    *,
    region_ref: str,
    target_state_id: str,
    source_sequence: str,
    regions: dict[str, RegionSpecV2],
    state_sequences: dict[str, str],
    state_segments_by_id: dict[str, list[_StateSegment]],
) -> tuple[str | None, ProjectedRegion | None]:
    region = regions.get(region_ref)
    if region is None:
        return None, None
    if target_state_id == "source_oligo_ssdna":
        projected = ProjectedRegion(
            id=f"{target_state_id}:{region.id}",
            source_region_id=region.id,
            state_id=target_state_id,
            spans_junction=False,
            parts=[ProjectedRegionPart(segment_id=region.id, start=region.start, end=region.end)],
        )
        return _sequence_for_region(source_sequence, region), projected
    state_sequence = state_sequences.get(target_state_id)
    state_segments = state_segments_by_id.get(target_state_id)
    if state_sequence is None or state_segments is None:
        return None, None
    projected = _project_region_to_segments_optional(region, state_segments, state_id=target_state_id)
    if projected is None:
        return None, None
    return _projected_region_sequence(state_sequence, projected), projected


def _resolve_compound_region_observed(
    compound_region: CompoundRegionRef,
    *,
    source_sequence: str,
    regions: dict[str, RegionSpecV2],
    state_sequences: dict[str, str],
    state_segments_by_id: dict[str, list[_StateSegment]],
    step_id: str,
    issues: list[YiuValidationIssue],
) -> dict[str, Any] | None:
    if compound_region.join_policy not in {"concatenate", "junction_assemble"}:
        issues.append(
            _issue(
                "YIU_COMPOUND_REGION_JOIN_POLICY_UNSUPPORTED",
                f"compound region {compound_region.id} join_policy {compound_region.join_policy!r} is unsupported",
                step_id=step_id,
            )
        )
        return None
    resolved_parts: list[dict[str, Any]] = []
    sequence_parts: list[str] = []
    for segment in compound_region.segments:
        segment_sequence, projected = _resolve_region_sequence_in_state(
            region_ref=segment.source_region_ref,
            target_state_id=segment.source_state,
            source_sequence=source_sequence,
            regions=regions,
            state_sequences=state_sequences,
            state_segments_by_id=state_segments_by_id,
        )
        if segment_sequence is None:
            issue_code = (
                "YIU_COMPOUND_REGION_SOURCE_STATE_UNSUPPORTED"
                if segment.source_state not in state_sequences
                else "YIU_COMPOUND_REGION_UNPROJECTABLE"
            )
            issues.append(
                _issue(
                    issue_code,
                    f"compound region {compound_region.id} segment {segment.source_region_ref!r} could not be "
                    f"resolved from source_state {segment.source_state!r}",
                    step_id=step_id,
                )
            )
            return None
        if segment.orientation == "reverse_complement":
            segment_sequence = reverse_complement_iupac(segment_sequence)
        resolved_parts.append(
            {
                "source_state": segment.source_state,
                "source_region_ref": segment.source_region_ref,
                "orientation": segment.orientation,
                "projected_region": projected.model_dump(mode="json") if projected is not None else None,
                "sequence": segment_sequence,
            }
        )
        sequence_parts.append(segment_sequence)
    if compound_region.join_policy == "junction_assemble" and len(sequence_parts) < 2:
        issues.append(
            _issue(
                "YIU_COMPOUND_REGION_JOIN_POLICY_UNSUPPORTED",
                f"compound region {compound_region.id} requires at least two segments for junction_assemble",
                step_id=step_id,
            )
        )
        return None
    return {
        "id": compound_region.id,
        "join_policy": compound_region.join_policy,
        "segments": resolved_parts,
        "sequence": "".join(sequence_parts),
    }


def _ligation_match_observed(match: _StickyEndMatch | None) -> dict[str, Any]:
    if match is None:
        return {"matched": False}
    return {
        "matched": True,
        "paired_nt": match.paired_nt,
        "left_start": match.left_start,
        "left_end": match.left_end,
        "right_start": match.right_start,
        "right_end": match.right_end,
        "unpaired_tail_nt": match.unpaired_tail_nt,
        "bulge_nt": match.bulge_nt,
        "bulge_side": match.bulge_side,
    }


def _hard_invariant_result(
    invariant: YiuHardInvariant,
    *,
    status: str,
    observed: Any,
) -> dict[str, Any]:
    return {
        "id": invariant.id,
        "class": invariant.class_,
        "status": status,
        "space_kind": invariant.space_kind,
        "state_ref": invariant.state_ref,
        "transform_ref": invariant.transform_ref,
        "region_ref": invariant.region_ref,
        "observed": observed,
    }


def _append_hard_invariant_issue(
    invariant: YiuHardInvariant,
    *,
    status: str,
    issues: list[YiuValidationIssue],
) -> None:
    if status == "guaranteed":
        return
    issues.append(
        _issue(
            "YIU_HARD_INVARIANT_NOT_GUARANTEED",
            f"hard invariant {invariant.id} is {status}",
            step_id=invariant.transform_ref,
            state_id=invariant.state_ref,
        )
    )


def _evaluate_split_hard_invariants(
    spec: YiuProcessSpecV2,
    *,
    catalogs: LoadedYiuCatalogs,
    source_sequence: str,
    regions: dict[str, RegionSpecV2],
    sites: dict[str, EnzymeSiteSpec],
    compound_regions: dict[str, CompoundRegionRef],
    assembled_payload: str,
    fragment_lengths: list[int],
    retained_product: str,
    state_sequences: dict[str, str],
    state_segments_by_id: dict[str, list[_StateSegment]],
    ligation_matches_by_step: dict[str, _StickyEndMatch | None],
    adapter_parts_by_step: dict[str, YiuOligoPartCatalogEntry],
    state_id: str,
    step_id: str,
    issues: list[YiuValidationIssue],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    compound_region_ids = set(compound_regions)
    bindings = _split_template_bindings(spec)
    primer_cores = _v2_primer_core_lookup(spec)
    for invariant in spec.hard_invariants:
        if invariant.state_ref is not None and invariant.state_ref != state_id:
            continue
        if invariant.transform_ref is not None and invariant.transform_ref != step_id:
            continue
        status = "impossible"
        observed: Any = None
        target_state_id = invariant.state_ref or state_id
        expected_pattern = str(
            invariant.params.get("expected_pattern") or invariant.params.get("sequence_pattern") or ""
        )
        observed_sequence: str | None = None
        projected_region: ProjectedRegion | None = None
        if invariant.region_ref is not None:
            if invariant.region_ref in compound_region_ids:
                observed = _resolve_compound_region_observed(
                    compound_regions[invariant.region_ref],
                    source_sequence=source_sequence,
                    regions=regions,
                    state_sequences=state_sequences,
                    state_segments_by_id=state_segments_by_id,
                    step_id=step_id,
                    issues=issues,
                )
                if observed is not None:
                    observed_sequence = str(observed["sequence"])
            else:
                observed_sequence, projected_region = _resolve_region_sequence_in_state(
                    region_ref=invariant.region_ref,
                    target_state_id=target_state_id,
                    source_sequence=source_sequence,
                    regions=regions,
                    state_sequences=state_sequences,
                    state_segments_by_id=state_segments_by_id,
                )
        if invariant.class_ == "payload_assembly":
            if observed_sequence is None:
                observed_sequence = assembled_payload
                observed = assembled_payload
            status = _iupac_match_status(str(observed_sequence), expected_pattern)
            if observed is None:
                observed = observed_sequence
        elif invariant.class_ == "region_pattern":
            observed = {
                "sequence": observed_sequence,
                "projected_region": projected_region.model_dump(mode="json") if projected_region is not None else None,
            }
            if observed_sequence is not None and expected_pattern:
                status = _iupac_match_status(observed_sequence, expected_pattern)
        elif invariant.class_ == "sacrificial_fragmentation":
            observed = {"fragment_lengths": fragment_lengths, "site_count": len(fragment_lengths) - 1}
            max_fragment_nt = int(invariant.params.get("max_fragment_nt", 0))
            min_site_count = int(invariant.params.get("min_site_count", 0))
            if (
                fragment_lengths
                and max(fragment_lengths) <= max_fragment_nt
                and (len(fragment_lengths) - 1) >= min_site_count
            ):
                status = "guaranteed"
        elif invariant.class_ == "snapback_exposure":
            sequence_pattern = str(invariant.params.get("sequence_pattern") or expected_pattern)
            observed_text = observed_sequence or str(state_sequences.get(target_state_id, "") or "")
            require_free_five_prime_end = bool(invariant.params.get("require_free_five_prime_end", False))
            observed = {
                "sequence": observed_text,
                "projected_region": projected_region.model_dump(mode="json") if projected_region is not None else None,
            }
            if require_free_five_prime_end:
                if projected_region is not None and projected_region.parts and projected_region.parts[0].start != 0:
                    status = "impossible"
                else:
                    status = _iupac_match_status(observed_text[: len(sequence_pattern)], sequence_pattern)
            else:
                status = _sequence_contains_status(observed_text, sequence_pattern)
        elif invariant.class_ == "retained_survival":
            if observed_sequence is None:
                observed_sequence = retained_product
            expected = expected_pattern or observed_sequence
            observed = {
                "sequence": observed_sequence,
                "projected_region": projected_region.model_dump(mode="json") if projected_region is not None else None,
            }
            status = _iupac_match_status(observed_sequence, expected)
        elif invariant.class_ == "ligation_compatibility":
            match = ligation_matches_by_step.get(step_id)
            observed = {"state": target_state_id, **_ligation_match_observed(match)}
            min_paired_nt = int(invariant.params.get("min_paired_nt", 0))
            if match is not None and match.paired_nt >= min_paired_nt:
                status = "guaranteed"
        elif invariant.class_ == "enzyme_site":
            site_ref = str(invariant.params.get("site_ref") or "")
            site = sites.get(site_ref)
            if site is not None:
                site_sequence, projected_site = _resolve_region_sequence_in_state(
                    region_ref=site.id,
                    target_state_id=target_state_id,
                    source_sequence=source_sequence,
                    regions={**regions, site.id: RegionSpecV2(id=site.id, start=site.start, end=site.end)},
                    state_sequences=state_sequences,
                    state_segments_by_id=state_segments_by_id,
                )
                observed = {
                    "site_id": site.id,
                    "enzyme_id": site.enzyme,
                    "sequence": site_sequence,
                    "projected_region": projected_site.model_dump(mode="json") if projected_site is not None else None,
                }
                if (
                    site_sequence is not None
                    and _iupac_match_status(site_sequence, site.recognition_sequence) == "guaranteed"
                ):
                    status = "guaranteed"
        elif invariant.class_ == "cut_geometry":
            site_ref = str(invariant.params.get("site_ref") or "")
            site = sites.get(site_ref)
            if site is not None:
                site_sequence, projected_site = _resolve_region_sequence_in_state(
                    region_ref=site.id,
                    target_state_id=target_state_id,
                    source_sequence=source_sequence,
                    regions={**regions, site.id: RegionSpecV2(id=site.id, start=site.start, end=site.end)},
                    state_sequences=state_sequences,
                    state_segments_by_id=state_segments_by_id,
                )
                if site_sequence is not None and projected_site is not None and projected_site.parts:
                    try:
                        geometry = derive_cut_geometry(
                            state_sequences[target_state_id],
                            start=projected_site.parts[0].start,
                            recognition_sequence=site.recognition_sequence,
                            orientation=site.orientation,
                            top_cut_offset=site.top_cut_offset,
                            bottom_cut_offset=site.bottom_cut_offset,
                        )
                    except ValueError:
                        geometry = None
                    observed = {
                        "site_id": site.id,
                        "projected_region": projected_site.model_dump(mode="json"),
                        "geometry": (
                            {
                                "top_boundary": geometry.top_boundary,
                                "bottom_boundary": geometry.bottom_boundary,
                                "overhang_sequence": geometry.overhang_sequence,
                            }
                            if geometry is not None
                            else None
                        ),
                    }
                    if geometry is not None:
                        expected_overhang = invariant.params.get("expected_overhang_sequence")
                        expected_top = invariant.params.get("expected_top_boundary")
                        expected_bottom = invariant.params.get("expected_bottom_boundary")
                        status = "guaranteed"
                        if expected_overhang is not None and geometry.overhang_sequence != str(expected_overhang):
                            status = "impossible"
                        if expected_top is not None and geometry.top_boundary != int(expected_top):
                            status = "impossible"
                        if expected_bottom is not None and geometry.bottom_boundary != int(expected_bottom):
                            status = "impossible"
        elif invariant.class_ == "adapter_binding":
            match = ligation_matches_by_step.get(step_id)
            adapter_part = adapter_parts_by_step.get(step_id)
            observed = {
                "adapter_id": adapter_part.id if adapter_part is not None else None,
                "phosphorylated_5p": adapter_part.phosphorylated_5p if adapter_part is not None else None,
                **_ligation_match_observed(match),
            }
            require_5p = bool(invariant.params.get("require_5p_phosphate", False))
            if match is not None and (not require_5p or (adapter_part is not None and adapter_part.phosphorylated_5p)):
                status = "guaranteed"
        elif invariant.class_ == "primer_binding":
            forward_ref = bindings.source_forward_primer_core_ref
            reverse_ref = bindings.source_reverse_primer_core_ref
            primer_side = str(invariant.params.get("primer_side") or "both")
            expected_refs = [forward_ref, reverse_ref]
            if primer_side == "forward":
                expected_refs = [forward_ref]
            elif primer_side == "reverse":
                expected_refs = [reverse_ref]
            primer_presence = {}
            for region_ref in expected_refs:
                primer_core = primer_cores.get(region_ref)
                extended_regions = regions
                if primer_core is not None and region_ref not in regions:
                    extended_regions = {
                        **regions,
                        region_ref: RegionSpecV2(id=primer_core.id, start=primer_core.start, end=primer_core.end),
                    }
                region_sequence, projected = _resolve_region_sequence_in_state(
                    region_ref=region_ref,
                    target_state_id=target_state_id,
                    source_sequence=source_sequence,
                    regions=extended_regions,
                    state_sequences=state_sequences,
                    state_segments_by_id=state_segments_by_id,
                )
                primer_presence[region_ref] = {
                    "sequence": region_sequence,
                    "projected_region": projected.model_dump(mode="json") if projected is not None else None,
                }
            observed = {"primer_core_refs": primer_presence}
            if primer_presence and all(item["sequence"] is not None for item in primer_presence.values()):
                status = "guaranteed"
        result = _hard_invariant_result(invariant, status=status, observed=observed)
        results.append(result)
        _append_hard_invariant_issue(invariant, status=status, issues=issues)
    return results
