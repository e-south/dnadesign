"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/app/yiu_workflow/v1_report.py

Legacy YIU v1 report builder.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any

from dnadesign.cruncher.app.yiu_workflow.helpers import (
    _best_bulged_sticky_end_match,
    _best_contiguous_sticky_end_match,
    _branched_state_arms,
    _compatible_sequence,
    _issue,
    _joined_region_segments,
    _match_sort_key,
    _nickase_lookup,
    _primer_lookup,
    _project_region_to_segments,
    _project_region_to_state,
    _projected_annotations,
    _projected_region_overlaps_interval,
    _projected_region_payload,
    _projected_region_sequence,
    _region_lookup,
    _resolve_region,
    _resolve_region_list,
    _restriction_lookup,
    _segments_for_projected_regions,
    _segments_for_source_regions,
    _sequence_for_region,
    _sequence_mode_for_values,
    _state,
    _StickyEndMatch,
    _validate_annotation_overlaps,
)
from dnadesign.cruncher.app.yiu_workflow.split_support import _catalog_site_issue, _resolve_adapter_sequence
from dnadesign.cruncher.bio import (
    derive_cut_geometry,
    longest_reverse_complement_overlap,
    motif_matches,
    reverse_complement_iupac,
    sequence_contains_iupac,
)
from dnadesign.cruncher.yiu.catalog import LoadedYiuCatalogs
from dnadesign.cruncher.yiu.models import (
    RegionSpec,
    YiuProcessSpec,
    YiuReportMetadata,
    YiuStateRecord,
    YiuValidationIssue,
    YiuValidationReport,
)


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
