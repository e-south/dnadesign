"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/app/yiu_workflow/split_template.py

YIU circularized-payload v2 report builder.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any

from dnadesign.cruncher.app.yiu_workflow.helpers import (
    _compound_annotation,
    _has_error_issue,
    _issue,
    _segment_rows,
    _segments_for_source_regions,
    _sequence_for_region,
    _split_template_bindings,
    _state,
    _StateSegment,
    _v2_overlap_issues,
    _v2_part,
    _v2_primer_core_lookup,
    _v2_region_lookup,
    _v2_report_sequence_mode,
    _v2_site_lookup,
)
from dnadesign.cruncher.app.yiu_workflow.split_support import (
    _evaluate_split_hard_invariants,
    _ligation_rule_match,
)
from dnadesign.cruncher.bio import derive_cut_geometry, reverse_complement_iupac
from dnadesign.cruncher.yiu.catalog import LoadedYiuCatalogs
from dnadesign.cruncher.yiu.models import (
    RegionSpec,
    YiuProcessSpecV2,
    YiuReportMetadata,
    YiuStateRecord,
    YiuValidationIssue,
    YiuValidationReport,
)


def _build_yiu_report_v2_split_template(
    spec: YiuProcessSpecV2,
    *,
    catalogs: LoadedYiuCatalogs | None = None,
) -> YiuValidationReport:
    catalogs = catalogs or LoadedYiuCatalogs()
    issues: list[YiuValidationIssue] = []
    states: list[YiuStateRecord] = []
    sequence = spec.source_oligo.sequence or ""
    annotations = spec.source_oligo.annotations
    bindings = _split_template_bindings(spec)
    primer_cores = _v2_primer_core_lookup(spec)
    regions = _v2_region_lookup(spec)
    sites = _v2_site_lookup(spec)
    compound_regions = {region.id: region for region in spec.compound_regions}
    left_region = regions[spec.payload_goal.left_half_ref]
    right_region = regions[spec.payload_goal.right_half_ref]
    assembled_payload = _sequence_for_region(sequence, left_region) + _sequence_for_region(sequence, right_region)
    issues.extend(_v2_overlap_issues(spec))
    state_sequences_by_id: dict[str, str] = {"source_oligo_ssdna": sequence}
    state_segments_by_id: dict[str, list[_StateSegment]] = {}

    source_segments = _segments_for_source_regions([RegionSpec(id="source", start=0, end=len(sequence))])
    state_segments_by_id["source_oligo_ssdna"] = source_segments
    source_invariants = _evaluate_split_hard_invariants(
        spec,
        catalogs=catalogs,
        source_sequence=sequence,
        regions=regions,
        sites=sites,
        compound_regions=compound_regions,
        assembled_payload=assembled_payload,
        fragment_lengths=[],
        retained_product="",
        state_sequences=state_sequences_by_id,
        state_segments_by_id=state_segments_by_id,
        ligation_matches_by_step={},
        adapter_parts_by_step={},
        state_id="source_oligo_ssdna",
        step_id="source_oligo",
        issues=issues,
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
            metadata={
                "length_nt": len(sequence),
                "authored_sequence": spec.source_oligo.authored_sequence,
                "hard_invariants": source_invariants,
            },
            view_contract_version=spec.output.publish_contract_version,
            segments=_segment_rows(source_segments),
            annotations=[
                {
                    "id": region.id,
                    "annotation_class": region.annotation_class or "region",
                    "start": region.start,
                    "end": region.end,
                }
                for region in (
                    annotations.payload_windows
                    + annotations.retained_regions
                    + annotations.sacrificial_regions
                    + annotations.named_regions
                )
            ],
            pattern_label="pattern",
        )
    )

    forward_core = primer_cores.get(bindings.source_forward_primer_core_ref)
    reverse_core = primer_cores.get(bindings.source_reverse_primer_core_ref)
    pcr_issues: list[YiuValidationIssue] = []
    if forward_core is None or reverse_core is None:
        pcr_issues.append(
            _issue("PCR_PRIMER_SITE_MISSING", "source_pcr primer binding core is missing", step_id="source_pcr")
        )
        amplicon_start = 0
        amplicon_end = len(sequence)
    else:
        amplicon_start = forward_core.start
        amplicon_end = reverse_core.end
    amplicon_primary = sequence[amplicon_start:amplicon_end]
    amplicon_complement = reverse_complement_iupac(amplicon_primary)
    amplicon_segments = _segments_for_source_regions(
        [RegionSpec(id="source_amplicon", start=amplicon_start, end=amplicon_end)]
    )
    state_sequences_by_id["pcr_linear_duplex"] = amplicon_primary
    state_segments_by_id["pcr_linear_duplex"] = amplicon_segments
    pcr_invariants = _evaluate_split_hard_invariants(
        spec,
        catalogs=catalogs,
        source_sequence=sequence,
        regions=regions,
        sites=sites,
        compound_regions=compound_regions,
        assembled_payload=assembled_payload,
        fragment_lengths=[],
        retained_product="",
        state_sequences=state_sequences_by_id,
        state_segments_by_id=state_segments_by_id,
        ligation_matches_by_step={},
        adapter_parts_by_step={},
        state_id="pcr_linear_duplex",
        step_id="source_pcr",
        issues=pcr_issues,
    )
    states.append(
        _state(
            state_id="pcr_linear_duplex",
            step_id="source_pcr",
            kind="pcr_linear_duplex",
            state_kind="pcr_linear_duplex",
            topology_kind="linear_dsdna",
            status="unsatisfied" if _has_error_issue(pcr_issues) else "satisfied",
            primary_sequence=amplicon_primary,
            complement_sequence=amplicon_complement,
            metadata={
                "amplicon_start": amplicon_start,
                "amplicon_end": amplicon_end,
                "hard_invariants": pcr_invariants,
            },
            view_contract_version=spec.output.publish_contract_version,
            segments=_segment_rows(amplicon_segments),
            pattern_label="pattern",
        )
    )
    issues.extend(pcr_issues)

    digest_issues: list[YiuValidationIssue] = []
    digest_cuts: list[dict[str, Any]] = []
    digest_geometry_by_site: dict[str, Any] = {}
    selected_site_ids = set(spec.steps.type_iis_digest.site_ids if spec.steps.type_iis_digest is not None else [])
    for site in annotations.restriction_sites:
        if site.id not in selected_site_ids:
            continue
        site_start = site.start - amplicon_start
        try:
            geometry = derive_cut_geometry(
                amplicon_primary,
                start=site_start,
                recognition_sequence=site.recognition_sequence,
                orientation=site.orientation,
                top_cut_offset=site.top_cut_offset,
                bottom_cut_offset=site.bottom_cut_offset,
            )
        except ValueError as exc:
            digest_issues.append(
                _issue("TYPE_IIS_SITE_INVALID", f"type IIS site {site.id} is invalid: {exc}", step_id="type_iis_digest")
            )
            continue
        digest_geometry_by_site[site.id] = geometry
        digest_cuts.append(
            {
                "site_id": site.id,
                "enzyme_id": site.enzyme,
                "top_boundary": geometry.top_boundary,
                "bottom_boundary": geometry.bottom_boundary,
                "overhang_sequence": geometry.overhang_sequence,
            }
        )
    digest_invariants = _evaluate_split_hard_invariants(
        spec,
        catalogs=catalogs,
        source_sequence=sequence,
        regions=regions,
        sites=sites,
        compound_regions=compound_regions,
        assembled_payload=assembled_payload,
        fragment_lengths=[],
        retained_product="",
        state_sequences={**state_sequences_by_id, "type_iis_digest_linear_duplex": amplicon_primary},
        state_segments_by_id={**state_segments_by_id, "type_iis_digest_linear_duplex": amplicon_segments},
        ligation_matches_by_step={},
        adapter_parts_by_step={},
        state_id="type_iis_digest_linear_duplex",
        step_id="type_iis_digest",
        issues=digest_issues,
    )
    states.append(
        _state(
            state_id="type_iis_digest_linear_duplex",
            step_id="type_iis_digest",
            kind="type_iis_digest_linear_duplex",
            state_kind="type_iis_digest_linear_duplex",
            topology_kind="linear_dsdna",
            status="unsatisfied" if _has_error_issue(digest_issues) else "satisfied",
            primary_sequence=amplicon_primary,
            complement_sequence=amplicon_complement,
            metadata={
                "enzyme_id": spec.steps.type_iis_digest.enzyme_id if spec.steps.type_iis_digest is not None else None,
                "hard_invariants": digest_invariants,
            },
            view_contract_version=spec.output.publish_contract_version,
            segments=_segment_rows(amplicon_segments),
            cuts=digest_cuts,
            pattern_label="pattern",
        )
    )
    state_sequences_by_id["type_iis_digest_linear_duplex"] = amplicon_primary
    state_segments_by_id["type_iis_digest_linear_duplex"] = amplicon_segments
    issues.extend(digest_issues)

    circularized_primary = sequence[: left_region.start] + assembled_payload + sequence[right_region.end :]
    circularized_complement = reverse_complement_iupac(circularized_primary)
    circularized_state_sequences = {
        "circularized_payload_candidate": circularized_primary,
    }
    circularization_issues: list[YiuValidationIssue] = []
    circularization_match = None
    left_digest_geometry = digest_geometry_by_site.get(bindings.circularization_left_overhang_ref)
    right_digest_geometry = digest_geometry_by_site.get(bindings.circularization_right_overhang_ref)
    if (
        spec.steps.circularization is not None
        and left_digest_geometry is not None
        and right_digest_geometry is not None
    ):
        circularization_match = _ligation_rule_match(
            left_digest_geometry.overhang_sequence,
            right_digest_geometry.overhang_sequence,
            spec.steps.circularization.ligation_rule,
        )
        if circularization_match is None:
            circularization_issues.append(
                _issue(
                    "CIRCULARIZATION_COMPATIBILITY_FAIL",
                    "type IIS overhangs do not satisfy the configured ligation rule",
                    step_id="circularization",
                )
            )
    elif spec.steps.circularization is not None:
        circularization_issues.append(
            _issue(
                "YIU_TEMPLATE_BINDING_REF_UNKNOWN",
                "circularization overhang bindings did not resolve to selected restriction sites",
                step_id="circularization",
            )
        )
    state_sequences_for_circularization = {**state_sequences_by_id, **circularized_state_sequences}
    circularized_invariants = _evaluate_split_hard_invariants(
        spec,
        catalogs=catalogs,
        source_sequence=sequence,
        regions=regions,
        sites=sites,
        compound_regions=compound_regions,
        assembled_payload=assembled_payload,
        fragment_lengths=[],
        retained_product="",
        state_sequences=state_sequences_for_circularization,
        state_segments_by_id={
            **state_segments_by_id,
            "circularized_payload_candidate": [
                _StateSegment("source_prefix", 0, left_region.start, 0, left_region.start),
                _StateSegment("payload_left", left_region.start, left_region.end, left_region.start, left_region.end),
                _StateSegment(
                    "payload_right",
                    right_region.start,
                    right_region.end,
                    left_region.end,
                    left_region.end + (right_region.end - right_region.start),
                ),
                _StateSegment(
                    "post_payload_suffix",
                    right_region.end,
                    len(sequence),
                    left_region.end + (right_region.end - right_region.start),
                    len(circularized_primary),
                ),
            ],
        },
        ligation_matches_by_step={"circularization": circularization_match},
        adapter_parts_by_step={},
        state_id="circularized_payload_candidate",
        step_id="circularization",
        issues=circularization_issues,
    )
    circularized_segments = [
        _StateSegment("source_prefix", 0, left_region.start, 0, left_region.start),
        _StateSegment("payload_left", left_region.start, left_region.end, left_region.start, left_region.end),
        _StateSegment(
            "payload_right",
            right_region.start,
            right_region.end,
            left_region.end,
            left_region.end + (right_region.end - right_region.start),
        ),
        _StateSegment(
            "post_payload_suffix",
            right_region.end,
            len(sequence),
            left_region.end + (right_region.end - right_region.start),
            len(circularized_primary),
        ),
    ]
    states.append(
        _state(
            state_id="circularized_payload_candidate",
            step_id="circularization",
            kind="circularized_payload_candidate",
            state_kind="circularized_payload_candidate",
            topology_kind="circular_dsdna_candidate",
            status="unsatisfied" if _has_error_issue(circularization_issues) else "satisfied",
            primary_sequence=circularized_primary,
            complement_sequence=circularized_complement,
            metadata={
                "assembly_space": spec.payload_goal.assembly_space,
                "assembled_payload": assembled_payload,
                "paired_nt": circularization_match.paired_nt if circularization_match is not None else 0,
                "hard_invariants": circularized_invariants,
            },
            view_contract_version=spec.output.publish_contract_version,
            segments=_segment_rows(circularized_segments),
            junctions=[
                {
                    "id": "circularized_payload_junction",
                    "assembly_space": spec.payload_goal.assembly_space,
                    "join_index": left_region.end,
                }
            ],
            pattern_label="pattern",
        )
    )
    state_sequences_by_id["circularized_payload_candidate"] = circularized_primary
    state_segments_by_id["circularized_payload_candidate"] = circularized_segments
    issues.extend(circularization_issues)

    exonuclease_issues: list[YiuValidationIssue] = []
    exonuclease_invariants = _evaluate_split_hard_invariants(
        spec,
        catalogs=catalogs,
        source_sequence=sequence,
        regions=regions,
        sites=sites,
        compound_regions=compound_regions,
        assembled_payload=assembled_payload,
        fragment_lengths=[],
        retained_product="",
        state_sequences={**state_sequences_by_id, "post_exonuclease_cleanup": circularized_primary},
        state_segments_by_id={**state_segments_by_id, "post_exonuclease_cleanup": circularized_segments},
        ligation_matches_by_step={},
        adapter_parts_by_step={},
        state_id="post_exonuclease_cleanup",
        step_id="exonuclease_cleanup",
        issues=exonuclease_issues,
    )
    states.append(
        _state(
            state_id="post_exonuclease_cleanup",
            step_id="exonuclease_cleanup",
            kind="post_exonuclease_cleanup",
            state_kind="post_exonuclease_cleanup",
            topology_kind="circular_dsdna_candidate",
            status="satisfied",
            primary_sequence=circularized_primary,
            complement_sequence=circularized_complement,
            metadata={"enzyme": spec.steps.exonuclease_cleanup.enzyme, "hard_invariants": exonuclease_invariants},
            view_contract_version=spec.output.publish_contract_version,
            segments=_segment_rows(circularized_segments),
            pattern_label="pattern",
        )
    )
    state_sequences_by_id["post_exonuclease_cleanup"] = circularized_primary
    state_segments_by_id["post_exonuclease_cleanup"] = circularized_segments
    issues.extend(exonuclease_issues)

    fragment_boundaries: list[int] = []
    site_ids = set(spec.steps.sacrificial_digest.site_ids if spec.steps.sacrificial_digest is not None else [])
    for site in annotations.nickase_sites:
        if site.id not in site_ids:
            continue
        boundary = site.start + int(site.bottom_cut_offset or 0)
        fragment_boundaries.append(boundary)
    fragment_lengths: list[int] = []
    for sacrificial_region_id in bindings.primary_sacrificial_region_refs:
        sacrificial_sequence_region = regions[sacrificial_region_id]
        fragment_cuts = [
            sacrificial_sequence_region.start,
            *sorted(
                boundary
                for boundary in fragment_boundaries
                if sacrificial_sequence_region.start <= boundary <= sacrificial_sequence_region.end
            ),
            sacrificial_sequence_region.end,
        ]
        fragment_lengths.extend(
            fragment_cuts[index + 1] - fragment_cuts[index]
            for index in range(len(fragment_cuts) - 1)
            if fragment_cuts[index + 1] > fragment_cuts[index]
        )
    retained_region_ids = (
        spec.steps.sacrificial_digest.retained_region_ids if spec.steps.sacrificial_digest is not None else []
    )
    retained_regions = [regions[region_id] for region_id in retained_region_ids]
    retained_product = "".join(_sequence_for_region(sequence, region) for region in retained_regions)
    fragmentation_issues: list[YiuValidationIssue] = []
    fragmentation_state_sequences = {
        "post_sacrificial_fragmentation": circularized_primary,
        "post_fragment_cleanup": retained_product,
    }
    fragmentation_invariants = _evaluate_split_hard_invariants(
        spec,
        catalogs=catalogs,
        source_sequence=sequence,
        regions=regions,
        sites=sites,
        compound_regions=compound_regions,
        assembled_payload=assembled_payload,
        fragment_lengths=fragment_lengths,
        retained_product=retained_product,
        state_sequences={**state_sequences_by_id, **fragmentation_state_sequences},
        state_segments_by_id={**state_segments_by_id, "post_sacrificial_fragmentation": circularized_segments},
        ligation_matches_by_step={},
        adapter_parts_by_step={},
        state_id="post_sacrificial_fragmentation",
        step_id="sacrificial_digest",
        issues=fragmentation_issues,
    )
    states.append(
        _state(
            state_id="post_sacrificial_fragmentation",
            step_id="sacrificial_digest",
            kind="post_sacrificial_fragmentation",
            state_kind="post_sacrificial_fragmentation",
            topology_kind="fragment_pool",
            status="unsatisfied" if _has_error_issue(fragmentation_issues) else "satisfied",
            primary_sequence=circularized_primary,
            metadata={
                "fragment_lengths": fragment_lengths,
                "retained_product": retained_product,
                "hard_invariants": fragmentation_invariants,
            },
            view_contract_version=spec.output.publish_contract_version,
            fragments=[
                {"fragment_id": f"sacrificial_fragment_{index + 1}", "length_nt": length}
                for index, length in enumerate(fragment_lengths)
            ],
            pattern_label="pattern",
        )
    )
    state_sequences_by_id["post_sacrificial_fragmentation"] = circularized_primary
    state_segments_by_id["post_sacrificial_fragmentation"] = circularized_segments
    issues.extend(fragmentation_issues)

    cleanup_issues: list[YiuValidationIssue] = []
    if (
        spec.steps.fragment_cleanup.enabled
        and spec.steps.fragment_cleanup.max_fragment_nt is not None
        and fragment_lengths
        and max(fragment_lengths) > spec.steps.fragment_cleanup.max_fragment_nt
    ):
        cleanup_issues.append(
            _issue(
                "SACRIFICIAL_FRAGMENT_TOO_LARGE",
                "sacrificial fragments "
                f"{fragment_lengths} exceed max_fragment_nt "
                f"{spec.steps.fragment_cleanup.max_fragment_nt}",
                step_id="fragment_cleanup",
            )
        )
    if (
        spec.steps.fragment_cleanup.enabled
        and spec.steps.fragment_cleanup.min_retained_nt is not None
        and len(retained_product) < spec.steps.fragment_cleanup.min_retained_nt
    ):
        cleanup_issues.append(
            _issue(
                "RETAINED_PRODUCT_TOO_SHORT",
                "retained product length "
                f"{len(retained_product)} is below min_retained_nt "
                f"{spec.steps.fragment_cleanup.min_retained_nt}",
                step_id="fragment_cleanup",
            )
        )
    retained_segments: list[_StateSegment] = []
    cursor = 0
    for region in retained_regions:
        length = region.end - region.start
        retained_segments.append(
            _StateSegment(
                segment_id=region.id,
                source_start=region.start,
                source_end=region.end,
                state_start=cursor,
                state_end=cursor + length,
            )
        )
        cursor += length
    cleanup_invariants = _evaluate_split_hard_invariants(
        spec,
        catalogs=catalogs,
        source_sequence=sequence,
        regions=regions,
        sites=sites,
        compound_regions=compound_regions,
        assembled_payload=assembled_payload,
        fragment_lengths=fragment_lengths,
        retained_product=retained_product,
        state_sequences={**state_sequences_by_id, **fragmentation_state_sequences},
        state_segments_by_id={**state_segments_by_id, "post_fragment_cleanup": retained_segments},
        ligation_matches_by_step={},
        adapter_parts_by_step={},
        state_id="post_fragment_cleanup",
        step_id="fragment_cleanup",
        issues=cleanup_issues,
    )
    states.append(
        _state(
            state_id="post_fragment_cleanup",
            step_id="fragment_cleanup",
            kind="post_fragment_cleanup",
            state_kind="post_fragment_cleanup",
            topology_kind="linear_ssdna",
            status="unsatisfied" if _has_error_issue(cleanup_issues) else "satisfied",
            primary_sequence=retained_product,
            metadata={"hard_invariants": cleanup_invariants, "fragment_lengths": fragment_lengths},
            view_contract_version=spec.output.publish_contract_version,
            segments=_segment_rows(retained_segments),
            pattern_label="pattern",
        )
    )
    state_sequences_by_id["post_fragment_cleanup"] = retained_product
    state_segments_by_id["post_fragment_cleanup"] = retained_segments
    issues.extend(cleanup_issues)

    snapback_seed_sequence = _sequence_for_region(sequence, regions[bindings.snapback_seed_region_ref])
    adapter_part = _v2_part(
        catalogs, spec.steps.snapback_adapter_engagement.adapter_id, label="snapback_adapter_engagement.adapter_id"
    )
    engagement_issues: list[YiuValidationIssue] = []
    engagement_match = _ligation_rule_match(
        snapback_seed_sequence,
        adapter_part.sequence,
        spec.steps.snapback_adapter_engagement.ligation_rule,
    )
    if engagement_match is None:
        engagement_issues.append(
            _issue(
                "SNAPBACK_ADAPTER_COMPATIBILITY_FAIL",
                "snapback adapter engagement does not satisfy the configured ligation rule",
                step_id="snapback_adapter_engagement",
            )
        )
    snapback_adapter_sequence = retained_product + adapter_part.sequence
    engagement_invariants = _evaluate_split_hard_invariants(
        spec,
        catalogs=catalogs,
        source_sequence=sequence,
        regions=regions,
        sites=sites,
        compound_regions=compound_regions,
        assembled_payload=assembled_payload,
        fragment_lengths=fragment_lengths,
        retained_product=retained_product,
        state_sequences={**state_sequences_by_id, "snapback_adapter_complex": snapback_adapter_sequence},
        state_segments_by_id={**state_segments_by_id, "snapback_adapter_complex": retained_segments},
        ligation_matches_by_step={"snapback_adapter_engagement": engagement_match},
        adapter_parts_by_step={"snapback_adapter_engagement": adapter_part},
        state_id="snapback_adapter_complex",
        step_id="snapback_adapter_engagement",
        issues=engagement_issues,
    )
    states.append(
        _state(
            state_id="snapback_adapter_complex",
            step_id="snapback_adapter_engagement",
            kind="snapback_adapter_complex",
            state_kind="snapback_adapter_complex",
            topology_kind="branched_y",
            status="unsatisfied" if _has_error_issue(engagement_issues) else "satisfied",
            primary_sequence=snapback_adapter_sequence,
            metadata={
                "adapter_id": adapter_part.id,
                "paired_nt": engagement_match.paired_nt if engagement_match else 0,
                "hard_invariants": engagement_invariants,
            },
            view_contract_version=spec.output.publish_contract_version,
            pattern_label="pattern",
        )
    )
    state_sequences_by_id["snapback_adapter_complex"] = snapback_adapter_sequence
    state_segments_by_id["snapback_adapter_complex"] = retained_segments
    issues.extend(engagement_issues)

    ligation_issues: list[YiuValidationIssue] = []
    ligation_rule = spec.steps.hairpin_ligation.ligation_rule or spec.steps.snapback_adapter_engagement.ligation_rule
    ligation_match = _ligation_rule_match(snapback_seed_sequence, adapter_part.sequence, ligation_rule)
    if ligation_match is None:
        ligation_issues.append(
            _issue(
                "HAIRPIN_LIGATION_COMPATIBILITY_FAIL",
                "hairpin ligation does not satisfy the configured ligation rule",
                step_id="hairpin_ligation",
            )
        )
    hairpin_sequence = retained_product + adapter_part.sequence
    ligation_invariants = _evaluate_split_hard_invariants(
        spec,
        catalogs=catalogs,
        source_sequence=sequence,
        regions=regions,
        sites=sites,
        compound_regions=compound_regions,
        assembled_payload=assembled_payload,
        fragment_lengths=fragment_lengths,
        retained_product=retained_product,
        state_sequences={**state_sequences_by_id, "ligated_ssdna_hairpin": hairpin_sequence},
        state_segments_by_id={**state_segments_by_id, "ligated_ssdna_hairpin": retained_segments},
        ligation_matches_by_step={"hairpin_ligation": ligation_match},
        adapter_parts_by_step={"hairpin_ligation": adapter_part},
        state_id="ligated_ssdna_hairpin",
        step_id="hairpin_ligation",
        issues=ligation_issues,
    )
    states.append(
        _state(
            state_id="ligated_ssdna_hairpin",
            step_id="hairpin_ligation",
            kind="ligated_ssdna_hairpin",
            state_kind="ligated_ssdna_hairpin",
            topology_kind="hairpin_ssdna",
            status="unsatisfied" if _has_error_issue(ligation_issues) else "satisfied",
            primary_sequence=hairpin_sequence,
            metadata={
                "adapter_id": adapter_part.id,
                "paired_nt": ligation_match.paired_nt if ligation_match else 0,
                "hard_invariants": ligation_invariants,
            },
            view_contract_version=spec.output.publish_contract_version,
            junctions=[
                {
                    "id": "hairpin_ligation_junction",
                    "paired_nt": ligation_match.paired_nt if ligation_match else 0,
                    "compatibility_mode": ligation_rule.mode,
                }
            ],
            pattern_label="pattern",
        )
    )
    state_sequences_by_id["ligated_ssdna_hairpin"] = hairpin_sequence
    state_segments_by_id["ligated_ssdna_hairpin"] = retained_segments
    issues.extend(ligation_issues)

    insert_primary = retained_product + reverse_complement_iupac(retained_product)
    insert_complement = reverse_complement_iupac(insert_primary)
    insert_segments = [
        *retained_segments,
        _StateSegment("rc_retained_product", 0, len(retained_product), len(retained_product), len(insert_primary)),
    ]
    assembled_payload_pieces = [
        segment for segment in retained_segments if segment.segment_id in {left_region.id, right_region.id}
    ]
    if not assembled_payload_pieces:
        assembled_payload_pieces = [
            segment
            for segment in retained_segments
            if segment.segment_id in {"retained_payload_left", "retained_payload_right"}
        ]
    hairpin_pcr_issues: list[YiuValidationIssue] = []
    hairpin_pcr_invariants = _evaluate_split_hard_invariants(
        spec,
        catalogs=catalogs,
        source_sequence=sequence,
        regions=regions,
        sites=sites,
        compound_regions=compound_regions,
        assembled_payload=assembled_payload,
        fragment_lengths=fragment_lengths,
        retained_product=retained_product,
        state_sequences={**state_sequences_by_id, "hairpin_pcr_linear_insert": insert_primary},
        state_segments_by_id={**state_segments_by_id, "hairpin_pcr_linear_insert": retained_segments},
        ligation_matches_by_step={},
        adapter_parts_by_step={},
        state_id="hairpin_pcr_linear_insert",
        step_id="hairpin_pcr",
        issues=hairpin_pcr_issues,
    )
    states.append(
        _state(
            state_id="hairpin_pcr_linear_insert",
            step_id="hairpin_pcr",
            kind="hairpin_pcr_linear_insert",
            state_kind="hairpin_pcr_linear_insert",
            topology_kind="linear_dsdna",
            status="unsatisfied" if _has_error_issue(hairpin_pcr_issues) else "satisfied",
            primary_sequence=insert_primary,
            complement_sequence=insert_complement,
            metadata={"hard_invariants": hairpin_pcr_invariants},
            view_contract_version=spec.output.publish_contract_version,
            segments=_segment_rows(insert_segments),
            annotations=[
                _compound_annotation(
                    annotation_id="assembled_payload",
                    pieces=assembled_payload_pieces,
                    assembled_coordinate_space=spec.payload_goal.assembly_space,
                )
            ],
            pattern_label="pattern",
        )
    )
    state_sequences_by_id["hairpin_pcr_linear_insert"] = insert_primary
    state_segments_by_id["hairpin_pcr_linear_insert"] = retained_segments
    issues.extend(hairpin_pcr_issues)

    report_sequence_mode = _v2_report_sequence_mode(states)
    report_validation_mode = "pattern_compatibility" if report_sequence_mode == "pattern" else "concrete_realization"
    states = [state.model_copy(update={"validation_mode": report_validation_mode}) for state in states]
    return YiuValidationReport(
        protocol=spec.protocol_template,
        protocol_template=spec.protocol_template,
        template_alias_used=spec.template_alias_used,
        template_alias_status=spec.template_alias_status,
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
