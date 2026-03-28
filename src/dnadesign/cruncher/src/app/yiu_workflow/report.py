"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/app/yiu_workflow/report.py

Top-level YIU report dispatch and validate entrypoints.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dnadesign.cruncher.app.yiu_workflow.helpers import (
    _compound_annotation,
    _evaluate_ligation_compatibility,
    _has_error_issue,
    _issue,
    _iupac_match_status,
    _pattern_policy_issue,
    _pattern_summary,
    _segment_rows,
    _segments_for_source_regions,
    _sequence_for_region,
    _state,
    _StateSegment,
    _v2_overlap_issues,
    _v2_part,
    _v2_primer_core_lookup,
    _v2_region_lookup,
    _v2_report_sequence_mode,
)
from dnadesign.cruncher.app.yiu_workflow.split_template import _build_yiu_report_v2_split_template
from dnadesign.cruncher.app.yiu_workflow.v1_report import _build_yiu_report_v1
from dnadesign.cruncher.bio import derive_cut_geometry, reverse_complement_iupac
from dnadesign.cruncher.yiu.catalog import LoadedYiuCatalogs, load_yiu_catalogs
from dnadesign.cruncher.yiu.load import load_yiu_spec
from dnadesign.cruncher.yiu.models import (
    RegionSpec,
    YiuProcessSpec,
    YiuProcessSpecV2,
    YiuReportMetadata,
    YiuStateRecord,
    YiuValidationIssue,
    YiuValidationReport,
)


def _build_yiu_report_v2(spec: YiuProcessSpecV2, *, catalogs: LoadedYiuCatalogs | None = None) -> YiuValidationReport:
    if spec.protocol_template == "yiu_circularized_payload_v1":
        return _build_yiu_report_v2_split_template(spec, catalogs=catalogs)
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
