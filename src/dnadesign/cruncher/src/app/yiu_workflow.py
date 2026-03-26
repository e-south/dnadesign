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
from pathlib import Path
from typing import Any

from dnadesign.cruncher.bio import (
    derive_cut_geometry,
    longest_reverse_complement_overlap,
    motif_matches,
    reverse_complement_iupac,
    sequence_contains_iupac,
)
from dnadesign.cruncher.yiu.artifacts import (
    annotations_path,
    build_run_dir,
    design_id,
    fragments_path,
    parts_path,
    prepare_run_dir,
    published_views_dir,
    report_path,
    state_view_path,
    status_path,
    trace_path,
    write_csv,
    write_manifest,
    write_report,
    write_status,
    write_trace,
)
from dnadesign.cruncher.yiu.catalog import LoadedYiuCatalogs, load_yiu_catalogs
from dnadesign.cruncher.yiu.load import load_yiu_spec
from dnadesign.cruncher.yiu.models import (
    EnzymeSiteSpec,
    RegionSpec,
    YiuProcessSpec,
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


def _sequence_for_region(sequence: str, region: RegionSpec) -> str:
    return sequence[region.start : region.end]


def _overlap(left: RegionSpec, right: RegionSpec) -> bool:
    return left.start < right.end and right.start < left.end


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


def _state(
    *,
    state_id: str,
    step_id: str,
    kind: str,
    status: str,
    primary_sequence: str | None,
    complement_sequence: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> YiuStateRecord:
    return YiuStateRecord(
        state_id=state_id,
        step_id=step_id,
        kind=kind,
        status=status,  # type: ignore[arg-type]
        primary_sequence=primary_sequence,
        complement_sequence=complement_sequence,
        metadata=metadata or {},
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


def _build_yiu_report(spec: YiuProcessSpec, *, catalogs: LoadedYiuCatalogs | None = None) -> YiuValidationReport:
    catalogs = catalogs or LoadedYiuCatalogs(restriction_enzymes={}, nickases={}, adapters={}, paths=())
    issues: list[YiuValidationIssue] = []
    states: list[YiuStateRecord] = []
    source_sequence = spec.source_oligo.sequence
    regions = _region_lookup(spec)
    primers = _primer_lookup(spec)
    restriction_sites = _restriction_lookup(spec)
    nickase_sites = _nickase_lookup(spec)

    overlap_collections = (
        spec.source_oligo.primer_sites
        + spec.source_oligo.payload_windows
        + spec.source_oligo.restriction_sites
        + spec.source_oligo.nickase_sites
    )
    for index, left in enumerate(overlap_collections):
        left_start = left.start
        left_end = left.end if hasattr(left, "end") else left.start + len(left.recognition_sequence)
        left_region = RegionSpec(id=left.id, start=left_start, end=left_end)
        for right in overlap_collections[index + 1 :]:
            right_start = right.start
            right_end = right.end if hasattr(right, "end") else right.start + len(right.recognition_sequence)
            right_region = RegionSpec(id=right.id, start=right_start, end=right_end)
            if _overlap(left_region, right_region):
                issues.append(
                    _issue(
                        "ANNOTATION_OVERLAP",
                        f"annotations {left.id} and {right.id} overlap on the source oligo",
                        state_id="source_oligo_ssdna",
                    )
                )

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
    pcr_primary = source_sequence
    pcr_complement: str | None = None
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
            if forward is None or reverse is None:
                step_issues.append(
                    _issue("PCR_PRIMER_SITE_MISSING", "PCR primer site reference is missing", step_id=step.id)
                )
            elif forward.start >= reverse.start:
                step_issues.append(
                    _issue("PCR_BOUNDARY_INVALID", "forward primer must start before reverse primer", step_id=step.id)
                )
            current_complement = reverse_complement_iupac(current_primary)
            amplicon_start = forward.start if forward is not None else None
            amplicon_end = reverse.end if reverse is not None else None
            if amplicon_start is not None and amplicon_end is not None:
                pcr_primary = source_sequence[amplicon_start:amplicon_end]
                pcr_complement = reverse_complement_iupac(pcr_primary)
                state_primary = pcr_primary
                state_complement = pcr_complement
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
                    len(pcr_primary) if amplicon_start is not None and amplicon_end is not None else 0
                ),
            }
        elif step.kind == "restriction_digest":
            state_primary = pcr_primary
            state_complement = pcr_complement
            left_site = restriction_sites.get(str(step.left_site))
            right_site = restriction_sites.get(str(step.right_site))
            if left_site is None or right_site is None:
                step_issues.append(
                    _issue("DIGEST_SITE_MISSING", "restriction digest site reference is missing", step_id=step.id)
                )
            else:
                for site_id, site in (("left", left_site), ("right", right_site)):
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
                            current_primary,
                            start=site.start,
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
                metadata = {
                    "left_overhang": digest_left_overhang,
                    "right_overhang": digest_right_overhang,
                }
        elif step.kind == "circularization":
            overlap = longest_reverse_complement_overlap(digest_left_overhang, digest_right_overhang)
            if step.compatibility == "exact_complement":
                if not digest_left_overhang or digest_left_overhang != reverse_complement_iupac(digest_right_overhang):
                    step_issues.append(
                        _issue(
                            "CIRCULARIZATION_COMPATIBILITY_FAIL",
                            "left and right sticky ends are not exact reverse complements",
                            step_id=step.id,
                        )
                    )
            else:
                if overlap < 1:
                    step_issues.append(
                        _issue(
                            "CIRCULARIZATION_COMPATIBILITY_FAIL",
                            f"sticky-end overlap {overlap} is below the required threshold",
                            step_id=step.id,
                        )
                    )
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
                current_primary = assembled_payload
                current_complement = reverse_complement_iupac(assembled_payload)
            metadata = {"assembled_payload": assembled_payload, "sticky_end_overlap": overlap}
        elif step.kind == "exonuclease_selection":
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
            retained_by_id = {region.id: region for region in retained_regions}
            for site_id in step.site_ids:
                site = nickase_sites.get(site_id)
                if site is None:
                    step_issues.append(
                        _issue("NICKASE_SITE_MISSING", f"nickase site {site_id} is missing", step_id=step.id)
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
                        source_sequence,
                        start=site.start,
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
                site_region = RegionSpec(id=site.id, start=site.start, end=site.end)
                for retained_region in retained_regions:
                    if _overlap(site_region, retained_region):
                        step_issues.append(
                            _issue(
                                "NICKASE_RETAINED_REGION_CONFLICT",
                                f"nickase site {site.id} overlaps retained region {retained_region.id}",
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
                cuts = [
                    region.start,
                    *sorted(boundary for boundary in boundaries if region.start <= boundary <= region.end),
                    region.end,
                ]
                fragment_lengths.extend(
                    cuts[idx + 1] - cuts[idx] for idx in range(0, len(cuts) - 1) if cuts[idx + 1] > cuts[idx]
                )
            retained_product = "".join(
                _sequence_for_region(source_sequence, retained_by_id[region_id])
                for region_id in step.retained_region_ids
                if region_id in retained_by_id
            )
            current_primary = retained_product
            current_complement = reverse_complement_iupac(retained_product)
            metadata = {"fragment_lengths": fragment_lengths, "retained_product": retained_product}
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
            if left_region is not None and right_region is not None:
                left = _sequence_for_region(source_sequence, left_region)
                right = _sequence_for_region(source_sequence, right_region)
                overlap = longest_reverse_complement_overlap(left, right)
                if overlap < required:
                    step_issues.append(
                        _issue(
                            "FOLDBACK_HOMOLOGY_INSUFFICIENT",
                            f"foldback homology overlap {overlap} is below required {required}",
                            step_id=step.id,
                        )
                    )
            metadata = {"left_homology": left, "right_homology": right, "complementary_bases": overlap}
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
            metadata = {
                "adapter_sequence": adapter_sequence,
                "y_adapter_id": spec.adapter_policy.y_adapter_id,
            }
        elif step.kind == "amplification":
            current_primary = f"{retained_product}{adapter_sequence}"
            current_complement = reverse_complement_iupac(current_primary)
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

    return YiuValidationReport(
        spec_name=spec.name,
        status="unsatisfied" if issues else "satisfied",
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


def validate_yiu_spec(path: str | Path) -> YiuValidationReport:
    spec, _spec_path, workspace_root = load_yiu_spec(path)
    catalogs = load_yiu_catalogs(spec, workspace_root=workspace_root)
    return _build_yiu_report(spec, catalogs=catalogs)


def _catalog_bytes(catalog_paths: list[Path]) -> bytes:
    if not catalog_paths:
        return b""
    return b"\n".join(path.read_bytes() for path in catalog_paths if path.exists())


def _annotation_rows(spec: YiuProcessSpec) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for category, collection in (
        ("primer_site", spec.source_oligo.primer_sites),
        ("restriction_site", spec.source_oligo.restriction_sites),
        ("nickase_site", spec.source_oligo.nickase_sites),
        ("payload_window", spec.source_oligo.payload_windows),
        ("homology_window", spec.source_oligo.homology_windows),
        ("retained_region", spec.source_oligo.retained_regions),
        ("sacrificial_region", spec.source_oligo.sacrificial_regions),
    ):
        for item in collection:
            rows.append(
                {
                    "category": category,
                    "id": item.id,
                    "start": item.start,
                    "end": item.end if hasattr(item, "end") else item.start + len(item.recognition_sequence),
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
    for state in report.states:
        payload = {
            "version": 1,
            "workflow": "yiu",
            "state_id": state.state_id,
            "kind": state.kind,
            "status": state.status,
            "primary_sequence": state.primary_sequence,
            "complement_sequence": state.complement_sequence,
            "meta": state.metadata,
        }
        state_view_path(run_dir, state.state_id).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _materialize_yiu_bundle(spec_path: str | Path, *, force_overwrite: bool) -> tuple[Path, YiuValidationReport]:
    spec, resolved_spec_path, workspace_root = load_yiu_spec(spec_path)
    catalogs = load_yiu_catalogs(spec, workspace_root=workspace_root)
    report = _build_yiu_report(spec, catalogs=catalogs)
    catalog_paths = list(catalogs.paths)
    run_id = design_id(spec_bytes=resolved_spec_path.read_bytes(), catalog_bytes=_catalog_bytes(catalog_paths))
    run_dir = build_run_dir(
        workspace_root=workspace_root,
        run_root=spec.output.run_dir,
        spec_name=spec.name,
        run_id=run_id,
    )
    prepare_run_dir(run_dir, force_overwrite=force_overwrite)
    report = report.model_copy(update={"run_dir": str(run_dir.resolve())})
    write_report(run_dir, report)
    write_status(run_dir, report)
    write_manifest(
        run_dir, workspace_root=workspace_root, spec_path=resolved_spec_path, report=report, catalog_paths=catalog_paths
    )
    write_trace(run_dir, report.states)
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
        "manifest_path": str((resolved / "yiu_manifest.json").resolve()),
        "status_path": str(status_path(resolved).resolve()),
        "report_path": str(report_path(resolved).resolve()),
        "trace_path": str(trace_path(resolved).resolve()),
        "published_views_dir": str(published_views_dir(resolved).resolve()),
    }
