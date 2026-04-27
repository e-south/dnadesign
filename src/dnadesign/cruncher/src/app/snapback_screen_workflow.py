"""
Application workflow for the released-product Snapback screen objective.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from dnadesign.cruncher.app.snapback_released_target_search_workflow import (
    run_released_snapback_target_search,
)
from dnadesign.cruncher.snapback.models import CatalogSources, CoordinateSpan
from dnadesign.cruncher.snapback.released_models import (
    ReleaseCatalogSources,
    ReleasedFinalTargetGeometry,
    ReleasedTargetSearchConfig,
    ReleasedTargetSearchHit,
    SingleNickReleasedTargetSearchRequest,
)
from dnadesign.cruncher.snapback.released_route_policy import (
    ReleasedActiveStrand,
    ReleasedRouteFamily,
    normalize_active_strand_list,
)
from dnadesign.cruncher.snapback.screen_models import (
    CoordinateFrameTransform,
    SnapbackMechanismClass,
    SnapbackMechanismLedgerEntry,
    SnapbackScreenReport,
    SnapbackScreenTargetTopology,
)


def _route_families_for_retained_strands(
    retained_product_strands: list[ReleasedActiveStrand],
) -> list[ReleasedRouteFamily]:
    route_families: list[ReleasedRouteFamily] = []
    if "bottom" in retained_product_strands:
        route_families.append("bottom_active_from_top_nick")
    if "top" in retained_product_strands:
        route_families.append("top_active_from_bottom_nick")
    return route_families


def parse_retained_product_strands(value: str) -> list[ReleasedActiveStrand]:
    raw_items = [item.strip() for item in value.split(",") if item.strip()]
    return normalize_active_strand_list(raw_items, label="allow_retained_strands")


def build_snapback_screen_request(
    *,
    target_origin: int = 0,
    stem_bp: int = 3,
    cap_nt: int = 3,
    nick_preset: str | None = "neb_nicking_v1",
    nick_additional_presets: list[str] | None = None,
    nick_additional_paths: list[Path] | None = None,
    release_preset: str | None = "type_iis_release_v1",
    release_additional_presets: list[str] | None = None,
    release_additional_paths: list[Path] | None = None,
    release_variant_ids: list[str] | None = None,
    retained_product_strands: list[ReleasedActiveStrand] | None = None,
    use_vendor_footprints: bool = True,
    max_results: int = 16,
    near_boundary_search_limit: int = 8,
    allow_demo_hits: bool = False,
    allow_frequent_cutter_nickases: bool = False,
) -> SingleNickReleasedTargetSearchRequest:
    retained_strands = retained_product_strands or ["top", "bottom"]
    retained_strands = normalize_active_strand_list(retained_strands, label="retained_product_strands")
    return SingleNickReleasedTargetSearchRequest(
        target=ReleasedFinalTargetGeometry(
            nick_boundary_from_left=target_origin,
            paired_bp=stem_bp,
            cap_nt=cap_nt,
        ),
        nick_sources=CatalogSources(
            preset=nick_preset,
            additional_presets=nick_additional_presets
            if nick_additional_presets is not None
            else ["thermo_nicking_v1"],
            additional_paths=nick_additional_paths or [],
        ),
        release_sources=ReleaseCatalogSources(
            preset=release_preset,
            additional_presets=release_additional_presets or [],
            additional_paths=release_additional_paths or [],
        ),
        search=ReleasedTargetSearchConfig(
            max_results=max_results,
            near_boundary_search_limit=near_boundary_search_limit,
            allow_demo_hits=allow_demo_hits,
            allowed_release_variant_ids=release_variant_ids if release_variant_ids is not None else ["BspQI"],
            allow_precut_footprint_outside_active_product=use_vendor_footprints,
            allowed_active_strands=retained_strands,
            allowed_route_families=_route_families_for_retained_strands(retained_strands),
            disallowed_nickase_warning_codes=[] if allow_frequent_cutter_nickases else ["FREQUENT_CUTTER"],
        ),
    )


def _mechanism_class(hit: ReleasedTargetSearchHit, provenance_counts: dict[str, int]) -> SnapbackMechanismClass:
    has_degenerate_bases = provenance_counts.get("degenerate_motif_base", 0) > 0
    has_user_bases = provenance_counts.get("user_sequence", 0) > 0
    has_release_trim = hit.sacrificial_downstream_tail_nt > 0 or hit.active_product_length_nt < hit.precursor_length_nt
    if has_degenerate_bases:
        return "mixed_footprint_payload" if has_user_bases else "degenerate_footprint_snapback"
    if has_release_trim:
        return "fixed_footprint_plus_release_trim"
    return "mixed_footprint_payload"


def _provenance_counts(hit: ReleasedTargetSearchHit) -> dict[str, int]:
    counts = Counter(base.source_constraint for base in hit.projection.active_product_provenance)
    return dict(sorted(counts.items()))


def _frame_transforms(hit: ReleasedTargetSearchHit) -> list[CoordinateFrameTransform]:
    active_start, active_end = hit.projection.active_product_span
    active_frame = f"precursor_{hit.active_strand}_frame"
    return [
        CoordinateFrameTransform(
            source_frame="vendor_site_frame",
            target_frame="precursor_top_frame",
            source_span=CoordinateSpan(start=0, end=len(hit.intended_nick_site_sequence)),
            target_span=CoordinateSpan(start=hit.pre_nick_site.start, end=hit.pre_nick_site.end),
            orientation=hit.intended_nick_site_orientation,
            label=f"{hit.nickase_variant_id} footprint",
        ),
        CoordinateFrameTransform(
            source_frame=active_frame,
            target_frame="retained_product_frame",
            source_span=CoordinateSpan(start=active_start, end=active_end),
            target_span=CoordinateSpan(start=0, end=hit.active_product_length_nt),
            orientation="forward",
            label="active product projection",
        ),
        CoordinateFrameTransform(
            source_frame="retained_product_frame",
            target_frame="logical_snapback_frame",
            source_span=CoordinateSpan(start=0, end=hit.active_product_length_nt),
            target_span=CoordinateSpan(start=0, end=hit.active_product_length_nt),
            orientation="forward",
            label="logical origin projection",
        ),
    ]


def build_snapback_mechanism_ledger_entry(hit: ReleasedTargetSearchHit) -> SnapbackMechanismLedgerEntry:
    logical_origin = hit.nick_boundary_from_left
    stem_bp = hit.final_candidate.paired_bp
    cap_nt = hit.final_candidate.cap_nt
    stem_span = CoordinateSpan(start=logical_origin, end=logical_origin + stem_bp)
    cap_span = CoordinateSpan(start=stem_span.end, end=stem_span.end + cap_nt)
    foldback_span = CoordinateSpan(start=cap_span.end, end=cap_span.end + stem_bp)
    provenance_counts = _provenance_counts(hit)
    return SnapbackMechanismLedgerEntry(
        rank=hit.rank,
        hit_kind=hit.hit_kind,
        nickase_variant_id=hit.nickase_variant_id,
        release_variant_id=hit.release_variant_id,
        route_family=hit.route_family,
        physical_nicked_strand=hit.physical_nicked_strand,
        retained_product_strand=hit.active_strand,
        oriented_nick_footprint=hit.intended_nick_site_sequence,
        oriented_nick_footprint_orientation=hit.intended_nick_site_orientation,
        oriented_release_footprint=hit.release_site_sequence,
        oriented_release_footprint_orientation=hit.release_site_orientation,
        logical_origin=logical_origin,
        logical_stem_bp=stem_bp,
        cap_nt=cap_nt,
        logical_stem_span=stem_span,
        logical_cap_span=cap_span,
        logical_foldback_return_span=foldback_span,
        upstream_retained_duplex_bp=hit.upstream_retained_duplex_bp,
        effective_foldback_pairing_bp=hit.effective_stem_bp,
        release_terminal_boundary=hit.active_product_length_nt,
        mechanism_class=_mechanism_class(hit, provenance_counts),
        provenance_counts=provenance_counts,
        foldback_mismatch_count=hit.final_candidate.mismatch_count,
        frame_transforms=_frame_transforms(hit),
    )


def build_snapback_screen_report(
    *,
    search_report,
    request: SingleNickReleasedTargetSearchRequest | None = None,
) -> SnapbackScreenReport:
    target = search_report.metadata.target
    selected_hits = search_report.exact_hits if search_report.exact_hits else search_report.near_hits
    retained_product_strands = (
        list(request.search.allowed_active_strands)
        if request is not None
        else list(search_report.metadata.allowed_active_strands)
    )
    allow_oriented_vendor_footprints = (
        request.search.allow_precut_footprint_outside_active_product if request is not None else True
    )
    target_topology = SnapbackScreenTargetTopology(
        logical_origin=target.nick_boundary_from_left,
        stem_bp=target.paired_bp,
        cap_nt=target.cap_nt,
        retained_product_strands=retained_product_strands,
        allow_oriented_vendor_footprints=allow_oriented_vendor_footprints,
        allow_degenerate_motif_assignment=True,
        allow_release_trim_after_foldback_return=True,
    )
    return SnapbackScreenReport(
        status=search_report.status,
        workspace_root=search_report.workspace_root,
        target_topology=target_topology,
        exact_hit_count=search_report.metadata.post_truncation_exact_hit_count,
        near_hit_count=search_report.metadata.post_truncation_near_hit_count,
        mechanism_ledger=[build_snapback_mechanism_ledger_entry(hit) for hit in selected_hits],
        search_report=search_report,
    )


def run_snapback_screen(
    *,
    request: SingleNickReleasedTargetSearchRequest,
    workspace_root: Path,
) -> SnapbackScreenReport:
    search_report = run_released_snapback_target_search(request=request, workspace_root=workspace_root)
    return build_snapback_screen_report(search_report=search_report, request=request)


__all__ = [
    "build_snapback_mechanism_ledger_entry",
    "build_snapback_screen_report",
    "build_snapback_screen_request",
    "parse_retained_product_strands",
    "run_snapback_screen",
]
