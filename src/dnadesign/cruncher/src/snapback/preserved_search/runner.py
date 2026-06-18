"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cruncher/src/snapback/preserved_search/runner.py

Runner for preserved-site target search.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from dnadesign.cruncher.nickases.models import NickaseCatalog
from dnadesign.cruncher.snapback.catalog_sources import catalog_source_label
from dnadesign.cruncher.snapback.preserved_search.candidate_builder import best_hit_for_boundary
from dnadesign.cruncher.snapback.preserved_search.placements import (
    build_feasibility_row,
    iter_target_strand_placements,
)
from dnadesign.cruncher.snapback.preserved_search.ranking import rank_hits
from dnadesign.cruncher.snapback.target_models import (
    SnapbackTargetGeometry,
    SnapbackTargetSearchMetadata,
    SnapbackTargetSearchReport,
)


def search_snapback_target_hits(
    *,
    catalog: NickaseCatalog,
    target: SnapbackTargetGeometry,
    workspace_root: Path,
    catalog_preset: str | None,
    catalog_presets: list[str],
    catalog_additional_paths: list[Path],
    normalize_to_top_strand_nick: bool = True,
    max_results: int = 8,
) -> SnapbackTargetSearchReport:
    placements = iter_target_strand_placements(
        catalog_entries=catalog.entries,
        target=target,
        normalize_to_top_strand_nick=normalize_to_top_strand_nick,
    )
    feasibility = [build_feasibility_row(placement) for placement in placements]

    exact_hits = []
    near_hits = []
    for placement in placements:
        if placement.exact_boundary_hit_possible and placement.exact_input_length_nt is not None:
            hit = best_hit_for_boundary(
                placement=placement,
                boundary=target.nick_boundary_from_left,
                input_length_nt=placement.exact_input_length_nt,
                target=target,
                hit_kind="exact",
            )
            if hit is not None:
                exact_hits.append(hit)
        if placement.any_boundary_hit_possible and placement.earliest_feasible_boundary is not None:
            if (
                placement.exact_boundary_hit_possible
                and placement.earliest_feasible_boundary == target.nick_boundary_from_left
            ):
                continue
            hit = best_hit_for_boundary(
                placement=placement,
                boundary=placement.earliest_feasible_boundary,
                input_length_nt=int(placement.earliest_input_length_nt),
                target=target,
                hit_kind="nearest",
            )
            if hit is not None:
                near_hits.append(hit)

    exact_hits = rank_hits(exact_hits, target=target, exact=True)[:max_results]
    near_hits = rank_hits(near_hits, target=target, exact=False)[:max_results]
    if exact_hits:
        status = "exact_hits_found"
    elif near_hits:
        status = "near_hits_only"
    else:
        status = "no_hits"
    return SnapbackTargetSearchReport(
        status=status,
        workspace_root=str(workspace_root),
        metadata=SnapbackTargetSearchMetadata(
            catalog_preset=catalog_preset,
            catalog_presets=catalog_presets,
            catalog_additional_paths=[str(path) for path in catalog_additional_paths],
            catalog_source=catalog_source_label(
                preset_ids=catalog_presets,
                resolved_paths=catalog_additional_paths,
            ),
            target=target,
            evaluated_orientation_count=len(feasibility),
            exact_hit_count=len(exact_hits),
            near_hit_count=len(near_hits),
        ),
        issues=[],
        exact_hits=exact_hits,
        near_hits=near_hits,
        feasibility=feasibility,
    )
