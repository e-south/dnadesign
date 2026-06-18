"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cruncher/src/app/snapback_released_solve_reporting.py

Report-shaping helpers for released-product Snapback solve runs.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from dnadesign.cruncher.app.snapback_released_catalogs import ReleasedResolvedCatalogs
from dnadesign.cruncher.snapback.released_models import (
    ReleasedSolveHit,
    ReleasedSolveOutputConfig,
    ReleasedSolveReport,
    ReleasedSolveReportMetadata,
    ReleasedTargetSearchHit,
    ReleasedTargetSearchReport,
    SingleNickReleasedTargetSearchRequest,
)

ReleasedSelectedHitKind = Literal["exact", "nearest"] | None


@dataclass(frozen=True)
class ReleasedSolveSelection:
    hits: list[ReleasedTargetSearchHit]
    selected_hit_kind: ReleasedSelectedHitKind


def select_released_solve_hits(search_report: ReleasedTargetSearchReport) -> ReleasedSolveSelection:
    if search_report.exact_hits:
        return ReleasedSolveSelection(hits=list(search_report.exact_hits), selected_hit_kind="exact")
    if search_report.near_hits:
        return ReleasedSolveSelection(hits=list(search_report.near_hits), selected_hit_kind="nearest")
    return ReleasedSolveSelection(hits=[], selected_hit_kind=None)


def build_released_solve_report(
    *,
    search_report: ReleasedTargetSearchReport,
    request: SingleNickReleasedTargetSearchRequest,
    output: ReleasedSolveOutputConfig,
    resolved_catalogs: ReleasedResolvedCatalogs,
    workspace_root: Path,
    run_dir: Path,
    materialized_hits: list[ReleasedSolveHit],
    selected_hit_kind: ReleasedSelectedHitKind,
) -> ReleasedSolveReport:
    status = (
        "exact_hits_materialized"
        if selected_hit_kind == "exact"
        else "near_hits_materialized"
        if selected_hit_kind == "nearest"
        else "no_hits"
    )
    return ReleasedSolveReport(
        status=status,
        workspace_root=str(workspace_root.resolve()),
        run_dir=str(run_dir.resolve()),
        metadata=ReleasedSolveReportMetadata(
            route_policy_final_geometry_source=search_report.metadata.route_policy_final_geometry_source,
            target=request.target,
            nick_catalog_source=resolved_catalogs.nick_catalog_source,
            release_catalog_source=resolved_catalogs.release_catalog_source,
            disallowed_nickase_warning_codes=list(request.search.disallowed_nickase_warning_codes),
            allowed_release_variant_ids=list(request.search.allowed_release_variant_ids),
            allowed_active_strands=list(request.search.allowed_active_strands),
            allowed_route_families=list(request.search.allowed_route_families),
            evaluated_pair_count=search_report.metadata.evaluated_pair_count,
            available_exact_hit_count=search_report.metadata.pre_truncation_exact_hit_count,
            available_near_hit_count=search_report.metadata.pre_truncation_near_hit_count,
            selected_hit_kind=selected_hit_kind,
            materialized_hit_count=len(materialized_hits),
            requested_materialize_top_k=output.materialize_top_k,
            render_format=output.render_format,
            emit_renders=output.emit_renders,
            blocker_counts=dict(search_report.metadata.blocker_counts),
        ),
        issues=list(search_report.issues),
        search_report=search_report,
        hits=materialized_hits,
    )


__all__ = [
    "ReleasedSolveSelection",
    "ReleasedSelectedHitKind",
    "build_released_solve_report",
    "select_released_solve_hits",
]
