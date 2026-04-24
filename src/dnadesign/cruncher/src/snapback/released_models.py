"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/snapback/released_models.py

Compatibility facade for released-product snapback contracts.

Module Author(s): Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dnadesign.cruncher.snapback.released_projection_models import (
    ReleaseCatalogNormalizationInfo,
    ReleasedFinalCandidate,
    ReleasedProductBaseProvenance,
    ReleasedProductProjection,
    build_release_catalog_info,
    build_released_nickase_catalog_info,
)
from dnadesign.cruncher.snapback.released_route_policy import (
    ReleasedActiveStrand,
    ReleasedFinalGeometrySource,
    ReleasedRouteFamily,
    ReleasedRouteSemantics,
    ReleasedSearchRoutePolicy,
    infer_released_search_final_geometry_source,
    released_route_semantics,
    released_search_route_policy,
    route_family_active_strand,
    route_family_final_geometry_source,
    route_family_physical_nicked_strand,
    route_family_retained_partner_strand,
)
from dnadesign.cruncher.snapback.released_search_models import (
    ReleasedTargetSearchConfig,
    ReleasedTargetSearchHit,
    ReleasedTargetSearchMetadata,
    ReleasedTargetSearchReport,
    SingleNickReleasedTargetSearchRequest,
)
from dnadesign.cruncher.snapback.released_solve_models import (
    ReleasedSnapbackEvaluationReport,
    ReleasedSnapbackReportMetadata,
    ReleasedSolveHit,
    ReleasedSolveOutputConfig,
    ReleasedSolveReport,
    ReleasedSolveReportMetadata,
)
from dnadesign.cruncher.snapback.released_spec_models import (
    ReleaseCatalogSources,
    ReleasedFinalTargetGeometry,
    ReleasedNickStageSpec,
    ReleasedReleaseStageSpec,
    ReleasedSnapbackConstraintsSpec,
    ReleasedSnapbackHeader,
    ReleasedSnapbackInputSpec,
    ReleasedSnapbackOutputConfig,
    SingleNickReleasedSnapbackSpec,
)

__all__ = [
    "ReleaseCatalogNormalizationInfo",
    "ReleaseCatalogSources",
    "ReleasedActiveStrand",
    "ReleasedFinalCandidate",
    "ReleasedFinalGeometrySource",
    "ReleasedFinalTargetGeometry",
    "ReleasedNickStageSpec",
    "ReleasedProductBaseProvenance",
    "ReleasedProductProjection",
    "ReleasedReleaseStageSpec",
    "ReleasedRouteFamily",
    "ReleasedRouteSemantics",
    "ReleasedSearchRoutePolicy",
    "ReleasedSnapbackConstraintsSpec",
    "ReleasedSnapbackEvaluationReport",
    "ReleasedSnapbackHeader",
    "ReleasedSnapbackInputSpec",
    "ReleasedSnapbackOutputConfig",
    "ReleasedSnapbackReportMetadata",
    "ReleasedSolveHit",
    "ReleasedSolveOutputConfig",
    "ReleasedSolveReport",
    "ReleasedSolveReportMetadata",
    "ReleasedTargetSearchConfig",
    "ReleasedTargetSearchHit",
    "ReleasedTargetSearchMetadata",
    "ReleasedTargetSearchReport",
    "SingleNickReleasedSnapbackSpec",
    "SingleNickReleasedTargetSearchRequest",
    "build_release_catalog_info",
    "build_released_nickase_catalog_info",
    "infer_released_search_final_geometry_source",
    "route_family_active_strand",
    "route_family_final_geometry_source",
    "route_family_physical_nicked_strand",
    "route_family_retained_partner_strand",
    "released_route_semantics",
    "released_search_route_policy",
]
