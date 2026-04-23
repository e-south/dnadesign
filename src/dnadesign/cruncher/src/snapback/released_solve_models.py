"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/snapback/released_solve_models.py

Report and solve materialization contracts for released-product snapback.

Module Author(s): Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator

from dnadesign.cruncher.nickases.models import NickEvent, RecognitionSiteInstance
from dnadesign.cruncher.release_enzymes.models import ReleaseCutEvent, ReleaseRecognitionSiteInstance
from dnadesign.cruncher.snapback.models import CatalogNormalizationInfo, SnapbackIssue, StrictSnapbackModel
from dnadesign.cruncher.snapback.released_projection_models import (
    ReleaseCatalogNormalizationInfo,
    ReleasedFinalCandidate,
    ReleasedProductProjection,
)
from dnadesign.cruncher.snapback.released_route_policy import (
    ReleasedActiveStrand,
    ReleasedFinalGeometrySource,
    ReleasedRouteFamily,
    normalize_active_strand_list,
    normalize_route_family_list,
    normalize_warning_code_list,
)
from dnadesign.cruncher.snapback.released_search_models import ReleasedTargetSearchHit, ReleasedTargetSearchReport
from dnadesign.cruncher.snapback.released_spec_models import ReleasedFinalTargetGeometry


class ReleasedSnapbackReportMetadata(StrictSnapbackModel):
    schema_version: int = 1
    kind: Literal["single_nick_released_snapback_v1"] = "single_nick_released_snapback_v1"
    coordinate_semantics: Literal["half_open_zero_based_v1"] = "half_open_zero_based_v1"
    boundary_semantics: Literal["closed_zero_based_boundary_v1"] = "closed_zero_based_boundary_v1"
    final_geometry_source: ReleasedFinalGeometrySource = "exposed_bottom_strand"
    nick_catalog_source: str
    release_catalog_source: str
    disallowed_nickase_warning_codes: list[str] = Field(default_factory=list)
    final_target: ReleasedFinalTargetGeometry
    nickase_catalog_variants: list[CatalogNormalizationInfo] = Field(default_factory=list)
    release_catalog_variants: list[ReleaseCatalogNormalizationInfo] = Field(default_factory=list)

    @field_validator("disallowed_nickase_warning_codes")
    @classmethod
    def _validate_report_disallowed_nickase_warning_codes(cls, value: list[str]) -> list[str]:
        return normalize_warning_code_list(value, label="metadata.disallowed_nickase_warning_codes")


class ReleasedSnapbackEvaluationReport(StrictSnapbackModel):
    schema_version: Literal[1] = 1
    workflow: Literal["snapback_released_design"] = "snapback_released_design"
    status: Literal[
        "satisfied",
        "unsatisfied",
        "invalid_catalog",
        "invalid_precursor",
        "no_release_path",
        "post_release_projection_failed",
    ]
    spec_name: str
    workspace_root: str
    spec_path: str
    metadata: ReleasedSnapbackReportMetadata
    issues: list[SnapbackIssue] = Field(default_factory=list)
    pre_nick_site: RecognitionSiteInstance | None = None
    pre_nick_event: NickEvent | None = None
    release_site: ReleaseRecognitionSiteInstance | None = None
    release_event: ReleaseCutEvent | None = None
    projection: ReleasedProductProjection | None = None
    candidate: ReleasedFinalCandidate | None = None
    run_dir: str | None = None


class ReleasedSolveOutputConfig(StrictSnapbackModel):
    run_dir: Path = Path("outputs/released_solve")
    materialize_top_k: int = Field(default=8, ge=1, le=64)
    render_format: Literal["png", "svg", "pdf"] = "pdf"
    emit_renders: bool = False

    @field_validator("run_dir", mode="before")
    @classmethod
    def _validate_run_dir(cls, value: Path | str) -> Path | str:
        raw_text = str(value or "").strip()
        if not raw_text:
            raise ValueError("output.run_dir must be non-empty.")
        path = Path(raw_text)
        if path.is_absolute():
            raise ValueError("output.run_dir must be a relative path inside the workspace.")
        if any(part == ".." for part in path.parts):
            raise ValueError("output.run_dir must not traverse outside the workspace.")
        return raw_text


class ReleasedSolveHit(StrictSnapbackModel):
    rank: int = Field(ge=1)
    hit_kind: Literal["exact", "nearest"]
    nickase_variant_id: str
    release_variant_id: str
    materialized_run_dir: str
    render_job_path: str | None = None
    rendered_plot_path: str | None = None
    target_search_hit: ReleasedTargetSearchHit


class ReleasedSolveReportMetadata(StrictSnapbackModel):
    schema_version: int = 1
    kind: Literal["single_nick_released_solve_v1"] = "single_nick_released_solve_v1"
    coordinate_semantics: Literal["half_open_zero_based_v1"] = "half_open_zero_based_v1"
    boundary_semantics: Literal["closed_zero_based_boundary_v1"] = "closed_zero_based_boundary_v1"
    final_geometry_source: ReleasedFinalGeometrySource = "exposed_bottom_strand"
    target: ReleasedFinalTargetGeometry
    nick_catalog_source: str
    release_catalog_source: str
    disallowed_nickase_warning_codes: list[str] = Field(default_factory=list)
    allowed_active_strands: list[ReleasedActiveStrand] = Field(default_factory=list)
    allowed_route_families: list[ReleasedRouteFamily] = Field(default_factory=list)
    evaluated_pair_count: int = Field(ge=0)
    available_exact_hit_count: int = Field(ge=0)
    available_near_hit_count: int = Field(ge=0)
    selected_hit_kind: Literal["exact", "nearest"] | None = None
    materialized_hit_count: int = Field(ge=0)
    requested_materialize_top_k: int = Field(ge=1, le=64)
    render_format: Literal["png", "svg", "pdf"]
    emit_renders: bool = False
    blocker_counts: dict[str, int] = Field(default_factory=dict)

    @field_validator("disallowed_nickase_warning_codes")
    @classmethod
    def _validate_solve_disallowed_nickase_warning_codes(cls, value: list[str]) -> list[str]:
        return normalize_warning_code_list(value, label="metadata.disallowed_nickase_warning_codes")

    @field_validator("allowed_active_strands")
    @classmethod
    def _validate_solve_allowed_active_strands(cls, value: list[str]) -> list[ReleasedActiveStrand]:
        return normalize_active_strand_list(value, label="metadata.allowed_active_strands")

    @field_validator("allowed_route_families")
    @classmethod
    def _validate_solve_allowed_route_families(cls, value: list[str]) -> list[ReleasedRouteFamily]:
        return normalize_route_family_list(value, label="metadata.allowed_route_families")


class ReleasedSolveReport(StrictSnapbackModel):
    schema_version: Literal[1] = 1
    workflow: Literal["snapback_released_solve"] = "snapback_released_solve"
    status: Literal["exact_hits_materialized", "near_hits_materialized", "no_hits", "invalid_catalog"]
    workspace_root: str | None = None
    run_dir: str | None = None
    metadata: ReleasedSolveReportMetadata
    issues: list[SnapbackIssue] = Field(default_factory=list)
    search_report: ReleasedTargetSearchReport
    hits: list[ReleasedSolveHit] = Field(default_factory=list)


__all__ = [
    "ReleasedSnapbackEvaluationReport",
    "ReleasedSnapbackReportMetadata",
    "ReleasedSolveHit",
    "ReleasedSolveOutputConfig",
    "ReleasedSolveReport",
    "ReleasedSolveReportMetadata",
]
