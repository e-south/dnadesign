"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/snapback/released_search_models.py

Search-side contracts for released-product snapback.

Module Author(s): Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator, model_validator

from dnadesign.cruncher.nickases.models import NickEvent, RecognitionSiteInstance, normalize_dna
from dnadesign.cruncher.release_enzymes.models import ReleaseCutEvent, ReleaseRecognitionSiteInstance
from dnadesign.cruncher.snapback.models import (
    CatalogNormalizationInfo,
    CatalogSources,
    SnapbackIssue,
    StrictSnapbackModel,
)
from dnadesign.cruncher.snapback.released_projection_models import (
    ReleaseCatalogNormalizationInfo,
    ReleasedFinalCandidate,
    ReleasedProductProjection,
)
from dnadesign.cruncher.snapback.released_route_policy import (
    _DEFAULT_ALLOWED_ACTIVE_STRANDS,
    _DEFAULT_ALLOWED_ROUTE_FAMILIES,
    _DEFAULT_DISALLOWED_NICKASE_WARNING_CODES,
    ReleasedActiveStrand,
    ReleasedFinalGeometrySource,
    ReleasedRouteFamily,
    infer_released_search_final_geometry_source,
    normalize_active_strand_list,
    normalize_route_family_list,
    normalize_warning_code_list,
    route_family_active_strand,
    route_family_physical_nicked_strand,
)
from dnadesign.cruncher.snapback.released_spec_models import (
    ReleaseCatalogSources,
    ReleasedFinalTargetGeometry,
)


class ReleasedTargetSearchConfig(StrictSnapbackModel):
    route_policy_final_geometry_source: ReleasedFinalGeometrySource = "exposed_bottom_strand"
    allow_post_release_loss_of_nickase_site: bool = True
    allow_precut_footprint_outside_active_product: bool = False
    allow_demo_hits: bool = False
    disallowed_nickase_warning_codes: list[str] = Field(
        default_factory=lambda: list(_DEFAULT_DISALLOWED_NICKASE_WARNING_CODES)
    )
    allowed_active_strands: list[ReleasedActiveStrand] = Field(
        default_factory=lambda: list(_DEFAULT_ALLOWED_ACTIVE_STRANDS)
    )
    allowed_route_families: list[ReleasedRouteFamily] = Field(
        default_factory=lambda: list(_DEFAULT_ALLOWED_ROUTE_FAMILIES)
    )
    retained_side: Literal["upstream"] = "upstream"
    stage_order: Literal["nick_then_release"] = "nick_then_release"
    max_results: int = Field(default=8, ge=1, le=64)
    near_boundary_search_limit: int = Field(default=8, ge=0, le=64)

    @model_validator(mode="after")
    def _validate_supported_mode(self) -> "ReleasedTargetSearchConfig":
        if self.allowed_active_strands == []:
            raise ValueError("search.allowed_active_strands must not be empty.")
        if self.allowed_route_families == []:
            raise ValueError("search.allowed_route_families must not be empty.")
        inferred_final_geometry_source = infer_released_search_final_geometry_source(
            allowed_active_strands=self.allowed_active_strands,
            allowed_route_families=self.allowed_route_families,
        )
        if (
            "route_policy_final_geometry_source" in self.model_fields_set
            and self.route_policy_final_geometry_source != inferred_final_geometry_source
        ):
            raise ValueError("search.route_policy_final_geometry_source must match search.allowed_route_families.")
        self.route_policy_final_geometry_source = inferred_final_geometry_source
        return self

    @field_validator("disallowed_nickase_warning_codes")
    @classmethod
    def _validate_search_disallowed_nickase_warning_codes(cls, value: list[str]) -> list[str]:
        return normalize_warning_code_list(value, label="search.disallowed_nickase_warning_codes")

    @field_validator("allowed_active_strands")
    @classmethod
    def _validate_allowed_active_strands(cls, value: list[str]) -> list[ReleasedActiveStrand]:
        return normalize_active_strand_list(value, label="search.allowed_active_strands")

    @field_validator("allowed_route_families")
    @classmethod
    def _validate_allowed_route_families(cls, value: list[str]) -> list[ReleasedRouteFamily]:
        return normalize_route_family_list(value, label="search.allowed_route_families")


class SingleNickReleasedTargetSearchRequest(StrictSnapbackModel):
    schema_version: Literal[1] = 1
    kind: Literal["single_nick_released_target_v1"] = "single_nick_released_target_v1"
    target: ReleasedFinalTargetGeometry
    nick_sources: CatalogSources
    release_sources: ReleaseCatalogSources
    search: ReleasedTargetSearchConfig = Field(default_factory=ReleasedTargetSearchConfig)


class ReleasedTargetSearchHit(StrictSnapbackModel):
    rank: int = Field(ge=1)
    hit_kind: Literal["exact", "nearest"]
    route_family: ReleasedRouteFamily = "bottom_active_from_top_nick"
    active_strand: ReleasedActiveStrand = "bottom"
    physical_nicked_strand: ReleasedActiveStrand = "top"
    nickase_variant_id: str
    release_variant_id: str
    intended_nick_site_orientation: Literal["forward", "reverse"]
    intended_nick_site_sequence: str
    release_site_orientation: Literal["forward", "reverse"]
    release_site_sequence: str
    nick_boundary_from_left: int = Field(ge=0)
    active_product_input_length_nt: int = Field(default=0, ge=0)
    active_product_length_nt: int = Field(default=0, ge=0)
    precursor_length_nt: int = Field(ge=0)
    sacrificial_downstream_tail_nt: int = Field(ge=0)
    extra_nick_event_count: int = Field(ge=0)
    extra_target_strand_nick_count: int = Field(ge=0)
    precursor_top_strand: str
    pre_nick_site: RecognitionSiteInstance
    pre_nick_event: NickEvent
    release_site: ReleaseRecognitionSiteInstance
    release_event: ReleaseCutEvent
    nickase: CatalogNormalizationInfo
    release_enzyme: ReleaseCatalogNormalizationInfo
    projection: ReleasedProductProjection
    final_candidate: ReleasedFinalCandidate

    @field_validator("precursor_top_strand")
    @classmethod
    def _validate_precursor_top_strand(cls, value: str) -> str:
        return normalize_dna(value)

    @model_validator(mode="after")
    def _validate_route_projection_mirrors(self) -> "ReleasedTargetSearchHit":
        if route_family_active_strand(self.route_family) != self.active_strand:
            raise ValueError("target-search hit route_family must match active_strand.")
        if route_family_physical_nicked_strand(self.route_family) != self.physical_nicked_strand:
            raise ValueError("target-search hit route_family must match physical_nicked_strand.")
        if self.projection.route_family != self.route_family:
            raise ValueError("target-search hit route_family must mirror projection.route_family.")
        if self.projection.active_strand != self.active_strand:
            raise ValueError("target-search hit active_strand must mirror projection.active_strand.")
        if self.projection.physical_nicked_strand != self.physical_nicked_strand:
            raise ValueError("target-search hit physical_nicked_strand must mirror projection.physical_nicked_strand.")
        if self.final_candidate.route_family != self.route_family:
            raise ValueError("target-search hit route_family must mirror final_candidate.route_family.")
        if self.final_candidate.active_strand != self.active_strand:
            raise ValueError("target-search hit active_strand must mirror final_candidate.active_strand.")
        if self.final_candidate.physical_nicked_strand != self.physical_nicked_strand:
            raise ValueError(
                "target-search hit physical_nicked_strand must mirror final_candidate.physical_nicked_strand."
            )
        return self


class ReleasedTargetSearchMetadata(StrictSnapbackModel):
    schema_version: int = 1
    kind: Literal["single_nick_released_target_v1"] = "single_nick_released_target_v1"
    coordinate_semantics: Literal["half_open_zero_based_v1"] = "half_open_zero_based_v1"
    boundary_semantics: Literal["closed_zero_based_boundary_v1"] = "closed_zero_based_boundary_v1"
    route_policy_final_geometry_source: ReleasedFinalGeometrySource = "exposed_bottom_strand"
    target: ReleasedFinalTargetGeometry
    nick_catalog_source: str
    release_catalog_source: str
    disallowed_nickase_warning_codes: list[str] = Field(default_factory=list)
    allowed_active_strands: list[ReleasedActiveStrand] = Field(default_factory=list)
    allowed_route_families: list[ReleasedRouteFamily] = Field(default_factory=list)
    evaluated_pair_count: int = Field(ge=0)
    pre_truncation_exact_hit_count: int = Field(ge=0)
    post_truncation_exact_hit_count: int = Field(ge=0)
    pre_truncation_near_hit_count: int = Field(ge=0)
    post_truncation_near_hit_count: int = Field(ge=0)
    blocker_counts: dict[str, int] = Field(default_factory=dict)

    @field_validator("disallowed_nickase_warning_codes")
    @classmethod
    def _validate_target_search_disallowed_nickase_warning_codes(cls, value: list[str]) -> list[str]:
        return normalize_warning_code_list(value, label="metadata.disallowed_nickase_warning_codes")

    @field_validator("allowed_active_strands")
    @classmethod
    def _validate_metadata_allowed_active_strands(cls, value: list[str]) -> list[ReleasedActiveStrand]:
        return normalize_active_strand_list(value, label="metadata.allowed_active_strands")

    @field_validator("allowed_route_families")
    @classmethod
    def _validate_metadata_allowed_route_families(cls, value: list[str]) -> list[ReleasedRouteFamily]:
        return normalize_route_family_list(value, label="metadata.allowed_route_families")

    @model_validator(mode="after")
    def _validate_metadata_route_policy(self) -> "ReleasedTargetSearchMetadata":
        if self.allowed_active_strands == []:
            raise ValueError("metadata.allowed_active_strands must not be empty.")
        if self.allowed_route_families == []:
            raise ValueError("metadata.allowed_route_families must not be empty.")
        inferred_final_geometry_source = infer_released_search_final_geometry_source(
            allowed_active_strands=self.allowed_active_strands,
            allowed_route_families=self.allowed_route_families,
        )
        if (
            "route_policy_final_geometry_source" in self.model_fields_set
            and self.route_policy_final_geometry_source != inferred_final_geometry_source
        ):
            raise ValueError("metadata.route_policy_final_geometry_source must match metadata.allowed_route_families.")
        self.route_policy_final_geometry_source = inferred_final_geometry_source
        return self


class ReleasedTargetSearchReport(StrictSnapbackModel):
    schema_version: Literal[1] = 1
    workflow: Literal["snapback_released_target_search"] = "snapback_released_target_search"
    status: Literal["exact_hits_found", "near_hits_only", "no_hits", "invalid_catalog"]
    workspace_root: str | None = None
    metadata: ReleasedTargetSearchMetadata
    issues: list[SnapbackIssue] = Field(default_factory=list)
    exact_hits: list[ReleasedTargetSearchHit] = Field(default_factory=list)
    near_hits: list[ReleasedTargetSearchHit] = Field(default_factory=list)


__all__ = [
    "ReleasedTargetSearchConfig",
    "ReleasedTargetSearchHit",
    "ReleasedTargetSearchMetadata",
    "ReleasedTargetSearchReport",
    "SingleNickReleasedTargetSearchRequest",
]
