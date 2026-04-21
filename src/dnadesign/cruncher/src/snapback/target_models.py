"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/snapback/target_models.py

Contracts for target-first snapback catalog search.

Module Author(s): Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from dnadesign.cruncher.snapback.models import (
    EFFECTIVE_CAP_LOOP_NT,
    CatalogNormalizationInfo,
    SnapbackCandidateDesign,
    SnapbackIssue,
    StrictSnapbackModel,
)


class SnapbackTargetGeometry(StrictSnapbackModel):
    nick_boundary_from_left: int = Field(ge=0)
    paired_bp: int = Field(ge=1)
    cap_nt: int = Field(ge=0)
    require_site_sequence_preserved: bool = True

    @model_validator(mode="after")
    def _validate_cap_nt(self) -> "SnapbackTargetGeometry":
        if self.cap_nt != EFFECTIVE_CAP_LOOP_NT:
            raise ValueError(
                f"target.cap_nt must equal the fixed snapback effective cap loop size of {EFFECTIVE_CAP_LOOP_NT}."
            )
        if self.require_site_sequence_preserved is not True:
            raise ValueError("target.require_site_sequence_preserved is reserved in v1 and must remain true.")
        return self


class SnapbackTargetFeasibilityRow(StrictSnapbackModel):
    variant_id: str
    orientation: Literal["forward", "reverse"]
    motif_top_5to3: str
    motif_len: int = Field(ge=1)
    site_start_at_target_boundary: int
    site_end_at_target_boundary: int
    boundary_offset: int
    outside_site: bool | None = None
    exact_boundary_hit_possible: bool
    exact_boundary_blockers: list[str] = Field(default_factory=list)
    any_boundary_hit_possible: bool
    earliest_feasible_boundary: int | None = Field(default=None, ge=0)
    exact_input_length_nt: int | None = Field(default=None, ge=0)
    earliest_input_length_nt: int | None = Field(default=None, ge=0)


class SnapbackTargetSearchHit(StrictSnapbackModel):
    rank: int = Field(ge=1)
    hit_kind: Literal["exact", "nearest"]
    variant_id: str
    intended_site_orientation: Literal["forward", "reverse"]
    intended_site_sequence: str
    nick_boundary_from_left: int = Field(ge=0)
    site_start: int = Field(ge=0)
    site_end: int = Field(ge=0)
    input_sequence: str
    designed_sequence: str
    input_length_nt: int = Field(ge=0)
    designed_length_nt: int = Field(ge=0)
    paired_bp: int = Field(ge=1)
    cap_nt: int = Field(ge=0)
    source_cap_nt: int = Field(ge=0)
    cap_extension_nt: int = Field(ge=0)
    site_mutation_count: int = Field(ge=0)
    extra_nick_event_count: int = Field(ge=0)
    extra_target_strand_nick_count: int = Field(ge=0)
    nickase: CatalogNormalizationInfo
    explicit_report: SnapbackCandidateDesign


class SnapbackTargetSearchMetadata(StrictSnapbackModel):
    spec_schema_version: int = 1
    contract: Literal["single_nick_snapback_target_search_v1"] = "single_nick_snapback_target_search_v1"
    coordinate_semantics: Literal["half_open_zero_based_v1"] = "half_open_zero_based_v1"
    boundary_semantics: Literal["closed_zero_based_boundary_v1"] = "closed_zero_based_boundary_v1"
    catalog_preset: str | None = None
    catalog_presets: list[str] = Field(default_factory=list)
    catalog_additional_paths: list[str] = Field(default_factory=list)
    catalog_source: str
    target: SnapbackTargetGeometry
    evaluated_orientation_count: int = Field(ge=0)
    exact_hit_count: int = Field(ge=0)
    near_hit_count: int = Field(ge=0)


class SnapbackTargetSearchReport(StrictSnapbackModel):
    schema_version: Literal[1] = 1
    workflow: Literal["snapback_target_search"] = "snapback_target_search"
    status: Literal["exact_hits_found", "near_hits_only", "no_hits", "invalid_catalog"]
    workspace_root: str | None = None
    metadata: SnapbackTargetSearchMetadata
    issues: list[SnapbackIssue] = Field(default_factory=list)
    exact_hits: list[SnapbackTargetSearchHit] = Field(default_factory=list)
    near_hits: list[SnapbackTargetSearchHit] = Field(default_factory=list)
    feasibility: list[SnapbackTargetFeasibilityRow] = Field(default_factory=list)
