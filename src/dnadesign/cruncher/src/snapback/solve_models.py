"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/snapback/solve_models.py

Schema and report contracts for v2 snapback solve workflows.

Module Author(s): Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator

from dnadesign.cruncher.snapback.models import (
    BoundaryRange,
    BoundedIntRange,
    CanonicalTopStrandSpec,
    CatalogSources,
    FractionRange,
    SnapbackCandidateDesign,
    SnapbackIssue,
    StrictSnapbackModel,
)


class SnapbackSolveHeader(StrictSnapbackModel):
    schema_version: Literal[2] = 2
    contract: Literal["single_nick_snapback_solve_v2"] = "single_nick_snapback_solve_v2"
    name: str

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError("snapback_solve.name must be non-empty.")
        return text


class SnapbackSolveInputSpec(StrictSnapbackModel):
    canonical_top_strand: CanonicalTopStrandSpec


class NickasePolicySpec(StrictSnapbackModel):
    allowed_variant_ids: list[str]
    normalize_to_top_strand_nick: bool = True

    @field_validator("allowed_variant_ids")
    @classmethod
    def _validate_variant_ids(cls, value: list[str]) -> list[str]:
        normalized = [str(item or "").strip() for item in value]
        if not normalized or any(not item for item in normalized):
            raise ValueError("nickase_policy.allowed_variant_ids must contain at least one non-empty value.")
        if len(set(normalized)) != len(normalized):
            raise ValueError("nickase_policy.allowed_variant_ids must not repeat values.")
        return normalized


class SnapbackSolveGoalSpec(StrictSnapbackModel):
    nick_boundary_window: BoundaryRange
    retained_start_from_nick: BoundedIntRange


class SnapbackSearchSpec(StrictSnapbackModel):
    retained_homology_length: BoundedIntRange
    cap_nt: BoundedIntRange
    max_added_nt: int = Field(ge=0)
    max_mismatches: int = Field(ge=0, le=2)
    max_enumerated_candidates: int = Field(ge=1)
    max_search_nodes: int = Field(ge=1)
    max_hits: int = Field(ge=1)
    materialize_top_k: int = Field(ge=0)

    @model_validator(mode="after")
    def _validate_bounds(self) -> "SnapbackSearchSpec":
        if self.materialize_top_k > self.max_hits:
            raise ValueError("search.materialize_top_k must be <= max_hits.")
        return self


class SnapbackSolveConstraintsSpec(StrictSnapbackModel):
    terminal_ligatable_duplex_bp: BoundedIntRange
    max_uninterrupted_duplex_bp: int = Field(ge=0)
    forbid_additional_target_strand_nicks: bool = False
    forbid_any_additional_nicks: bool = False

    @model_validator(mode="after")
    def _validate_bounds(self) -> "SnapbackSolveConstraintsSpec":
        if self.max_uninterrupted_duplex_bp < self.terminal_ligatable_duplex_bp.min:
            raise ValueError("max_uninterrupted_duplex_bp must be >= terminal_ligatable_duplex_bp.min.")
        return self


class SnapbackSolveSequenceQualitySpec(StrictSnapbackModel):
    gc_fraction: FractionRange | None = None
    max_homopolymer_run: int | None = Field(default=None, ge=1)


class SnapbackSolveOutputConfig(StrictSnapbackModel):
    run_dir: Path = Path("outputs/snapback_solves")
    emit_visual_contracts: bool = True
    emit_baserender_jobs: bool = True

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


class SingleNickSnapbackSolveSpec(StrictSnapbackModel):
    snapback_solve: SnapbackSolveHeader
    input: SnapbackSolveInputSpec
    catalog: CatalogSources
    nickase_policy: NickasePolicySpec
    goal: SnapbackSolveGoalSpec
    search: SnapbackSearchSpec
    constraints: SnapbackSolveConstraintsSpec
    sequence_quality: SnapbackSolveSequenceQualitySpec = Field(default_factory=SnapbackSolveSequenceQualitySpec)
    output: SnapbackSolveOutputConfig = Field(default_factory=SnapbackSolveOutputConfig)

    @model_validator(mode="after")
    def _validate_output_flags(self) -> "SingleNickSnapbackSolveSpec":
        if self.output.emit_baserender_jobs and not self.output.emit_visual_contracts:
            raise ValueError("output.emit_baserender_jobs requires output.emit_visual_contracts: true.")
        return self

    @property
    def name(self) -> str:
        return self.snapback_solve.name


class SnapbackSolveReportMetadata(StrictSnapbackModel):
    spec_schema_version: int = 2
    contract: Literal["single_nick_snapback_solve_v2"] = "single_nick_snapback_solve_v2"
    coordinate_semantics: Literal["half_open_zero_based_v1"] = "half_open_zero_based_v1"
    boundary_semantics: Literal["closed_zero_based_boundary_v1"] = "closed_zero_based_boundary_v1"
    catalog_preset: str | None = None
    catalog_additional_paths: list[str] = Field(default_factory=list)
    visited_search_node_count: int = Field(ge=0)
    enumerated_candidate_count: int = Field(ge=0)
    accepted_candidate_count: int = Field(ge=0)
    materialized_hit_count: int = Field(ge=0)
    search_truncated: bool = False
    warnings: list[str] = Field(default_factory=list)
    warning_codes: list[str] = Field(default_factory=list)


class SnapbackSolveHit(StrictSnapbackModel):
    rank: int = Field(ge=1)
    hit_id: str
    variant_id: str
    intended_site_orientation: str
    nick_boundary: int = Field(ge=0)
    nick_boundary_from_left: int = Field(ge=0)
    retained_start_from_nick: int = Field(ge=0)
    cap_sequence: str
    foldback_arm: str
    added_nt: int = Field(ge=0)
    paired_bp: int = Field(ge=0)
    mismatch_count: int = Field(ge=0)
    terminal_ligatable_duplex_bp: int = Field(ge=0)
    max_uninterrupted_duplex_bp: int = Field(ge=0)
    extra_nick_event_count: int = Field(ge=0)
    gc_fraction_added: float = Field(ge=0.0, le=1.0)
    materialized_run_dir: str | None = None
    explicit_report: SnapbackCandidateDesign


class SnapbackSolveReport(StrictSnapbackModel):
    schema_version: Literal[2] = 2
    workflow: Literal["snapback_solve"] = "snapback_solve"
    status: Literal["satisfied", "no_hits", "search_truncated", "invalid_spec", "invalid_catalog"]
    spec_name: str
    spec_path: str
    workspace_root: str | None = None
    solve_id: str | None = None
    run_dir: str | None = None
    metadata: SnapbackSolveReportMetadata
    issues: list[SnapbackIssue] = Field(default_factory=list)
    hits: list[SnapbackSolveHit] = Field(default_factory=list)
