"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/snapback/solve_models.py

Schema and report contracts for v3 co-design snapback solve workflows.

Module Author(s): Eric J. South
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
    CatalogNormalizationInfo,
    CatalogSources,
    FractionRange,
    SnapbackCandidateDesign,
    SnapbackIssue,
    StrictSnapbackModel,
)


class SnapbackSolveHeader(StrictSnapbackModel):
    schema_version: Literal[3] = 3
    contract: Literal["single_nick_snapback_solve_v3"] = "single_nick_snapback_solve_v3"
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


class SnapbackSolveOrientationPolicySpec(StrictSnapbackModel):
    normalize_to_top_strand_nick: bool = True


class SnapbackSolveGoalSpec(StrictSnapbackModel):
    nick_boundary_window: BoundaryRange | None = None


class SnapbackSearchSpec(StrictSnapbackModel):
    retained_homology_length: BoundedIntRange | None = None
    min_paired_bp: int = Field(default=3, ge=1)
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
        if self.retained_homology_length is not None and self.retained_homology_length.min < self.min_paired_bp:
            raise ValueError("search.retained_homology_length.min must be >= search.min_paired_bp.")
        return self


class SnapbackSolveConstraintsSpec(StrictSnapbackModel):
    terminal_ligatable_duplex_bp: BoundedIntRange | None = None
    max_uninterrupted_duplex_bp: int | None = Field(default=None, ge=0)
    forbid_additional_target_strand_nicks: bool = False
    forbid_any_additional_nicks: bool = False

    @model_validator(mode="after")
    def _validate_bounds(self) -> "SnapbackSolveConstraintsSpec":
        if (
            self.terminal_ligatable_duplex_bp is not None
            and self.max_uninterrupted_duplex_bp is not None
            and self.max_uninterrupted_duplex_bp < self.terminal_ligatable_duplex_bp.min
        ):
            raise ValueError("max_uninterrupted_duplex_bp must be >= terminal_ligatable_duplex_bp.min.")
        return self


class SnapbackSolveSequenceQualitySpec(StrictSnapbackModel):
    gc_fraction: FractionRange | None = None
    max_homopolymer_run: int | None = Field(default=None, ge=1)


class SnapbackSolveOutputConfig(StrictSnapbackModel):
    run_dir: Path = Path("outputs/solve")
    emit_visual_contracts: bool = True
    emit_baserender_jobs: bool = True
    render_format: Literal["png", "svg", "pdf"] = "png"

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


class SnapbackSolveResolvedSearchSpace(StrictSnapbackModel):
    nick_boundary_window: BoundaryRange
    retained_homology_length: BoundedIntRange
    min_paired_bp: int = Field(ge=1)
    terminal_ligatable_duplex_bp: BoundedIntRange
    max_uninterrupted_duplex_bp: int = Field(ge=0)


class SnapbackSolveFrontierRow(StrictSnapbackModel):
    nick_boundary_from_left: int = Field(ge=0)
    paired_bp: int = Field(ge=1)
    cap_extension_nt: int = Field(ge=0)
    codesigned_input_count: int = Field(ge=0)
    enumerated_candidate_count: int = Field(ge=0)
    accepted_candidate_count: int = Field(ge=0)


class SingleNickSnapbackSolveSpec(StrictSnapbackModel):
    snapback_solve: SnapbackSolveHeader
    input: SnapbackSolveInputSpec
    catalog: CatalogSources
    orientation_policy: SnapbackSolveOrientationPolicySpec = Field(default_factory=SnapbackSolveOrientationPolicySpec)
    goal: SnapbackSolveGoalSpec = Field(default_factory=SnapbackSolveGoalSpec)
    search: SnapbackSearchSpec
    constraints: SnapbackSolveConstraintsSpec
    sequence_quality: SnapbackSolveSequenceQualitySpec = Field(default_factory=SnapbackSolveSequenceQualitySpec)
    output: SnapbackSolveOutputConfig = Field(default_factory=SnapbackSolveOutputConfig)

    @model_validator(mode="after")
    def _validate_output_flags(self) -> "SingleNickSnapbackSolveSpec":
        if self.output.emit_baserender_jobs and not self.output.emit_visual_contracts:
            raise ValueError("output.emit_baserender_jobs requires output.emit_visual_contracts: true.")
        input_len = len(self.input.canonical_top_strand.sequence)
        if self.goal.nick_boundary_window is not None and self.goal.nick_boundary_window.max > input_len:
            raise ValueError("goal.nick_boundary_window must stay inside input.canonical_top_strand.sequence.")
        if self.search.min_paired_bp > input_len:
            raise ValueError("search.min_paired_bp must be <= input.canonical_top_strand.sequence length.")
        if self.search.retained_homology_length is not None and self.search.retained_homology_length.max > input_len:
            raise ValueError("search.retained_homology_length must stay inside input.canonical_top_strand.sequence.")
        if (
            self.constraints.terminal_ligatable_duplex_bp is not None
            and self.constraints.terminal_ligatable_duplex_bp.min < self.search.min_paired_bp
        ):
            raise ValueError("constraints.terminal_ligatable_duplex_bp.min must be >= search.min_paired_bp.")
        if (
            self.constraints.max_uninterrupted_duplex_bp is not None
            and self.constraints.max_uninterrupted_duplex_bp < self.search.min_paired_bp
        ):
            raise ValueError("constraints.max_uninterrupted_duplex_bp must be >= search.min_paired_bp.")
        return self

    @property
    def name(self) -> str:
        return self.snapback_solve.name

    def resolved_nick_boundary_window(self) -> BoundaryRange:
        input_len = len(self.input.canonical_top_strand.sequence)
        return self.goal.nick_boundary_window or BoundaryRange(min=0, max=input_len)

    def resolved_retained_homology_length(self) -> BoundedIntRange:
        input_len = len(self.input.canonical_top_strand.sequence)
        return self.search.retained_homology_length or BoundedIntRange(
            min=self.search.min_paired_bp,
            max=input_len,
        )

    def resolved_terminal_ligatable_duplex_bp(self) -> BoundedIntRange:
        input_len = len(self.input.canonical_top_strand.sequence)
        return self.constraints.terminal_ligatable_duplex_bp or BoundedIntRange(
            min=self.search.min_paired_bp,
            max=input_len,
        )

    def resolved_max_uninterrupted_duplex_bp(self) -> int:
        input_len = len(self.input.canonical_top_strand.sequence)
        if self.constraints.max_uninterrupted_duplex_bp is not None:
            return self.constraints.max_uninterrupted_duplex_bp
        return input_len

    def resolved_search_space(self) -> SnapbackSolveResolvedSearchSpace:
        return SnapbackSolveResolvedSearchSpace(
            nick_boundary_window=self.resolved_nick_boundary_window(),
            retained_homology_length=self.resolved_retained_homology_length(),
            min_paired_bp=self.search.min_paired_bp,
            terminal_ligatable_duplex_bp=self.resolved_terminal_ligatable_duplex_bp(),
            max_uninterrupted_duplex_bp=self.resolved_max_uninterrupted_duplex_bp(),
        )


class SnapbackSolveReportMetadata(StrictSnapbackModel):
    spec_schema_version: int = 3
    contract: Literal["single_nick_snapback_solve_v3"] = "single_nick_snapback_solve_v3"
    coordinate_semantics: Literal["half_open_zero_based_v1"] = "half_open_zero_based_v1"
    boundary_semantics: Literal["closed_zero_based_boundary_v1"] = "closed_zero_based_boundary_v1"
    catalog_preset: str | None = None
    catalog_presets: list[str] = Field(default_factory=list)
    catalog_additional_paths: list[str] = Field(default_factory=list)
    resolved_search_space: SnapbackSolveResolvedSearchSpace
    visited_search_node_count: int = Field(ge=0)
    enumerated_candidate_count: int = Field(ge=0)
    accepted_candidate_count: int = Field(ge=0)
    materialized_hit_count: int = Field(ge=0)
    frontier_row_count: int = Field(ge=0)
    first_satisfied_frontier: SnapbackSolveFrontierRow | None = None
    search_truncated: bool = False
    warnings: list[str] = Field(default_factory=list)
    warning_codes: list[str] = Field(default_factory=list)


class SnapbackSolveHit(StrictSnapbackModel):
    rank: int = Field(ge=1)
    hit_id: str
    variant_id: str
    intended_site_orientation: str
    intended_site_sequence: str
    nick_boundary: int = Field(ge=0)
    nick_boundary_from_left: int = Field(ge=0)
    site_mutation_count: int = Field(ge=0)
    retained_start_from_nick: int = Field(ge=0)
    cap_nt: int = Field(ge=0)
    cap_extension_nt: int = Field(ge=0)
    cap_sequence: str
    foldback_arm: str
    added_nt: int = Field(ge=0)
    paired_bp: int = Field(ge=0)
    mismatch_count: int = Field(ge=0)
    terminal_ligatable_duplex_bp: int = Field(ge=0)
    max_uninterrupted_duplex_bp: int = Field(ge=0)
    extra_nick_event_count: int = Field(ge=0)
    gc_fraction_added: float = Field(ge=0.0, le=1.0)
    nickase: CatalogNormalizationInfo
    materialized_run_dir: str | None = None
    explicit_report: SnapbackCandidateDesign


class SnapbackSolveReport(StrictSnapbackModel):
    schema_version: Literal[3] = 3
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
    frontier: list[SnapbackSolveFrontierRow] = Field(default_factory=list)
