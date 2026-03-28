"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/yiu/models/solve.py

YIU solve/search schema contracts.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from dnadesign.cruncher.bio.iupac import normalize_iupac
from dnadesign.cruncher.config.schema_v3 import StrictBaseModel
from dnadesign.cruncher.yiu.models.common import _validate_slug


class YiuSolveSourceWindowSpec(StrictBaseModel):
    id: str
    span_ref: str
    alphabet: Literal["dna", "iupac_dna"] = "iupac_dna"
    pattern: str | None = None
    allowed_patterns: list[str] = Field(default_factory=list)

    @field_validator("id", "span_ref")
    @classmethod
    def _validate_id_like(cls, value: str, info) -> str:
        return _validate_slug(value, label=str(info.field_name))

    @field_validator("pattern")
    @classmethod
    def _validate_pattern(cls, value: str | None) -> str | None:
        if value is None:
            return value
        return normalize_iupac(value)

    @field_validator("allowed_patterns")
    @classmethod
    def _validate_allowed_patterns(cls, value: list[str]) -> list[str]:
        return [normalize_iupac(item) for item in value]

    @model_validator(mode="after")
    def _validate_variable_source(self) -> "YiuSolveSourceWindowSpec":
        if self.pattern is None and not self.allowed_patterns:
            raise ValueError("solve source_window requires pattern or allowed_patterns")
        if self.pattern is not None and self.allowed_patterns:
            raise ValueError("solve source_window must use either pattern or allowed_patterns, not both")
        return self


class YiuSolveVariablesSpec(StrictBaseModel):
    source_windows: list[YiuSolveSourceWindowSpec] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_ids(self) -> "YiuSolveVariablesSpec":
        if not self.source_windows:
            raise ValueError("yiu_solve.variables.source_windows must be non-empty")
        ids = [window.id for window in self.source_windows]
        if len(set(ids)) != len(ids):
            raise ValueError("yiu_solve.variables.source_windows ids must be unique")
        return self


class YiuSolveSearchSpec(StrictBaseModel):
    max_hits: int = Field(default=32, ge=1, le=128)
    materialize_top_k: int = Field(default=8, ge=0, le=128)
    max_search_nodes: int = Field(default=100_000, ge=1, le=1_000_000)
    max_enumerated_candidates: int = Field(default=10_000, ge=1, le=1_000_000)

    @model_validator(mode="after")
    def _validate_limits(self) -> "YiuSolveSearchSpec":
        if self.materialize_top_k > self.max_hits:
            raise ValueError("search.materialize_top_k must be <= search.max_hits")
        return self


class YiuSolveCandidatePolicy(StrictBaseModel):
    require_guaranteed_hard_invariants: bool = True
    forbid_possible_hits: bool = True


class YiuSolveOutputSpec(StrictBaseModel):
    run_dir: Path = Path("outputs/yiu/solve")
    emit_view_contracts: bool = True
    emit_baserender_jobs: bool = True
    publish_contract_version: int = 3

    @field_validator("run_dir")
    @classmethod
    def _validate_run_dir(cls, value: Path) -> Path:
        path = Path(value)
        if path.is_absolute():
            raise ValueError("output.run_dir must be relative to the workspace root")
        if ".." in path.parts:
            raise ValueError("output.run_dir must stay inside the workspace root")
        return path

    @field_validator("publish_contract_version")
    @classmethod
    def _validate_publish_contract_version(cls, value: int) -> int:
        if int(value) not in {2, 3}:
            raise ValueError("output.publish_contract_version must be 2 or 3")
        return int(value)

    @model_validator(mode="after")
    def _validate_visual_output_dependencies(self) -> "YiuSolveOutputSpec":
        if self.emit_baserender_jobs and not self.emit_view_contracts:
            raise ValueError("output.emit_baserender_jobs requires output.emit_view_contracts=true.")
        return self


class YiuSolveSpec(StrictBaseModel):
    schema_version: int = 1
    base_spec: Path
    search: YiuSolveSearchSpec = Field(default_factory=YiuSolveSearchSpec)
    variables: YiuSolveVariablesSpec
    candidate_policy: YiuSolveCandidatePolicy = Field(default_factory=YiuSolveCandidatePolicy)
    output: YiuSolveOutputSpec = Field(default_factory=YiuSolveOutputSpec)

    @field_validator("schema_version")
    @classmethod
    def _validate_schema_version(cls, value: int) -> int:
        if int(value) != 1:
            raise ValueError("yiu_solve.schema_version must be 1")
        return int(value)


class YiuSolveSpecDocument(StrictBaseModel):
    yiu_solve: YiuSolveSpec


class YiuSolveIssue(StrictBaseModel):
    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class YiuSolveHit(StrictBaseModel):
    rank: int = Field(ge=1)
    hit_id: str
    score: list[float | int | str]
    source_sequence: str
    variable_assignments: dict[str, str] = Field(default_factory=dict)
    report_status: Literal["satisfied"] = "satisfied"
    materialized_run_dir: str | None = None
    explicit_design_id: str | None = None
    final_state_id: str | None = None
    final_state_view_path: str | None = None
    final_state_job_path: str | None = None


class YiuSolveReportMetadata(StrictBaseModel):
    max_hits: int = 0
    materialize_top_k: int = 0
    warnings: list[str] = Field(default_factory=list)
    warning_codes: list[str] = Field(default_factory=list)
    search_node_count: int = 0
    enumerated_candidate_count: int = 0
    accepted_candidate_count: int = 0
    returned_hit_count: int = 0
    materialized_hit_count: int = 0
    search_truncated: bool = False
    accepted_pool_truncated: bool = False


class YiuSolveReport(StrictBaseModel):
    workflow: Literal["yiu_solve"] = "yiu_solve"
    family: Literal["yiu"] = "yiu"
    status: Literal["solved", "no_hits", "invalid_spec"]
    solve_id: str | None = None
    spec_path: str
    base_spec_path: str | None = None
    run_dir: str | None = None
    metadata: YiuSolveReportMetadata = Field(default_factory=YiuSolveReportMetadata)
    hits: list[YiuSolveHit] = Field(default_factory=list)
    issues: list[YiuSolveIssue] = Field(default_factory=list)
