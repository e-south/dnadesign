"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/yiu/models/solve.py

YIU v4 solve/search schema contracts.

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
from dnadesign.cruncher.yiu.models.v4 import YIU_V4_ALLOWED_SOLVE_WINDOW_OWNER_IDS


class YiuSolvePayloadTargetSpec(StrictBaseModel):
    payload_sequence: str | None = None
    payload_pattern: str | None = None
    bulge_mask: list[int] = Field(default_factory=list)

    @field_validator("payload_sequence", "payload_pattern")
    @classmethod
    def _validate_optional_sequence(cls, value: str | None) -> str | None:
        if value is None:
            return value
        return normalize_iupac(value)

    @field_validator("bulge_mask")
    @classmethod
    def _validate_bulge_mask(cls, value: list[int]) -> list[int]:
        normalized = [int(item) for item in value]
        if len(set(normalized)) != len(normalized):
            raise ValueError("target.bulge_mask positions must be unique")
        if any(item not in {1, 2} for item in normalized):
            raise ValueError("target.bulge_mask positions are allowed only at indices 1 and 2")
        return normalized

    @model_validator(mode="after")
    def _validate_target(self) -> "YiuSolvePayloadTargetSpec":
        if bool(self.payload_sequence) == bool(self.payload_pattern):
            raise ValueError("target must declare exactly one of payload_sequence or payload_pattern")
        return self


class YiuSolveScaffoldWindowSpec(StrictBaseModel):
    id: str
    owner_id: str
    relative_start: int = Field(ge=0)
    relative_end: int = Field(gt=0)
    alphabet: Literal["dna", "iupac_dna"] = "iupac_dna"
    allowed_patterns: list[str] = Field(default_factory=list)

    @field_validator("id", "owner_id")
    @classmethod
    def _validate_id_like(cls, value: str, info) -> str:
        normalized = _validate_slug(value, label=str(info.field_name))
        if str(info.field_name) == "owner_id" and normalized not in YIU_V4_ALLOWED_SOLVE_WINDOW_OWNER_IDS:
            raise ValueError(f"owner_id must be one of {sorted(YIU_V4_ALLOWED_SOLVE_WINDOW_OWNER_IDS)}")
        return normalized

    @field_validator("allowed_patterns")
    @classmethod
    def _validate_allowed_patterns(cls, value: list[str]) -> list[str]:
        return [normalize_iupac(item) for item in value]

    @model_validator(mode="after")
    def _validate_window(self) -> "YiuSolveScaffoldWindowSpec":
        if self.relative_end <= self.relative_start:
            raise ValueError("scaffold_window.relative_end must be > scaffold_window.relative_start")
        if not self.allowed_patterns:
            raise ValueError("scaffold_window.allowed_patterns must be non-empty")
        expected_length = self.relative_end - self.relative_start
        if any(len(pattern) != expected_length for pattern in self.allowed_patterns):
            raise ValueError("each scaffold_window.allowed_patterns entry must match the declared window length")
        return self


class YiuSolveSearchSpec(StrictBaseModel):
    max_search_nodes: int = Field(default=100_000, ge=1, le=1_000_000)
    max_enumerated_candidates: int = Field(default=10_000, ge=1, le=1_000_000)


class YiuSolveSelectionSpec(StrictBaseModel):
    compare_solutions: bool = False
    max_solutions: int = Field(default=1, ge=1, le=32)

    @model_validator(mode="after")
    def _validate_selection(self) -> "YiuSolveSelectionSpec":
        if not self.compare_solutions and self.max_solutions != 1:
            raise ValueError("solve.max_solutions must be 1 when solve.compare_solutions=false")
        return self


class YiuSolveOutputSpec(StrictBaseModel):
    run_dir: Path = Path("outputs/yiu/solve")
    emit_view_contracts: bool = True
    publish_contract_version: int = 4
    persist_render_jobs_debug: bool = False

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
        if int(value) != 4:
            raise ValueError("output.publish_contract_version must be 4 for canonical YIU solve specs")
        return int(value)


class YiuSolveSpec(StrictBaseModel):
    schema_version: int = 1
    base_spec: Path
    target: YiuSolvePayloadTargetSpec
    scaffold_windows: list[YiuSolveScaffoldWindowSpec] = Field(default_factory=list)
    search: YiuSolveSearchSpec = Field(default_factory=YiuSolveSearchSpec)
    solve: YiuSolveSelectionSpec = Field(default_factory=YiuSolveSelectionSpec)
    output: YiuSolveOutputSpec = Field(default_factory=YiuSolveOutputSpec)

    @field_validator("schema_version")
    @classmethod
    def _validate_schema_version(cls, value: int) -> int:
        if int(value) != 1:
            raise ValueError("yiu_solve.schema_version must be 1")
        return int(value)

    @model_validator(mode="after")
    def _validate_windows(self) -> "YiuSolveSpec":
        if not self.scaffold_windows:
            raise ValueError("yiu_solve.scaffold_windows must be non-empty")
        ids = [window.id for window in self.scaffold_windows]
        if len(set(ids)) != len(ids):
            raise ValueError("yiu_solve.scaffold_windows ids must be unique")
        windows_by_owner: dict[str, list[tuple[int, int, str]]] = {}
        for window in self.scaffold_windows:
            rows = windows_by_owner.setdefault(window.owner_id, [])
            for start, end, other_id in rows:
                if not (window.relative_end <= start or end <= window.relative_start):
                    raise ValueError(f"scaffold window {window.id} overlaps scaffold window {other_id}")
            rows.append((window.relative_start, window.relative_end, window.id))
        return self


class YiuSolveSpecDocument(StrictBaseModel):
    yiu_solve: YiuSolveSpec


class YiuSolveIssue(StrictBaseModel):
    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class YiuSolveReportMetadata(StrictBaseModel):
    search_node_count: int = 0
    enumerated_candidate_count: int = 0
    satisfying_solution_count: int = 0
    exhaustive_search: bool = False
    warning_codes: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class YiuSolveReport(StrictBaseModel):
    workflow: Literal["yiu_solve"] = "yiu_solve"
    family: Literal["yiu"] = "yiu"
    status: Literal["solved", "unsatisfied", "incomplete_search", "invalid_spec"]
    solve_id: str | None = None
    spec_path: str
    base_spec_path: str | None = None
    run_dir: str | None = None
    satisfying_solution_count: int = 0
    comparison_solution_count: int = 0
    selected_solution_path: str | None = None
    selected_source_sequence: str | None = None
    metadata: YiuSolveReportMetadata = Field(default_factory=YiuSolveReportMetadata)
    issues: list[YiuSolveIssue] = Field(default_factory=list)
