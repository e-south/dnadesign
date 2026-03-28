"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/yiu/models/reporting.py

YIU validation report and projected-region contracts.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from dnadesign.cruncher.config.schema_v3 import StrictBaseModel
from dnadesign.cruncher.yiu.models.common import (
    PublishedAssemblySpace,
    SequenceMode,
    TopologyKind,
    ValidationMode,
    WorkflowScope,
    _validate_slug,
)


class YiuValidationIssue(StrictBaseModel):
    code: str
    message: str
    step_id: str | None = None
    state_id: str | None = None
    severity: Literal["error", "warning"] = "error"


class ProjectedRegionPart(StrictBaseModel):
    segment_id: str
    start: int = Field(ge=0)
    end: int = Field(gt=0)

    @field_validator("segment_id")
    @classmethod
    def _validate_segment_id(cls, value: str) -> str:
        return _validate_slug(value, label="projected_region_part.segment_id")

    @model_validator(mode="after")
    def _validate_bounds(self) -> "ProjectedRegionPart":
        if self.end <= self.start:
            raise ValueError("projected_region_part.end must be > projected_region_part.start")
        return self


class ProjectedRegion(StrictBaseModel):
    id: str
    source_region_id: str
    state_id: str
    spans_junction: bool = False
    projection_kind: Literal["atomic", "compound"] = "atomic"
    assembled_coordinate_space: PublishedAssemblySpace | None = None
    parts: list[ProjectedRegionPart] = Field(default_factory=list)

    @field_validator("id", "source_region_id", "state_id")
    @classmethod
    def _validate_id_like_fields(cls, value: str, info) -> str:
        return _validate_slug(value, label=str(info.field_name))

    @model_validator(mode="after")
    def _validate_parts(self) -> "ProjectedRegion":
        if not self.parts:
            raise ValueError("projected_region.parts must be non-empty")
        return self


class YiuPatternEvidenceSummary(StrictBaseModel):
    guaranteed_checks: int = 0
    possible_checks: int = 0
    impossible_checks: int = 0


class YiuStateRecord(StrictBaseModel):
    state_id: str
    step_id: str
    kind: str
    status: Literal["satisfied", "unsatisfied"]
    sequence_mode: SequenceMode = "concrete"
    validation_mode: ValidationMode = "concrete_realization"
    view_contract_version: int | None = None
    state_kind: str | None = None
    topology_kind: TopologyKind | None = None
    primary_sequence: str | None = None
    complement_sequence: str | None = None
    segments: list[dict[str, Any]] = Field(default_factory=list)
    annotations: list[dict[str, Any]] = Field(default_factory=list)
    cuts: list[dict[str, Any]] = Field(default_factory=list)
    junctions: list[dict[str, Any]] = Field(default_factory=list)
    fragments: list[dict[str, Any]] = Field(default_factory=list)
    pattern_evidence_summary: YiuPatternEvidenceSummary = Field(default_factory=YiuPatternEvidenceSummary)
    metadata: dict[str, Any] = Field(default_factory=dict)


class YiuReportMetadata(StrictBaseModel):
    spec_schema_version: int
    step_count: int
    state_count: int
    emitted_view_count: int = 0
    view_contract_version: int | None = None
    catalog_paths: list[str] = Field(default_factory=list)


class YiuValidationReport(StrictBaseModel):
    workflow: Literal["yiu"] = "yiu"
    family: Literal["yiu"] = "yiu"
    protocol: str = "yiu_v1"
    protocol_template: str | None = None
    template_alias_used: str | None = None
    template_alias_status: Literal["deprecated_alias"] | None = None
    workflow_scope: WorkflowScope | None = None
    spec_name: str
    status: Literal["satisfied", "unsatisfied"]
    sequence_mode: SequenceMode = "concrete"
    validation_mode: ValidationMode = "concrete_realization"
    run_dir: str | None = None
    metadata: YiuReportMetadata
    states: list[YiuStateRecord]
    issues: list[YiuValidationIssue] = Field(default_factory=list)
