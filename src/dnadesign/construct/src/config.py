"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/construct/src/config.py

Configuration schema and YAML loading for construct.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Literal, Optional

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic import ValidationError as PydanticValidationError

from .errors import ConfigError


class StrictConfigModel(BaseModel):
    model_config = {"extra": "forbid"}


class InputConfig(StrictConfigModel):
    source: Literal["usr"]
    dataset: str
    root: Optional[str] = None
    field: str = "sequence"
    ids: Optional[List[str]] = None

    @field_validator("field")
    @classmethod
    def _field_not_blank(cls, value: str) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError("input.field cannot be empty.")
        return text


class TemplateConfig(StrictConfigModel):
    id: str
    kind: Optional[Literal["literal", "path", "usr"]] = None
    sequence: Optional[str] = None
    path: Optional[str] = None
    dataset: Optional[str] = None
    root: Optional[str] = None
    record_id: Optional[str] = None
    field: str = "sequence"
    circular: bool = False
    source: Optional[str] = None

    @field_validator("id", "field")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError("template.id and template.field cannot be empty.")
        return text

    @model_validator(mode="after")
    def _validate_source(self) -> "TemplateConfig":
        has_sequence = self.sequence is not None
        has_path = self.path is not None
        has_usr = bool(str(self.dataset or "").strip()) or bool(str(self.record_id or "").strip())

        if self.kind is None:
            selected = [has_sequence, has_path, has_usr]
            if sum(1 for item in selected if item) != 1:
                raise ValueError(
                    "template must define exactly one source: template.sequence, template.path, "
                    "or template.dataset + template.record_id."
                )
            if has_sequence:
                self.kind = "literal"
            elif has_path:
                self.kind = "path"
            else:
                self.kind = "usr"

        if self.kind == "literal":
            if not has_sequence or has_path or has_usr:
                raise ValueError(
                    "template.kind='literal' requires template.sequence and disallows template.path/template.dataset."
                )
        elif self.kind == "path":
            if not has_path or has_sequence or has_usr:
                raise ValueError(
                    "template.kind='path' requires template.path and disallows template.sequence/template.dataset."
                )
        elif self.kind == "usr":
            if has_sequence or has_path:
                raise ValueError("template.kind='usr' disallows template.sequence and template.path.")
            if not str(self.dataset or "").strip():
                raise ValueError("template.dataset is required when template.kind='usr'.")
            if not str(self.record_id or "").strip():
                raise ValueError("template.record_id is required when template.kind='usr'.")
        else:
            raise ValueError("template.kind must be one of: literal, path, usr.")
        return self


class PartSequenceConfig(StrictConfigModel):
    source: Literal["input_field", "literal"]
    field: Optional[str] = None
    literal: Optional[str] = None

    @model_validator(mode="after")
    def _validate_shape(self) -> "PartSequenceConfig":
        if self.source == "input_field":
            if not str(self.field or "").strip():
                raise ValueError("part.sequence.field is required when source='input_field'.")
            if self.literal is not None:
                raise ValueError("part.sequence.literal is not allowed when source='input_field'.")
        if self.source == "literal":
            if not str(self.literal or "").strip():
                raise ValueError("part.sequence.literal is required when source='literal'.")
            if self.field is not None:
                raise ValueError("part.sequence.field is not allowed when source='literal'.")
        return self


class PlacementConfig(StrictConfigModel):
    kind: Literal["insert", "replace"]
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    orientation: Literal["forward", "reverse_complement"] = "forward"
    expected_template_sequence: Optional[str] = None

    @model_validator(mode="after")
    def _validate_bounds(self) -> "PlacementConfig":
        if self.end < self.start:
            raise ValueError("placement.end must be >= placement.start.")
        if self.kind == "insert" and self.end != self.start:
            raise ValueError("insert placement requires placement.end == placement.start.")
        if self.kind == "replace" and self.end == self.start:
            raise ValueError("replace placement requires placement.end > placement.start.")
        if self.kind == "insert" and self.expected_template_sequence is not None:
            raise ValueError("placement.expected_template_sequence is only allowed when kind='replace'.")
        return self


class PartConfig(StrictConfigModel):
    name: str
    role: str = "part"
    sequence: PartSequenceConfig
    placement: PlacementConfig

    @field_validator("name", "role")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError("part name/role cannot be empty.")
        return text


class RealizeConfig(StrictConfigModel):
    mode: Literal["window", "full_construct"] = "window"
    focal_part: Optional[str] = None
    focal_point: Literal["start", "center", "end"] = "center"
    anchor_offset_bp: int = 0
    window_bp: Optional[int] = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _validate_mode(self) -> "RealizeConfig":
        if self.mode == "window":
            if not str(self.focal_part or "").strip():
                raise ValueError("realize.focal_part is required when realize.mode='window'.")
            if self.window_bp is None:
                raise ValueError("realize.window_bp is required when realize.mode='window'.")
        else:
            if self.window_bp is not None:
                raise ValueError("realize.window_bp is only allowed when realize.mode='window'.")
        return self


class OutputConfig(StrictConfigModel):
    dataset: str
    root: Optional[str] = None
    source: Optional[str] = None
    on_conflict: Literal["error", "ignore"] = "error"
    allow_same_as_input: bool = False

    @field_validator("dataset")
    @classmethod
    def _dataset_not_blank(cls, value: str) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError("output.dataset cannot be empty.")
        return text


class InnerJobConfig(StrictConfigModel):
    id: str
    input: InputConfig
    template: TemplateConfig
    parts: List[PartConfig]
    realize: RealizeConfig
    output: OutputConfig

    @model_validator(mode="after")
    def _validate_parts(self) -> "InnerJobConfig":
        if not self.parts:
            raise ValueError("job.parts must define at least one part.")
        seen: set[str] = set()
        input_driven = 0
        for part in self.parts:
            if part.name in seen:
                raise ValueError(f"Duplicate part name '{part.name}'.")
            seen.add(part.name)
            if part.sequence.source == "input_field":
                input_driven += 1
        if input_driven < 1:
            raise ValueError("job.parts must include at least one source='input_field' part.")
        if self.realize.mode == "window" and self.realize.focal_part not in seen:
            raise ValueError(f"realize.focal_part '{self.realize.focal_part}' is not defined in job.parts.")
        return self


class JobConfig(StrictConfigModel):
    job: InnerJobConfig


def load_job_config(path: str | Path) -> tuple[JobConfig, Path]:
    config_path = Path(path).expanduser().resolve()
    if not config_path.exists():
        raise ConfigError(f"Config not found: {config_path}")
    try:
        data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in config: {config_path}") from exc
    try:
        return JobConfig.model_validate(data), config_path
    except PydanticValidationError as exc:
        raise ConfigError(f"Invalid config {config_path}: {exc}") from exc
