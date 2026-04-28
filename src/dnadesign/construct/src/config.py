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
from typing import Annotated, List, Literal, Optional

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic import ValidationError as PydanticValidationError

from .errors import ConfigError


class StrictConfigModel(BaseModel):
    model_config = {"extra": "forbid"}


class USRDatasetLocatorConfig(StrictConfigModel):
    kind: Literal["usr"]
    dataset: str
    root: Optional[str] = None

    @field_validator("dataset")
    @classmethod
    def _dataset_not_blank(cls, value: str) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError("USR dataset locator dataset cannot be empty.")
        return text


class InputConfig(StrictConfigModel):
    source: USRDatasetLocatorConfig
    field: str = "sequence"
    ids: Optional[List[str]] = None

    @model_validator(mode="before")
    @classmethod
    def _reject_legacy_shape(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        legacy_fields = [field for field in ("dataset", "root") if field in data]
        if isinstance(data.get("source"), str):
            legacy_fields.append("source")
        if legacy_fields:
            joined = ", ".join(f"input.{field}" for field in legacy_fields)
            raise ValueError(
                f"{joined} is no longer supported. Use input.source.kind, "
                "input.source.dataset, and input.source.root instead."
            )
        return data

    @field_validator("field")
    @classmethod
    def _field_not_blank(cls, value: str) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError("input.field cannot be empty.")
        return text


class TemplateLiteralSourceConfig(StrictConfigModel):
    kind: Literal["literal"]
    sequence: str
    label: Optional[str] = None

    @field_validator("sequence")
    @classmethod
    def _sequence_not_blank(cls, value: str) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError("template.source.sequence cannot be empty when kind='literal'.")
        return text

    @field_validator("label")
    @classmethod
    def _label_not_blank(cls, value: str | None) -> str | None:
        if value is None:
            return None
        text = str(value or "").strip()
        if not text:
            raise ValueError("template.source.label cannot be empty when provided.")
        return text


class TemplatePathSourceConfig(StrictConfigModel):
    kind: Literal["path"]
    path: str
    label: Optional[str] = None

    @field_validator("path")
    @classmethod
    def _path_not_blank(cls, value: str) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError("template.source.path cannot be empty when kind='path'.")
        return text

    @field_validator("label")
    @classmethod
    def _label_not_blank(cls, value: str | None) -> str | None:
        if value is None:
            return None
        text = str(value or "").strip()
        if not text:
            raise ValueError("template.source.label cannot be empty when provided.")
        return text


class TemplateUSRSourceConfig(StrictConfigModel):
    kind: Literal["usr"]
    dataset: str
    root: Optional[str] = None
    record_id: str
    field: str = "sequence"
    label: Optional[str] = None

    @field_validator("dataset", "record_id", "field")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError("template.source.dataset, record_id, and field cannot be empty.")
        return text

    @field_validator("label")
    @classmethod
    def _label_not_blank(cls, value: str | None) -> str | None:
        if value is None:
            return None
        text = str(value or "").strip()
        if not text:
            raise ValueError("template.source.label cannot be empty when provided.")
        return text


TemplateSourceConfig = Annotated[
    TemplateLiteralSourceConfig | TemplatePathSourceConfig | TemplateUSRSourceConfig,
    Field(discriminator="kind"),
]


class TemplateConfig(StrictConfigModel):
    id: str
    source: TemplateSourceConfig
    circular: bool = False

    @model_validator(mode="before")
    @classmethod
    def _reject_legacy_shape(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        legacy_fields = [
            field for field in ("kind", "sequence", "path", "dataset", "root", "record_id", "field") if field in data
        ]
        if isinstance(data.get("source"), str):
            legacy_fields.append("source")
        if legacy_fields:
            joined = ", ".join(f"template.{field}" for field in legacy_fields)
            raise ValueError(
                f"{joined} is no longer supported. Move template locator fields under template.source.* "
                "and keep any human-readable provenance text in template.source.label."
            )
        return data

    @field_validator("id")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError("template.id cannot be empty.")
        return text


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


class CoordinatePlacementLocatorConfig(StrictConfigModel):
    kind: Literal["coordinates"]
    start: int = Field(ge=0)
    end: int = Field(ge=0)


class FlankPlacementLocatorConfig(StrictConfigModel):
    kind: Literal["flanks"]
    upstream_sequence: str
    downstream_sequence: str

    @field_validator("upstream_sequence", "downstream_sequence")
    @classmethod
    def _sequence_not_blank(cls, value: str) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError("placement.locator flank sequences cannot be empty.")
        return text


PlacementLocatorConfig = Annotated[
    CoordinatePlacementLocatorConfig | FlankPlacementLocatorConfig,
    Field(discriminator="kind"),
]


class PlacementGuardsConfig(StrictConfigModel):
    replaced_sequence: Optional[str] = None
    upstream_sequence: Optional[str] = None
    downstream_sequence: Optional[str] = None
    replaced_span_bp: Optional[int] = Field(default=None, ge=0)
    require_unique_forward_matches: bool = False

    @field_validator("replaced_sequence", "upstream_sequence", "downstream_sequence")
    @classmethod
    def _optional_sequence_not_blank(cls, value: str | None) -> str | None:
        if value is None:
            return None
        text = str(value or "").strip()
        if not text:
            raise ValueError("placement.guards sequences cannot be empty when provided.")
        return text

    @model_validator(mode="after")
    def _validate_meaningful_guard(self) -> "PlacementGuardsConfig":
        has_guard = any(
            value is not None
            for value in (
                self.replaced_sequence,
                self.upstream_sequence,
                self.downstream_sequence,
                self.replaced_span_bp,
            )
        )
        if self.require_unique_forward_matches and not any(
            value is not None for value in (self.replaced_sequence, self.upstream_sequence, self.downstream_sequence)
        ):
            raise ValueError("placement.guards.require_unique_forward_matches requires at least one guard sequence.")
        if not has_guard and not self.require_unique_forward_matches:
            raise ValueError("placement.guards must declare at least one guard.")
        return self


class PlacementConfig(StrictConfigModel):
    kind: Literal["insert", "replace"]
    orientation: Literal["forward", "reverse_complement"] = "forward"
    locator: PlacementLocatorConfig
    guards: Optional[PlacementGuardsConfig] = None

    @model_validator(mode="after")
    def _validate_shape(self) -> "PlacementConfig":
        if isinstance(self.locator, CoordinatePlacementLocatorConfig):
            if self.locator.end < self.locator.start:
                raise ValueError("placement.locator.end must be >= placement.locator.start.")
            if self.kind == "insert" and self.locator.end != self.locator.start:
                raise ValueError(
                    "insert placement requires placement.locator.end == placement.locator.start "
                    "when locator.kind='coordinates'."
                )
            if self.kind == "replace" and self.locator.end == self.locator.start:
                raise ValueError(
                    "replace placement requires placement.locator.end > placement.locator.start "
                    "when locator.kind='coordinates'."
                )
        if self.kind == "insert" and self.guards is not None and self.guards.replaced_sequence is not None:
            raise ValueError("placement.guards.replaced_sequence is only allowed when kind='replace'.")
        if isinstance(self.locator, FlankPlacementLocatorConfig) and self.guards is not None:
            if self.guards.upstream_sequence is not None or self.guards.downstream_sequence is not None:
                raise ValueError(
                    "placement.guards.upstream_sequence/downstream_sequence are not allowed when "
                    "locator.kind='flanks'. Use placement.locator.upstream_sequence/downstream_sequence instead."
                )
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


class WindowConfig(StrictConfigModel):
    semantics: Literal["fixed_total", "anchor_plus_context"] = "fixed_total"
    reference: Literal["start", "center", "end"] = "center"
    direction: Literal["symmetric", "five_prime", "three_prime"] = "symmetric"
    size_bp: Optional[int] = Field(default=None, ge=1)
    upstream_bp: Optional[int] = Field(default=None, ge=0)
    downstream_bp: Optional[int] = Field(default=None, ge=0)
    offset_bp: int = 0

    @model_validator(mode="after")
    def _validate_shape(self) -> "WindowConfig":
        if self.semantics == "fixed_total":
            if self.size_bp is None:
                raise ValueError("realize.window.size_bp is required when realize.window.semantics='fixed_total'.")
            if self.upstream_bp is not None or self.downstream_bp is not None:
                raise ValueError(
                    "realize.window.upstream_bp/downstream_bp are only allowed when "
                    "realize.window.semantics='anchor_plus_context'."
                )
            return self

        if self.size_bp is not None:
            raise ValueError("realize.window.size_bp is only allowed when realize.window.semantics='fixed_total'.")
        if self.upstream_bp is None or self.downstream_bp is None:
            raise ValueError(
                "realize.window.upstream_bp and realize.window.downstream_bp are required when "
                "realize.window.semantics='anchor_plus_context'."
            )
        if self.direction != "symmetric":
            raise ValueError(
                "realize.window.direction must stay 'symmetric' when semantics='anchor_plus_context'. "
                "Use upstream_bp/downstream_bp to express asymmetric flanks."
            )
        if self.reference != "center":
            raise ValueError(
                "realize.window.reference must stay 'center' when semantics='anchor_plus_context'. "
                "The extracted span is defined by the focal part plus explicit upstream/downstream flanks."
            )
        if self.offset_bp != 0:
            raise ValueError("realize.window.offset_bp is only supported when semantics='fixed_total'.")
        return self


class RealizeConfig(StrictConfigModel):
    mode: Literal["window", "full_construct"] = "window"
    focal_part: Optional[str] = None
    window: Optional[WindowConfig] = None

    @model_validator(mode="before")
    @classmethod
    def _reject_legacy_window_fields(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        legacy_fields = [field for field in ("focal_point", "anchor_offset_bp", "window_bp") if field in data]
        if legacy_fields:
            joined = ", ".join(f"realize.{field}" for field in legacy_fields)
            raise ValueError(
                f"{joined} is no longer supported. Use realize.window.reference, "
                "realize.window.offset_bp, and realize.window.size_bp instead."
            )
        if data.get("mode", "window") == "window" and "window" in data and data["window"] is None:
            raise ValueError("realize.window must be a mapping when realize.mode='window'.")
        return data

    @model_validator(mode="after")
    def _validate_mode(self) -> "RealizeConfig":
        if self.mode == "window":
            if not str(self.focal_part or "").strip():
                raise ValueError("realize.focal_part is required when realize.mode='window'.")
            if self.window is None:
                raise ValueError("realize.window is required when realize.mode='window'.")
            return self
        if self.window is not None:
            raise ValueError("realize.window is only allowed when realize.mode='window'.")
        return self


class FeatureMatchConfig(StrictConfigModel):
    role_hint: Optional[str] = None
    labels: List[str] = Field(default_factory=list)

    @field_validator("role_hint")
    @classmethod
    def _role_hint_not_blank(cls, value: str | None) -> str | None:
        if value is None:
            return None
        text = str(value or "").strip()
        if not text:
            raise ValueError("feature matcher role_hint cannot be empty when provided.")
        return text

    @field_validator("labels")
    @classmethod
    def _labels_not_blank(cls, value: List[str]) -> List[str]:
        normalized: list[str] = []
        for item in value:
            text = str(item or "").strip()
            if not text:
                raise ValueError("feature matcher labels cannot contain empty strings.")
            normalized.append(text)
        return normalized

    @model_validator(mode="after")
    def _validate_meaningful_matcher(self) -> "FeatureMatchConfig":
        if self.role_hint is None and not self.labels:
            raise ValueError("feature matcher requires role_hint and/or labels.")
        return self


class AnnotationPairMidpointSelectorConfig(StrictConfigModel):
    kind: Literal["annotation_pair_midpoint"]
    first: FeatureMatchConfig
    second: FeatureMatchConfig
    confidence: Literal["high", "medium", "low"] = "high"


class AnnotationFeatureCenterSelectorConfig(StrictConfigModel):
    kind: Literal["annotation_feature_center"]
    role_hint: Optional[str] = None
    labels: List[str] = Field(default_factory=list)
    confidence: Literal["high", "medium", "low"] = "medium"

    @model_validator(mode="after")
    def _validate_meaningful_matcher(self) -> "AnnotationFeatureCenterSelectorConfig":
        if self.role_hint is None and not self.labels:
            raise ValueError("annotation_feature_center requires role_hint and/or labels.")
        return self


class SequenceMidpointSelectorConfig(StrictConfigModel):
    kind: Literal["sequence_midpoint"]
    allowed: bool = False
    confidence: Literal["high", "medium", "low"] = "low"


NormalizeAnchorSelectorConfig = Annotated[
    AnnotationPairMidpointSelectorConfig | AnnotationFeatureCenterSelectorConfig | SequenceMidpointSelectorConfig,
    Field(discriminator="kind"),
]


class SelectorChainConfig(StrictConfigModel):
    kind: Literal["chain"]
    selectors: List[NormalizeAnchorSelectorConfig]

    @model_validator(mode="after")
    def _validate_non_empty(self) -> "SelectorChainConfig":
        if not self.selectors:
            raise ValueError("normalize_anchor.focal_selector.selectors must define at least one selector.")
        return self


class OverLengthTrimPolicyConfig(StrictConfigModel):
    kind: Literal["trim"] = "trim"
    target_length: int = Field(ge=1)
    require_focal_inside: bool = True


class NormalizeTemplateConfig(StrictConfigModel):
    source: TemplateSourceConfig
    circular: bool = False
    id: Optional[str] = None

    @field_validator("id")
    @classmethod
    def _id_not_blank(cls, value: str | None) -> str | None:
        if value is None:
            return None
        text = str(value or "").strip()
        if not text:
            raise ValueError("normalize_anchor template.id cannot be empty when provided.")
        return text


class UnderLengthExpandFromTemplatePolicyConfig(StrictConfigModel):
    kind: Literal["expand_from_template"]
    target_length: int = Field(ge=1)
    template: NormalizeTemplateConfig
    placement_ref: str

    @field_validator("placement_ref")
    @classmethod
    def _placement_ref_not_blank(cls, value: str) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError("normalize_anchor under_length_policy.placement_ref cannot be empty.")
        return text


class FeatureRetentionPolicyConfig(StrictConfigModel):
    fail_if_loses_roles: List[str] = Field(default_factory=list)
    warn_if_clips_roles: List[str] = Field(default_factory=list)

    @field_validator("fail_if_loses_roles", "warn_if_clips_roles")
    @classmethod
    def _roles_not_blank(cls, value: List[str]) -> List[str]:
        normalized: list[str] = []
        for item in value:
            text = str(item or "").strip()
            if not text:
                raise ValueError("feature retention roles cannot contain empty strings.")
            normalized.append(text)
        return normalized


class FallbackPolicyConfig(StrictConfigModel):
    allow_low_confidence: bool = False
    mark_low_confidence_rows: bool = False


class OutputSequenceViewConfig(StrictConfigModel):
    create: bool = False
    recommended_pooling: Optional[Literal["seq_mean", "anchor_mean", "core60_mean"]] = None


class NormalizeAnchorConfig(StrictConfigModel):
    product_kind: Literal["analysis_window"]
    target_length: int = Field(ge=1)
    focal_selector: SelectorChainConfig
    over_length_policy: OverLengthTrimPolicyConfig
    under_length_policy: Optional[UnderLengthExpandFromTemplatePolicyConfig] = None
    feature_retention_policy: FeatureRetentionPolicyConfig = Field(default_factory=FeatureRetentionPolicyConfig)
    fallback_policy: FallbackPolicyConfig = Field(default_factory=FallbackPolicyConfig)
    emit_feature_retention_report: bool = False
    output_sequence_view: OutputSequenceViewConfig = Field(default_factory=OutputSequenceViewConfig)


class OutputVariantConfig(StrictConfigModel):
    product_kind: Literal["realized_context"]
    orientation: Literal["forward", "reverse_complement"]
    recommended_pooling: Optional[Literal["seq_mean", "anchor_mean", "core60_mean"]] = None

    @model_validator(mode="after")
    def _validate_product_kind_orientation(self) -> "OutputVariantConfig":
        if self.product_kind != "realized_context":
            raise ValueError("output_variants product_kind must be 'realized_context'.")
        return self


class OutputConfig(StrictConfigModel):
    target: USRDatasetLocatorConfig
    record_source: Optional[str] = None
    on_conflict: Literal["error", "ignore"] = "error"
    allow_same_as_input: bool = False

    @model_validator(mode="before")
    @classmethod
    def _reject_legacy_shape(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        legacy_fields = [field for field in ("dataset", "root", "source") if field in data]
        if legacy_fields:
            joined = ", ".join(f"output.{field}" for field in legacy_fields)
            raise ValueError(
                f"{joined} is no longer supported. Use output.target.kind, output.target.dataset, "
                "output.target.root, and output.record_source instead."
            )
        return data

    @field_validator("record_source")
    @classmethod
    def _record_source_not_blank(cls, value: str | None) -> str | None:
        if value is None:
            return None
        text = str(value or "").strip()
        if not text:
            raise ValueError("output.record_source cannot be empty when provided.")
        return text


class InnerJobConfig(StrictConfigModel):
    id: str
    mode: Literal["realize_template", "normalize_anchor"] = "realize_template"
    input: InputConfig
    template: Optional[TemplateConfig] = None
    parts: List[PartConfig] = Field(default_factory=list)
    realize: Optional[RealizeConfig] = None
    normalize_anchor: Optional[NormalizeAnchorConfig] = None
    output_variants: List[OutputVariantConfig] = Field(default_factory=list)
    output: OutputConfig

    @model_validator(mode="after")
    def _validate_parts(self) -> "InnerJobConfig":
        if not str(self.input.source.root or "").strip():
            raise ValueError("job.input.source.root is required for construct jobs that read USR datasets.")
        if self.mode == "normalize_anchor":
            if self.normalize_anchor is None:
                raise ValueError("job.normalize_anchor is required when job.mode='normalize_anchor'.")
            if self.template is not None:
                raise ValueError("job.template is only allowed when job.mode='realize_template'.")
            if self.parts:
                raise ValueError("job.parts is only allowed when job.mode='realize_template'.")
            if self.realize is not None:
                raise ValueError("job.realize is only allowed when job.mode='realize_template'.")
            if self.output_variants:
                raise ValueError("job.output_variants is only allowed when job.mode='realize_template'.")
            return self
        if self.template is None:
            raise ValueError("job.template is required when job.mode='realize_template'.")
        if not self.parts:
            raise ValueError("job.parts must define at least one part.")
        if self.realize is None:
            raise ValueError("job.realize is required when job.mode='realize_template'.")
        if self.normalize_anchor is not None:
            raise ValueError("job.normalize_anchor is only allowed when job.mode='normalize_anchor'.")
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
