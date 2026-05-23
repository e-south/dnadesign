"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/construct/src/contracts/normalize_anchor.py

Normalize-anchor selector, retention, and policy contracts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Annotated, List, Literal, Optional

from pydantic import Field, field_validator, model_validator

from .base import StrictConfigModel
from .templates import TemplateSourceConfig


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


class SequenceOffsetSelectorConfig(StrictConfigModel):
    kind: Literal["sequence_offset"]
    offset_0: int = Field(ge=0)
    label: Optional[str] = None
    confidence: Literal["high", "medium", "low"] = "high"

    @field_validator("label")
    @classmethod
    def _label_not_blank(cls, value: str | None) -> str | None:
        if value is None:
            return None
        text = str(value or "").strip()
        if not text:
            raise ValueError("sequence_offset label cannot be empty when provided.")
        return text


NormalizeAnchorSelectorConfig = Annotated[
    AnnotationPairMidpointSelectorConfig
    | AnnotationFeatureCenterSelectorConfig
    | SequenceMidpointSelectorConfig
    | SequenceOffsetSelectorConfig,
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
    window_anchor: Literal["retention_optimized", "upstream_of_focal"] = "retention_optimized"


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
