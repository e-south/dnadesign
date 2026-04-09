"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/infer/src/features/contracts.py

Semantic contracts for Evo2 promoter-feature extraction.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

FEATURE_SCHEMA_VERSION = "evo2_promoter_v1"


class _StrictFeatureModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PromoterContextConfig(_StrictFeatureModel):
    kind: Literal["anchor_only", "template_1kb", "template_custom"] = "anchor_only"
    template_id: Optional[str] = None

    @model_validator(mode="after")
    def _validate_context(self) -> "PromoterContextConfig":
        if self.kind == "anchor_only" and self.template_id is not None:
            raise ValueError("feature_bundle.context.template_id is not allowed for kind='anchor_only'.")
        if self.kind == "template_custom" and not str(self.template_id or "").strip():
            raise ValueError("feature_bundle.context.template_id is required for kind='template_custom'.")
        return self


class PromoterPoolingConfig(_StrictFeatureModel):
    seq_mean: bool = True
    anchor_mean_for_templated: bool = True


class PromoterDebugConfig(_StrictFeatureModel):
    persist_tokenwise: bool = False

    @model_validator(mode="after")
    def _reject_unimplemented_debug_mode(self) -> "PromoterDebugConfig":
        if self.persist_tokenwise:
            raise ValueError("feature_bundle.debug.persist_tokenwise is not supported in the repo-aligned v1 contract.")
        return self


class PromoterFeatureBundleConfig(_StrictFeatureModel):
    kind: Literal["evo2_promoter_v1"] = "evo2_promoter_v1"
    intermediate_block: int = Field(default=26, ge=0)
    collect_log_likelihood: bool = True
    collect_output_layer_mean: bool = True
    collect_output_embedding: Optional[bool] = None
    collect_intermediate_embedding: bool = True
    context: PromoterContextConfig = Field(default_factory=PromoterContextConfig)
    pooling: PromoterPoolingConfig = Field(default_factory=PromoterPoolingConfig)
    debug: PromoterDebugConfig = Field(default_factory=PromoterDebugConfig)
    feature_schema_version: str = FEATURE_SCHEMA_VERSION

    @model_validator(mode="after")
    def _normalize_aliases_and_contract(self) -> "PromoterFeatureBundleConfig":
        if self.collect_output_embedding is not None:
            if self.collect_output_layer_mean not in {self.collect_output_embedding, True}:
                raise ValueError(
                    "feature_bundle.collect_output_embedding is an alias for "
                    "feature_bundle.collect_output_layer_mean; keep them equal."
                )
            self.collect_output_layer_mean = bool(self.collect_output_embedding)

        if not any(
            (
                self.collect_log_likelihood,
                self.collect_output_layer_mean,
                self.collect_intermediate_embedding,
            )
        ):
            raise ValueError("feature_bundle must enable at least one feature group.")

        if not str(self.feature_schema_version or "").strip():
            raise ValueError("feature_bundle.feature_schema_version must be non-empty.")
        return self


@dataclass(frozen=True)
class SelectorResolution:
    intermediate_block: int
    intermediate_selector: str
    provider_layer: str


@dataclass(frozen=True)
class SequenceContextRecord:
    sequence_id: str
    anchor_id: str
    context_id: str
    context_kind: str
    template_id: str | None
    resolved_sequence: str
    resolved_length: int
    anchor_start: int
    anchor_end: int
    anchor_orientation: str | None
    construct_version: str | None
    is_wildtype: bool | None


@dataclass(frozen=True)
class OpalMatrixExport:
    x: list[list[float]]
    feature_names: list[str]
    row_ids: list[str]
