"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/infer/src/features/contracts.py

Semantic contracts for Evo2 sequence-view feature extraction.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

FEATURE_SCHEMA_VERSION = "evo2_sequence_feature_v1"


class _StrictFeatureModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SequenceFeatureContextConfig(_StrictFeatureModel):
    kind: Literal["anchor_only", "template_1kb", "template_custom"] = "anchor_only"
    template_id: Optional[str] = None

    @model_validator(mode="after")
    def _validate_context(self) -> "SequenceFeatureContextConfig":
        if self.kind == "anchor_only" and self.template_id is not None:
            raise ValueError("feature_bundle.context.template_id is not allowed for kind='anchor_only'.")
        if self.kind == "template_custom" and not str(self.template_id or "").strip():
            raise ValueError("feature_bundle.context.template_id is required for kind='template_custom'.")
        return self


class SequenceFeaturePoolingConfig(_StrictFeatureModel):
    seq_mean: bool = True
    anchor_mean_for_templated: bool = True


class SequenceViewSelectorConfig(_StrictFeatureModel):
    product_kind: Optional[
        Literal[
            "source_record",
            "selected_region",
            "construct_insert",
            "analysis_window",
            "realized_context",
        ]
    ] = None
    view_name: Optional[str] = None
    alias: Optional[str] = None
    orientation: Optional[Literal["forward", "reverse_complement", "unknown"]] = None

    @model_validator(mode="after")
    def _require_selector_field(self) -> "SequenceViewSelectorConfig":
        if not any((self.product_kind, self.view_name, self.alias, self.orientation)):
            raise ValueError(
                "feature_bundle.sequence_view_inputs[].view_selector must set at least one selector field."
            )
        return self


class SequenceViewPoolingConfig(_StrictFeatureModel):
    operation: Literal["seq_mean", "anchor_mean", "core60_mean"]
    bounds_from: Optional[Literal["sequence_view", "construct_overlay"]] = None
    start_0: Optional[int] = Field(default=None, ge=0)
    end_0: Optional[int] = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _validate_bounds_source(self) -> "SequenceViewPoolingConfig":
        explicit_bounds = self.start_0 is not None or self.end_0 is not None
        if explicit_bounds and (self.start_0 is None or self.end_0 is None):
            raise ValueError("feature_bundle.sequence_view_inputs[].pooling explicit bounds require start_0 and end_0.")
        if self.start_0 is not None and self.end_0 is not None and self.end_0 <= self.start_0:
            raise ValueError("feature_bundle.sequence_view_inputs[].pooling.end_0 must be greater than start_0.")
        if self.operation == "anchor_mean" and self.bounds_from is None:
            raise ValueError("feature_bundle.sequence_view_inputs[].pooling.bounds_from is required for anchor_mean.")
        if self.operation != "anchor_mean" and self.bounds_from is not None:
            raise ValueError("feature_bundle.sequence_view_inputs[].pooling.bounds_from is only valid for anchor_mean.")
        if explicit_bounds and self.operation != "core60_mean":
            raise ValueError(
                "feature_bundle.sequence_view_inputs[].pooling explicit bounds are only valid for core60_mean."
            )
        if self.operation == "core60_mean" and explicit_bounds and self.end_0 - self.start_0 != 60:
            raise ValueError(
                "feature_bundle.sequence_view_inputs[].pooling core60_mean explicit bounds must span 60 bp."
            )
        return self


class SequenceViewInputConfig(_StrictFeatureModel):
    dataset: str
    root: Optional[str] = None
    view_selector: SequenceViewSelectorConfig
    pooling: SequenceViewPoolingConfig


class FeatureDeduplicateConfig(_StrictFeatureModel):
    by_forward_pass_key: bool = True
    by_feature_vector_key: bool = True
    write_alias_map: bool = True


class SequenceFeatureDebugConfig(_StrictFeatureModel):
    persist_tokenwise: bool = False

    @model_validator(mode="after")
    def _reject_unimplemented_debug_mode(self) -> "SequenceFeatureDebugConfig":
        if self.persist_tokenwise:
            raise ValueError("feature_bundle.debug.persist_tokenwise is not supported in the repo-aligned v1 contract.")
        return self


class SequenceFeatureBundleConfig(_StrictFeatureModel):
    kind: Literal["evo2_sequence_feature_v1"] = "evo2_sequence_feature_v1"
    intermediate_block: int = Field(default=26, ge=0)
    collect_log_likelihood: bool = True
    collect_output_layer_mean: bool = True
    collect_output_embedding: Optional[bool] = None
    collect_intermediate_embedding: bool = True
    context: SequenceFeatureContextConfig = Field(default_factory=SequenceFeatureContextConfig)
    pooling: SequenceFeaturePoolingConfig = Field(default_factory=SequenceFeaturePoolingConfig)
    sequence_view_inputs: list[SequenceViewInputConfig] = Field(default_factory=list)
    deduplicate: FeatureDeduplicateConfig = Field(default_factory=FeatureDeduplicateConfig)
    debug: SequenceFeatureDebugConfig = Field(default_factory=SequenceFeatureDebugConfig)
    feature_schema_version: str = FEATURE_SCHEMA_VERSION

    @model_validator(mode="after")
    def _normalize_aliases_and_contract(self) -> "SequenceFeatureBundleConfig":
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
        if self.sequence_view_inputs and self.pooling != SequenceFeaturePoolingConfig():
            raise ValueError(
                "feature_bundle.pooling is a context-scoped contract and is not used with "
                "feature_bundle.sequence_view_inputs."
            )
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
    view_id: str | None = None
    view_name: str | None = None
    product_kind: str | None = None
    orientation: str | None = None
    parent_sequence_id: str | None = None
    derivation_id: str | None = None
    source_dataset_id: str | None = None
    source_dataset_root: str | None = None
    pooling_operation: str | None = None
    pooling_start_0: int | None = None
    pooling_end_0: int | None = None


@dataclass(frozen=True)
class OpalMatrixExport:
    x: list[list[float]]
    feature_names: list[str]
    row_ids: list[str]
