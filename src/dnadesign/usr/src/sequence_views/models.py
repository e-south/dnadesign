"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/src/sequence_views/models.py

Sequence-view contracts for semantically distinct products that may share one.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

VIEW_ID_SCHEMA_VERSION = 1
SEQUENCE_VIEW_SIDECAR_RELATIVE_PATH = "_views/sequence_views.parquet"
VIEW_SEMANTICS_SIDECAR_RELATIVE_PATH = "_views/view_semantics.parquet"

# Product kind describes generic sequence-product lineage, not domain role,
# cohort membership, orientation, length, or pooling. Study-specific terms live
# in view-semantics addenda or context/pooling fields.
ProductKind = Literal[
    "source_record",
    "selected_region",
    "construct_insert",
    "analysis_window",
    "realized_context",
]
Orientation = Literal["forward", "reverse_complement", "unknown"]
ContextKind = Literal["anchor_only", "template_1kb", "template_custom", "native_reference", "analysis_window"]
PoolingOperation = Literal["seq_mean", "anchor_mean", "core60_mean"]
SequenceViewConflictPolicy = Literal["error", "idempotent", "replace", "append_alias"]
ViewSemanticsConflictPolicy = Literal["error", "idempotent", "replace"]


def _none_if_blank(value: object) -> str | None:
    text = str(value or "").strip()
    return text or None


def _stable_aliases(values: list[str] | None) -> list[str] | None:
    if values is None:
        return None
    seen: set[str] = set()
    out: list[str] = []
    for raw in values:
        text = _none_if_blank(raw)
        if text is None:
            continue
        lowered = text.casefold()
        if lowered in seen:
            continue
        seen.add(lowered)
        out.append(text)
    return out or None


class StrictSequenceViewModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SequenceViewSemanticKey(StrictSequenceViewModel):
    sequence_id: str
    source_dataset_id: str | None = None
    product_kind: ProductKind
    parent_sequence_id: str | None = None
    parent_dataset_id: str | None = None
    derivation_spec_id: str | None = None
    source_interval_start_0: int | None = Field(default=None, ge=0)
    source_interval_end_0: int | None = Field(default=None, ge=0)
    anchor_start_0: int | None = Field(default=None, ge=0)
    anchor_end_0: int | None = Field(default=None, ge=0)
    orientation: Orientation
    template_sequence_id: str | None = None
    template_dataset_id: str | None = None
    analysis_only: bool = False

    @field_validator(
        "sequence_id",
        "source_dataset_id",
        "parent_sequence_id",
        "parent_dataset_id",
        "derivation_spec_id",
        "template_sequence_id",
        "template_dataset_id",
    )
    @classmethod
    def _normalize_strings(cls, value: str | None) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            raise ValueError("Sequence-view semantic key fields must be non-empty when provided.")
        return text

    @model_validator(mode="after")
    def _validate_bounds(self) -> "SequenceViewSemanticKey":
        if self.source_interval_start_0 is None and self.source_interval_end_0 is not None:
            raise ValueError("source_interval_end_0 requires source_interval_start_0.")
        if self.source_interval_start_0 is not None and self.source_interval_end_0 is None:
            raise ValueError("source_interval_start_0 requires source_interval_end_0.")
        if (
            self.source_interval_start_0 is not None
            and self.source_interval_end_0 is not None
            and self.source_interval_end_0 < self.source_interval_start_0
        ):
            raise ValueError("source_interval_end_0 must be >= source_interval_start_0.")
        if self.anchor_start_0 is None and self.anchor_end_0 is not None:
            raise ValueError("anchor_end_0 requires anchor_start_0.")
        if self.anchor_start_0 is not None and self.anchor_end_0 is None:
            raise ValueError("anchor_start_0 requires anchor_end_0.")
        if (
            self.anchor_start_0 is not None
            and self.anchor_end_0 is not None
            and self.anchor_end_0 < self.anchor_start_0
        ):
            raise ValueError("anchor_end_0 must be >= anchor_start_0.")
        return self

    def canonical_payload(self) -> dict[str, object]:
        return {
            "schema_version": VIEW_ID_SCHEMA_VERSION,
            "sequence_id": self.sequence_id,
            "source_dataset_id": self.source_dataset_id,
            "product_kind": self.product_kind,
            "parent_sequence_id": self.parent_sequence_id,
            "parent_dataset_id": self.parent_dataset_id,
            "derivation_spec_id": self.derivation_spec_id,
            "source_interval_start_0": self.source_interval_start_0,
            "source_interval_end_0": self.source_interval_end_0,
            "anchor_start_0": self.anchor_start_0,
            "anchor_end_0": self.anchor_end_0,
            "orientation": self.orientation,
            "template_sequence_id": self.template_sequence_id,
            "template_dataset_id": self.template_dataset_id,
            "analysis_only": bool(self.analysis_only),
        }


def compute_sequence_view_id(key: SequenceViewSemanticKey) -> str:
    payload = json.dumps(key.canonical_payload(), sort_keys=True, separators=(",", ":"))
    return f"view_{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:24]}"


class SequenceViewRecord(StrictSequenceViewModel):
    view_id: str | None = None
    sequence_id: str
    view_name: str | None = None
    aliases: list[str] | None = None
    product_kind: ProductKind
    context_kind: ContextKind | None = None
    orientation: Orientation
    analysis_only: bool = False
    source_dataset_id: str | None = None
    source_label: str | None = None
    parent_sequence_id: str | None = None
    parent_dataset_id: str | None = None
    derivation_id: str | None = None
    derivation_spec_id: str | None = None
    template_sequence_id: str | None = None
    template_dataset_id: str | None = None
    source_interval_start_0: int | None = Field(default=None, ge=0)
    source_interval_end_0: int | None = Field(default=None, ge=0)
    anchor_start_0: int | None = Field(default=None, ge=0)
    anchor_end_0: int | None = Field(default=None, ge=0)
    forward_anchor_start_0: int | None = Field(default=None, ge=0)
    forward_anchor_end_0: int | None = Field(default=None, ge=0)
    recommended_pooling: PoolingOperation | None = None
    created_at: str
    created_by: str | None = None

    @field_validator(
        "sequence_id",
        "view_name",
        "source_dataset_id",
        "source_label",
        "parent_sequence_id",
        "parent_dataset_id",
        "derivation_id",
        "derivation_spec_id",
        "template_sequence_id",
        "template_dataset_id",
        "created_by",
    )
    @classmethod
    def _normalize_optional_strings(cls, value: str | None) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            raise ValueError("Sequence-view string fields must be non-empty when provided.")
        return text

    @field_validator("view_id")
    @classmethod
    def _normalize_view_id(cls, value: str | None) -> str | None:
        return _none_if_blank(value)

    @field_validator("aliases")
    @classmethod
    def _normalize_aliases(cls, value: list[str] | None) -> list[str] | None:
        return _stable_aliases(value)

    @model_validator(mode="after")
    def _finalize_and_validate(self) -> "SequenceViewRecord":
        semantic_key = self.semantic_key()
        expected_view_id = compute_sequence_view_id(semantic_key)
        if self.view_id is None:
            self.view_id = expected_view_id
        elif self.view_id != expected_view_id:
            raise ValueError("view_id does not match the semantic key for this sequence view.")
        if self.anchor_start_0 is None and self.anchor_end_0 is not None:
            raise ValueError("anchor_end_0 requires anchor_start_0.")
        if self.anchor_start_0 is not None and self.anchor_end_0 is None:
            raise ValueError("anchor_start_0 requires anchor_end_0.")
        if self.forward_anchor_start_0 is None and self.forward_anchor_end_0 is not None:
            raise ValueError("forward_anchor_end_0 requires forward_anchor_start_0.")
        if self.forward_anchor_start_0 is not None and self.forward_anchor_end_0 is None:
            raise ValueError("forward_anchor_start_0 requires forward_anchor_end_0.")
        if self.orientation == "unknown" and self.product_kind != "source_record":
            raise ValueError("orientation='unknown' is only valid for source_record sequence views.")
        return self

    def semantic_key(self) -> SequenceViewSemanticKey:
        return SequenceViewSemanticKey(
            sequence_id=self.sequence_id,
            source_dataset_id=self.source_dataset_id,
            product_kind=self.product_kind,
            parent_sequence_id=self.parent_sequence_id,
            parent_dataset_id=self.parent_dataset_id,
            derivation_spec_id=self.derivation_spec_id,
            source_interval_start_0=self.source_interval_start_0,
            source_interval_end_0=self.source_interval_end_0,
            anchor_start_0=self.anchor_start_0,
            anchor_end_0=self.anchor_end_0,
            orientation=self.orientation,
            template_sequence_id=self.template_sequence_id,
            template_dataset_id=self.template_dataset_id,
            analysis_only=bool(self.analysis_only),
        )

    def semantic_payload(self) -> dict[str, object]:
        return self.semantic_key().canonical_payload()

    def mutable_payload(self) -> dict[str, object]:
        return {
            "view_name": self.view_name,
            "aliases": list(self.aliases or []),
            "context_kind": self.context_kind,
            "source_label": self.source_label,
            "derivation_id": self.derivation_id,
            "forward_anchor_start_0": self.forward_anchor_start_0,
            "forward_anchor_end_0": self.forward_anchor_end_0,
            "recommended_pooling": self.recommended_pooling,
            "created_by": self.created_by,
        }


class SequenceViewSelector(StrictSequenceViewModel):
    view_id: str | None = None
    sequence_id: str | None = None
    product_kind: ProductKind | None = None
    view_name: str | None = None
    alias: str | None = None

    @field_validator("view_id", "sequence_id", "view_name", "alias")
    @classmethod
    def _normalize_selector_string(cls, value: str | None) -> str | None:
        return _none_if_blank(value)


class ViewSemanticsRecord(StrictSequenceViewModel):
    """Mutable study/provenance semantics that must not affect stable view ids."""

    view_id: str
    sequence_id: str
    source_family: str | None = None
    selection_basis: str | None = None
    view_collections: list[str] | None = None
    role_tags: list[str] | None = None
    study_id: str | None = None
    created_at: str
    created_by: str | None = None

    @field_validator(
        "view_id",
        "sequence_id",
        "source_family",
        "selection_basis",
        "study_id",
        "created_at",
        "created_by",
    )
    @classmethod
    def _normalize_strings(cls, value: str | None) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            raise ValueError("View-semantics string fields must be non-empty when provided.")
        return text

    @field_validator("view_collections", "role_tags")
    @classmethod
    def _normalize_lists(cls, value: list[str] | None) -> list[str] | None:
        return _stable_aliases(value)
