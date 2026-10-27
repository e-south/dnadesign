"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/src/genbank/models.py

Manifest and parsed-record contracts for GenBank-backed USR import.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..sequence_views import ProductKind


def _none_if_blank(value: object) -> str | None:
    text = str(value or "").strip()
    return text or None


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RoleHintRule(StrictModel):
    match_label: str | None = None
    match_any_label: list[str] | None = None
    role_hint: str

    @field_validator("match_label", "role_hint")
    @classmethod
    def _normalize_string(cls, value: str | None) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            raise ValueError("Role-hint rule strings must be non-empty when provided.")
        return text

    @field_validator("match_any_label")
    @classmethod
    def _normalize_label_list(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        out: list[str] = []
        seen: set[str] = set()
        for raw in value:
            text = _none_if_blank(raw)
            if text is None:
                continue
            lowered = text.casefold()
            if lowered in seen:
                continue
            seen.add(lowered)
            out.append(text)
        return out or None

    @model_validator(mode="after")
    def _validate_selector(self) -> "RoleHintRule":
        if self.match_label is None and not self.match_any_label:
            raise ValueError("RoleHintRule requires match_label or match_any_label.")
        return self

    def matches(self, label: str | None) -> bool:
        if label is None:
            return False
        candidate = label.casefold()
        if self.match_label is not None and candidate == self.match_label.casefold():
            return True
        return any(candidate == item.casefold() for item in self.match_any_label or [])


class GenBankImportRecordSpec(StrictModel):
    source_file: str
    label: str
    aliases: list[str] | None = None
    product_kind: Literal["source_record"] = "source_record"

    @field_validator("source_file", "label")
    @classmethod
    def _normalize_required_string(cls, value: str) -> str:
        text = str(value).strip()
        if not text:
            raise ValueError("Manifest record fields must be non-empty.")
        return text

    @field_validator("aliases")
    @classmethod
    def _normalize_aliases(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        out: list[str] = []
        seen: set[str] = set()
        for raw in value:
            text = _none_if_blank(raw)
            if text is None:
                continue
            lowered = text.casefold()
            if lowered in seen:
                continue
            seen.add(lowered)
            out.append(text)
        return out or None


class FeatureSelector(StrictModel):
    kind: Literal["label", "feature_id"]
    label: str | None = None
    feature_id: str | None = None

    @field_validator("label", "feature_id")
    @classmethod
    def _normalize_optional_string(cls, value: str | None) -> str | None:
        return _none_if_blank(value)

    @model_validator(mode="after")
    def _validate_selector(self) -> "FeatureSelector":
        if self.kind == "label" and self.label is None:
            raise ValueError("FeatureSelector kind='label' requires label.")
        if self.kind == "feature_id" and self.feature_id is None:
            raise ValueError("FeatureSelector kind='feature_id' requires feature_id.")
        return self


class FeatureExtractionSpec(StrictModel):
    source_label: str
    selector: FeatureSelector
    product_kind: ProductKind
    view_name: str
    aliases: list[str] | None = None
    on_ambiguous: Literal["error"] = "error"

    @field_validator("source_label", "view_name")
    @classmethod
    def _normalize_required_string(cls, value: str) -> str:
        text = str(value).strip()
        if not text:
            raise ValueError("Feature extraction fields must be non-empty.")
        return text

    @field_validator("aliases")
    @classmethod
    def _normalize_aliases(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        out: list[str] = []
        seen: set[str] = set()
        for raw in value:
            text = _none_if_blank(raw)
            if text is None:
                continue
            lowered = text.casefold()
            if lowered in seen:
                continue
            seen.add(lowered)
            out.append(text)
        return out or None


class GenBankImportManifest(StrictModel):
    kind: Literal["usr.genbank_import"]
    version: Literal[1]
    output_dataset: str
    on_conflict: Literal["error", "idempotent"] = "error"
    copy_source_artifacts: bool = False
    role_hint_rules: list[RoleHintRule] = Field(default_factory=list)
    records: list[GenBankImportRecordSpec]
    extract_features: list[FeatureExtractionSpec] = Field(default_factory=list)

    @field_validator("output_dataset")
    @classmethod
    def _normalize_dataset(cls, value: str) -> str:
        text = str(value).strip()
        if not text:
            raise ValueError("output_dataset must be non-empty.")
        if "/" in text or "\\" in text:
            raise ValueError("output_dataset must be a flat owner-first dataset id, not a path.")
        if re.fullmatch(r"[a-z][a-z0-9_]*", text) is None:
            raise ValueError("output_dataset must use flat lowercase owner-first snake_case.")
        return text

    @model_validator(mode="after")
    def _validate_uniqueness(self) -> "GenBankImportManifest":
        seen_labels: set[str] = set()
        for record in self.records:
            lowered = record.label.casefold()
            if lowered in seen_labels:
                raise ValueError(f"Duplicate record label '{record.label}' in manifest.")
            seen_labels.add(lowered)
        known_labels = {record.label.casefold() for record in self.records}
        for extraction in self.extract_features:
            if extraction.source_label.casefold() not in known_labels:
                raise ValueError(f"extract_features references unknown source_label '{extraction.source_label}'.")
        return self


class ParsedFeatureInterval(StrictModel):
    start_0: int = Field(ge=0)
    end_0: int = Field(ge=0)
    strand: int | None = None
    partial: bool = False

    @model_validator(mode="after")
    def _validate_bounds(self) -> "ParsedFeatureInterval":
        if self.end_0 < self.start_0:
            raise ValueError("Parsed feature interval end must be >= start.")
        return self


class ParsedQualifier(StrictModel):
    key: str
    value: str

    @field_validator("key")
    @classmethod
    def _normalize_required_key(cls, value: str) -> str:
        text = str(value).strip()
        if not text:
            raise ValueError("Parsed qualifier keys must be non-empty.")
        return text

    @field_validator("value")
    @classmethod
    def _normalize_value(cls, value: str) -> str:
        # GenBank supports valueless/blank qualifiers; preserving them is part
        # of source fidelity. Only keys are required to be non-empty.
        return str(value)


class ParsedGenBankFeature(StrictModel):
    feature_id: str
    feature_order: int = Field(ge=0)
    feature_type: str
    label: str | None = None
    role_hint: str | None = None
    location_raw: str
    location_kind: str
    start_0: int | None = Field(default=None, ge=0)
    end_0: int | None = Field(default=None, ge=0)
    strand: int | None = None
    intervals_0: list[ParsedFeatureInterval] = Field(default_factory=list)
    is_fuzzy: bool = False
    is_compound: bool = False
    qualifiers: list[ParsedQualifier] = Field(default_factory=list)
    confidence: Literal["high", "low", "unknown"]
    source: str


class ParsedGenBankRecord(StrictModel):
    source_file: str
    source_sha256: str
    record_id: str | None = None
    record_name: str | None = None
    description: str | None = None
    topology: str | None = None
    molecule_type: str | None = None
    sequence_region_start_0: int | None = Field(default=None, ge=0)
    sequence_region_end_0: int | None = Field(default=None, ge=0)
    sequence: str
    features: list[ParsedGenBankFeature] = Field(default_factory=list)

    @field_validator(
        "source_file",
        "source_sha256",
        "record_id",
        "record_name",
        "description",
        "topology",
        "molecule_type",
        "sequence",
    )
    @classmethod
    def _normalize_optional_string(cls, value: str | None) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            raise ValueError("Parsed record strings must be non-empty when provided.")
        return text

    @model_validator(mode="after")
    def _validate_region(self) -> "ParsedGenBankRecord":
        if self.sequence_region_start_0 is None and self.sequence_region_end_0 is not None:
            raise ValueError("sequence_region_end_0 requires sequence_region_start_0.")
        if self.sequence_region_start_0 is not None and self.sequence_region_end_0 is None:
            raise ValueError("sequence_region_start_0 requires sequence_region_end_0.")
        if (
            self.sequence_region_start_0 is not None
            and self.sequence_region_end_0 is not None
            and self.sequence_region_end_0 < self.sequence_region_start_0
        ):
            raise ValueError("sequence_region_end_0 must be >= sequence_region_start_0.")
        return self


__all__ = [
    "FeatureExtractionSpec",
    "FeatureSelector",
    "GenBankImportManifest",
    "GenBankImportRecordSpec",
    "ParsedFeatureInterval",
    "ParsedGenBankFeature",
    "ParsedGenBankRecord",
    "ParsedQualifier",
    "RoleHintRule",
]
