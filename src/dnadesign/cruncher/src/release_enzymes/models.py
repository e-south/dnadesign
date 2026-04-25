"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/release_enzymes/models.py

Normalized release-enzyme contracts and resolved site/cut models.

Module Author(s): Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from dnadesign.cruncher.nickases.models import normalize_dna, normalize_iupac


class StrictReleaseEnzymeModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


ReleaseCommercialConfidence = Literal[
    "primary_vendor_current",
    "secondary_vendor_current",
    "legacy_vendor_page",
]
ReleaseClassLabel = Literal["type_iis", "type_iia", "other_ds_re"]


class ReleaseEnzymeEntry(StrictReleaseEnzymeModel):
    variant_id: str
    display_name: str
    recognition_sequence: str
    top_cut_offset: int
    bottom_cut_offset: int
    class_label: ReleaseClassLabel
    commercial_confidence: ReleaseCommercialConfidence
    warning_codes: list[str] = Field(default_factory=list)
    recommended_5prime_flanking_bases: int | None = Field(default=None, ge=0)
    source_catalog_id: str
    source_url: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("variant_id", "display_name", "source_catalog_id")
    @classmethod
    def _validate_required_text(cls, value: str) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError("release-enzyme required text fields must be non-empty.")
        return text

    @field_validator("recognition_sequence")
    @classmethod
    def _validate_recognition_sequence(cls, value: str) -> str:
        return normalize_iupac(value)

    @field_validator("source_url")
    @classmethod
    def _validate_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @field_validator("warning_codes")
    @classmethod
    def _validate_warning_codes(cls, value: list[str]) -> list[str]:
        normalized = [str(item or "").strip() for item in value]
        if any(not item for item in normalized):
            raise ValueError("release-enzyme warning_codes must not contain blank values.")
        return normalized

    @property
    def recognition_len(self) -> int:
        return len(self.recognition_sequence)

    @property
    def outside_site(self) -> bool:
        motif_len = self.recognition_len
        return (
            self.top_cut_offset < 0
            or self.bottom_cut_offset < 0
            or self.top_cut_offset > motif_len
            or self.bottom_cut_offset > motif_len
        )

    @property
    def proximal_reach_from_site_end(self) -> int:
        motif_len = self.recognition_len
        return min(abs(self.top_cut_offset - motif_len), abs(self.bottom_cut_offset - motif_len))


class ReleaseEnzymeCatalog(StrictReleaseEnzymeModel):
    schema_version: int = 1
    entries: list[ReleaseEnzymeEntry]
    preset_id: str | None = None
    preset_ids: list[str] = Field(default_factory=list)
    catalog_version: int | None = None
    generated_from: str | None = None
    generated_on: str | date | None = None
    normalization_policy: str | None = None

    @field_validator("schema_version")
    @classmethod
    def _validate_schema_version(cls, value: int) -> int:
        if int(value) != 1:
            raise ValueError("release_enzymes.schema_version must be 1.")
        return int(value)

    @field_validator("preset_id", "generated_from", "generated_on", "normalization_policy")
    @classmethod
    def _validate_optional_text(cls, value: str | date | None) -> str | None:
        if value is None:
            return None
        if isinstance(value, date):
            return value.isoformat()
        text = str(value).strip()
        return text or None

    @field_validator("preset_ids")
    @classmethod
    def _validate_preset_ids(cls, value: list[str]) -> list[str]:
        normalized = [str(item or "").strip() for item in value]
        if any(not item for item in normalized):
            raise ValueError("release-enzyme catalog preset_ids must not contain blank values.")
        if len(set(normalized)) != len(normalized):
            raise ValueError("release-enzyme catalog preset_ids must be unique.")
        return normalized

    @field_validator("entries")
    @classmethod
    def _validate_entries(cls, value: list[ReleaseEnzymeEntry]) -> list[ReleaseEnzymeEntry]:
        if not value:
            raise ValueError("release-enzyme catalog must define at least one entry.")
        variant_ids = [entry.variant_id for entry in value]
        if len(set(variant_ids)) != len(variant_ids):
            raise ValueError("release-enzyme catalog variant ids must be unique.")
        return value

    @model_validator(mode="after")
    def _validate_presets(self) -> "ReleaseEnzymeCatalog":
        if self.preset_id is not None and not self.preset_ids:
            self.preset_ids = [self.preset_id]
        elif self.preset_id is None and self.preset_ids:
            self.preset_id = self.preset_ids[0]
        elif self.preset_id is not None and self.preset_ids and self.preset_ids[0] != self.preset_id:
            raise ValueError("release-enzyme catalog preset_id must match preset_ids[0] when both are provided.")
        return self

    def by_id(self) -> dict[str, ReleaseEnzymeEntry]:
        return {entry.variant_id: entry for entry in self.entries}


class ReleaseEnzymeCatalogDocument(StrictReleaseEnzymeModel):
    release_enzymes: ReleaseEnzymeCatalog


class ReleaseRecognitionSiteInstance(StrictReleaseEnzymeModel):
    variant_id: str
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    orientation: Literal["forward", "reverse"]
    matched_span_sequence: str
    local_start: int | None = None
    local_end: int | None = None

    @field_validator("matched_span_sequence")
    @classmethod
    def _validate_matched_span_sequence(cls, value: str) -> str:
        return normalize_dna(value)

    @model_validator(mode="after")
    def _validate_bounds(self) -> "ReleaseRecognitionSiteInstance":
        if self.end <= self.start:
            raise ValueError("release recognition site end must be > start.")
        return self


class ReleaseCutEvent(StrictReleaseEnzymeModel):
    variant_id: str
    top_cut_boundary: int
    bottom_cut_boundary: int
    source_site_start: int = Field(ge=0)
    source_site_end: int = Field(ge=0)
    source_site_orientation: Literal["forward", "reverse"]

    @property
    def proximal_cut_boundary(self) -> int:
        return min(self.top_cut_boundary, self.bottom_cut_boundary)

    @property
    def distal_cut_boundary(self) -> int:
        return max(self.top_cut_boundary, self.bottom_cut_boundary)


__all__ = [
    "ReleaseClassLabel",
    "ReleaseCommercialConfidence",
    "ReleaseCutEvent",
    "ReleaseEnzymeCatalog",
    "ReleaseEnzymeCatalogDocument",
    "ReleaseEnzymeEntry",
    "ReleaseRecognitionSiteInstance",
    "StrictReleaseEnzymeModel",
]
