"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cruncher/src/nickases/models.py

Shared sequence helpers and normalized nickase contracts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_DNA_RE = re.compile(r"^[ACGT]+$")
_IUPAC_RE = re.compile(r"^[ACGTRYSWKMBDHVN]+$")
_IUPAC_MAP: dict[str, set[str]] = {
    "A": {"A"},
    "C": {"C"},
    "G": {"G"},
    "T": {"T"},
    "R": {"A", "G"},
    "Y": {"C", "T"},
    "S": {"G", "C"},
    "W": {"A", "T"},
    "K": {"G", "T"},
    "M": {"A", "C"},
    "B": {"C", "G", "T"},
    "D": {"A", "G", "T"},
    "H": {"A", "C", "T"},
    "V": {"A", "C", "G"},
    "N": {"A", "C", "G", "T"},
}
_DNA_COMPLEMENT = str.maketrans("ACGT", "TGCA")
_IUPAC_COMPLEMENT = str.maketrans(
    {
        "A": "T",
        "C": "G",
        "G": "C",
        "T": "A",
        "R": "Y",
        "Y": "R",
        "S": "S",
        "W": "W",
        "K": "M",
        "M": "K",
        "B": "V",
        "D": "H",
        "H": "D",
        "V": "B",
        "N": "N",
    }
)


class StrictNickaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


CommercialConfidence = Literal[
    "primary_vendor_current",
    "secondary_vendor_current",
    "produced_on_demand",
    "literature_only",
]
YesNoUnknown = Literal["yes", "no", "unknown"]
RawCutOffsetReference = Literal["motif_start", "motif_end"]


def normalize_dna(value: str, *, allow_empty: bool = False) -> str:
    text = str(value or "").strip().upper()
    if not text:
        if allow_empty:
            return ""
        raise ValueError("DNA sequence cannot be empty.")
    if not _DNA_RE.fullmatch(text):
        raise ValueError(f"DNA sequence must contain only A/C/G/T: {value!r}")
    return text


def normalize_iupac(value: str) -> str:
    text = str(value or "").strip().upper()
    if not text:
        raise ValueError("DNA motif cannot be empty.")
    if not _IUPAC_RE.fullmatch(text):
        raise ValueError(f"DNA motif must contain only IUPAC nucleotide symbols: {value!r}")
    return text


def reverse_complement(sequence: str) -> str:
    return sequence.upper().translate(_DNA_COMPLEMENT)[::-1]


def reverse_complement_iupac(sequence: str) -> str:
    return sequence.upper().translate(_IUPAC_COMPLEMENT)[::-1]


def iupac_bases_for_symbol(symbol: str) -> set[str]:
    text = str(symbol or "").strip().upper()
    if len(text) != 1 or text not in _IUPAC_MAP:
        raise ValueError(f"Unknown IUPAC nucleotide symbol: {symbol!r}")
    return set(_IUPAC_MAP[text])


def motif_matches(sequence: str, motif: str) -> bool:
    sequence_text = normalize_dna(sequence)
    motif_text = normalize_iupac(motif)
    if len(sequence_text) != len(motif_text):
        return False
    return all(base in _IUPAC_MAP[symbol] for base, symbol in zip(sequence_text, motif_text, strict=True))


class NickaseSelectionProfile(StrictNickaseModel):
    outside_site: bool | None = None
    commercial_confidence: CommercialConfidence | None = None
    warning_codes: list[str] = Field(default_factory=list)

    @field_validator("warning_codes")
    @classmethod
    def _validate_warning_codes(cls, value: list[str]) -> list[str]:
        normalized = [str(item or "").strip() for item in value]
        if any(not item for item in normalized):
            raise ValueError("warning_codes must not contain blank values.")
        return normalized


class NickaseOperationalProfile(StrictNickaseModel):
    incubation_temp_c: int | None = Field(default=None, ge=0)
    buffer_family: str | None = None
    heat_inactivation: str | None = None
    methylation_sensitivity: dict[str, YesNoUnknown] = Field(default_factory=dict)
    star_activity_warning: bool | None = None
    double_strand_cleavage_warning: bool | None = None

    @field_validator("buffer_family", "heat_inactivation")
    @classmethod
    def _validate_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @field_validator("methylation_sensitivity")
    @classmethod
    def _validate_methylation_sensitivity(cls, value: dict[str, YesNoUnknown]) -> dict[str, YesNoUnknown]:
        normalized: dict[str, YesNoUnknown] = {}
        for key, item in value.items():
            key_text = str(key or "").strip()
            if not key_text:
                raise ValueError("methylation_sensitivity keys must be non-empty.")
            normalized[key_text] = item
        return normalized


class NickaseCatalogEntry(StrictNickaseModel):
    id: str
    specificity_id: str
    motif_top_5to3: str
    vendor_diagram_top_5to3: str | None = None
    motif_len: int | None = None
    top_cut_offset: int | None = None
    bottom_cut_offset: int | None = None
    source: str | None = None
    vendor: str | None = None
    vendor_catalog_number: str | None = None
    source_url: str | None = None
    origin_class: str | None = None
    source_family: str | None = None
    notes: list[str] = Field(default_factory=list)
    selection: NickaseSelectionProfile | None = None
    operational: NickaseOperationalProfile | None = None
    raw_cut_notation: str | None = None
    raw_cut_offset_reference: RawCutOffsetReference | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("id", "specificity_id")
    @classmethod
    def _validate_id(cls, value: str) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError("nickase id and specificity_id must be non-empty.")
        return text

    @field_validator("source", "vendor", "vendor_catalog_number", "source_url", "origin_class", "source_family")
    @classmethod
    def _validate_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @field_validator("notes")
    @classmethod
    def _validate_notes(cls, value: list[str]) -> list[str]:
        normalized = [str(item or "").strip() for item in value]
        if any(not item for item in normalized):
            raise ValueError("nickase notes must not contain blank values.")
        return normalized

    @field_validator("motif_top_5to3", "vendor_diagram_top_5to3")
    @classmethod
    def _validate_motif(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return normalize_iupac(value)

    @model_validator(mode="after")
    def _validate_offsets(self) -> "NickaseCatalogEntry":
        if (self.top_cut_offset is None) == (self.bottom_cut_offset is None):
            raise ValueError("nickase variants must define exactly one of top_cut_offset or bottom_cut_offset.")
        expected_len = len(self.motif_top_5to3)
        if self.motif_len is None:
            self.motif_len = expected_len
        elif self.motif_len != expected_len:
            raise ValueError("motif_len must equal len(motif_top_5to3).")
        if self.vendor_diagram_top_5to3 is not None and len(self.vendor_diagram_top_5to3) < expected_len:
            raise ValueError("vendor_diagram_top_5to3 must be at least as long as motif_top_5to3.")
        return self

    @property
    def nicked_strand(self) -> Literal["top", "bottom"]:
        return "top" if self.top_cut_offset is not None else "bottom"

    @property
    def active_cut_offset(self) -> int:
        return int(self.top_cut_offset if self.top_cut_offset is not None else self.bottom_cut_offset)

    @property
    def outside_site(self) -> bool | None:
        if self.selection is None:
            return None
        return self.selection.outside_site

    @property
    def resolved_vendor_diagram_top_5to3(self) -> str:
        if self.vendor_diagram_top_5to3 is not None:
            return self.vendor_diagram_top_5to3
        if self.active_cut_offset >= len(self.motif_top_5to3):
            suffix_nt = self.active_cut_offset - len(self.motif_top_5to3) + 1
            return f"{self.motif_top_5to3}{'N' * suffix_nt}"
        return self.motif_top_5to3

    @property
    def resolved_vendor_diagram_len(self) -> int:
        return len(self.resolved_vendor_diagram_top_5to3)


class NickaseProductAlias(StrictNickaseModel):
    alias_id: str
    canonical_variant_id: str
    vendor: str | None = None
    vendor_catalog_number: str | None = None
    source_url: str | None = None
    alias_kind: str | None = None
    notes: list[str] = Field(default_factory=list)

    @field_validator("alias_id", "canonical_variant_id")
    @classmethod
    def _validate_id(cls, value: str) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError("product alias ids must be non-empty.")
        return text

    @field_validator("vendor", "vendor_catalog_number", "source_url", "alias_kind")
    @classmethod
    def _validate_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None


class NickaseCatalog(StrictNickaseModel):
    schema_version: int = 1
    entries: list[NickaseCatalogEntry]
    preset_id: str | None = None
    preset_ids: list[str] = Field(default_factory=list)
    catalog_version: int | None = None
    generated_from: str | None = None
    generated_on: str | None = None
    normalization_policy: str | None = None
    product_aliases: list[NickaseProductAlias] = Field(default_factory=list)

    @field_validator("schema_version")
    @classmethod
    def _validate_schema_version(cls, value: int) -> int:
        if int(value) != 1:
            raise ValueError("nickases.schema_version must be 1.")
        return int(value)

    @field_validator("preset_id", "generated_from", "generated_on", "normalization_policy")
    @classmethod
    def _validate_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @field_validator("preset_ids")
    @classmethod
    def _validate_preset_ids(cls, value: list[str]) -> list[str]:
        normalized = [str(item or "").strip() for item in value]
        if any(not item for item in normalized):
            raise ValueError("nickase catalog preset_ids must not contain blank values.")
        if len(set(normalized)) != len(normalized):
            raise ValueError("nickase catalog preset_ids must be unique.")
        return normalized

    @field_validator("entries")
    @classmethod
    def _validate_entries(cls, value: list[NickaseCatalogEntry]) -> list[NickaseCatalogEntry]:
        if not value:
            raise ValueError("nickase catalog must define at least one entry.")
        ids = [entry.id for entry in value]
        if len(set(ids)) != len(ids):
            raise ValueError("nickase catalog ids must be unique.")
        motifs_by_specificity: dict[str, str] = {}
        for entry in value:
            existing = motifs_by_specificity.get(entry.specificity_id)
            if existing is None:
                motifs_by_specificity[entry.specificity_id] = entry.motif_top_5to3
            elif existing != entry.motif_top_5to3:
                raise ValueError(
                    "nickase catalog entries that share a specificity_id must use the same motif_top_5to3."
                )
        return value

    @model_validator(mode="after")
    def _validate_product_aliases(self) -> "NickaseCatalog":
        if self.preset_id is not None and not self.preset_ids:
            self.preset_ids = [self.preset_id]
        elif self.preset_id is None and self.preset_ids:
            self.preset_id = self.preset_ids[0]
        elif self.preset_id is not None and self.preset_ids and self.preset_ids[0] != self.preset_id:
            raise ValueError("nickase catalog preset_id must match preset_ids[0] when both are provided.")
        entry_ids = {entry.id for entry in self.entries}
        alias_ids = [alias.alias_id for alias in self.product_aliases]
        if len(set(alias_ids)) != len(alias_ids):
            raise ValueError("nickase product alias ids must be unique.")
        for alias in self.product_aliases:
            if alias.canonical_variant_id not in entry_ids:
                raise ValueError(
                    f"nickase product alias {alias.alias_id} references unknown canonical_variant_id "
                    f"{alias.canonical_variant_id!r}."
                )
        return self

    def by_id(self) -> dict[str, NickaseCatalogEntry]:
        return {entry.id: entry for entry in self.entries}


class NickaseCatalogDocument(StrictNickaseModel):
    nickases: NickaseCatalog


class RecognitionSiteInstance(StrictNickaseModel):
    variant_id: str
    specificity_id: str
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    orientation: str
    matched_span_sequence: str
    local_start: int | None = None
    local_end: int | None = None

    @model_validator(mode="after")
    def _validate_bounds(self) -> "RecognitionSiteInstance":
        if self.end <= self.start:
            raise ValueError("recognition site end must be > start.")
        return self

    @property
    def cassette_start(self) -> int | None:
        return self.local_start

    @property
    def cassette_end(self) -> int | None:
        return self.local_end


class NickEvent(StrictNickaseModel):
    variant_id: str
    specificity_id: str
    strand: str
    boundary: int
    boundary_context: int = Field(ge=0)
    source_site_start: int = Field(ge=0)
    source_site_end: int = Field(ge=0)
    source_site_orientation: str
