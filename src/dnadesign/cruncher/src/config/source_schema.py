"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/config/source_schema.py

Source and IO schema models used by Cruncher configuration v3.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrictBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ParserConfig(StrictBaseModel):
    extra_modules: List[str] = Field(
        default_factory=list,
        description="Additional modules to import for parser registration.",
    )

    @field_validator("extra_modules")
    @classmethod
    def _check_extra_modules(cls, v: List[str]) -> List[str]:
        cleaned = []
        for mod in v:
            name = str(mod).strip()
            if not name:
                raise ValueError("io.parsers.extra_modules entries must be non-empty strings")
            cleaned.append(name)
        return cleaned


class IOConfig(StrictBaseModel):
    parsers: ParserConfig = ParserConfig()


class OrganismConfig(StrictBaseModel):
    taxon: Optional[int] = None
    name: Optional[str] = None
    strain: Optional[str] = None
    assembly: Optional[str] = None


class LocalMotifSourceConfig(StrictBaseModel):
    source_id: str = Field(..., description="Unique identifier for the local motif source.")
    description: Optional[str] = Field(None, description="Human-readable description for the source list.")
    root: Path = Field(..., description="Root directory containing motif files.")
    patterns: List[str] = Field(
        default_factory=lambda: ["*.txt"],
        description="Glob patterns to select motif files under root.",
    )
    recursive: bool = Field(False, description="Recursively search subdirectories when true.")
    format_map: Dict[str, str] = Field(
        default_factory=dict,
        description="Mapping from file extension to parser format (e.g., .txt -> MEME).",
    )
    default_format: Optional[str] = Field(
        None,
        description="Default parser format when an extension is not listed in format_map.",
    )
    tf_name_strategy: Literal["stem", "filename"] = Field(
        "stem",
        description="Strategy for deriving TF names from files.",
    )
    matrix_semantics: Literal["probabilities", "weights"] = Field(
        "probabilities",
        description="Matrix semantics to store in the catalog.",
    )
    organism: Optional[OrganismConfig] = None
    citation: Optional[str] = None
    license: Optional[str] = None
    source_url: Optional[str] = None
    source_version: Optional[str] = None
    tags: Dict[str, str] = Field(default_factory=dict)
    extract_sites: bool = Field(
        False,
        description="Enable binding-site extraction from MEME BLOCKS sections.",
    )
    meme_motif_selector: Optional[Union[str, int]] = Field(
        None,
        description="Select a motif from multi-motif MEME files (name_match, MEME-1, index, or label).",
    )

    @field_validator("source_id")
    @classmethod
    def _check_source_id(cls, v: str) -> str:
        source_id = str(v).strip()
        if not source_id:
            raise ValueError("local source_id must be a non-empty string")
        return source_id

    @field_validator("patterns")
    @classmethod
    def _check_patterns(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("local source patterns must be a non-empty list")
        cleaned = []
        for pat in v:
            pat = str(pat).strip()
            if not pat:
                raise ValueError("local source patterns must be non-empty strings")
            cleaned.append(pat)
        return cleaned

    @model_validator(mode="after")
    def _check_format_config(self) -> "LocalMotifSourceConfig":
        if not self.format_map and not self.default_format:
            raise ValueError("local source must set format_map or default_format")
        return self


class LocalSiteSourceConfig(StrictBaseModel):
    source_id: str = Field(..., description="Unique identifier for the local site source.")
    description: Optional[str] = Field(None, description="Human-readable description for the source list.")
    path: Path = Field(..., description="Path to a FASTA file containing binding sites.")
    tf_name: Optional[str] = Field(
        None,
        description="Optional TF name override (defaults to the FASTA header prefix).",
    )
    record_kind: Optional[str] = Field(
        None,
        description="Optional site-kind label stored in provenance tags (e.g., chip_exo).",
    )
    organism: Optional[OrganismConfig] = None
    citation: Optional[str] = None
    license: Optional[str] = None
    source_url: Optional[str] = None
    source_version: Optional[str] = None
    tags: Dict[str, str] = Field(default_factory=dict)

    @field_validator("source_id")
    @classmethod
    def _check_source_id(cls, v: str) -> str:
        source_id = str(v).strip()
        if not source_id:
            raise ValueError("site source_id must be a non-empty string")
        return source_id

    @field_validator("tf_name", "record_kind")
    @classmethod
    def _check_optional_text(cls, v: Optional[str], info) -> Optional[str]:
        if v is None:
            return None
        value = str(v).strip()
        if not value:
            raise ValueError(f"{info.field_name} must be a non-empty string")
        return value


class RegulonDBConfig(StrictBaseModel):
    base_url: str = "https://regulondb.ccg.unam.mx/graphql"
    verify_ssl: bool = True
    ca_bundle: Optional[Path] = None
    timeout_seconds: int = 30
    motif_matrix_source: Literal["alignment", "sites"] = "alignment"
    alignment_matrix_semantics: Literal["probabilities", "counts"] = "probabilities"
    min_sites_for_pwm: int = 2
    pseudocounts: float = Field(
        0.5,
        description="Pseudocounts for PWM construction from curated sites (Biopython).",
    )
    allow_low_sites: bool = False
    curated_sites: bool = True
    ht_sites: bool = False
    ht_dataset_sources: Optional[List[str]] = None
    ht_dataset_type: Literal["TFBINDING"] = "TFBINDING"
    ht_binding_mode: Literal["tfbinding", "peaks"] = "tfbinding"
    uppercase_binding_site_only: bool = True

    @field_validator("min_sites_for_pwm")
    @classmethod
    def _check_min_sites(cls, v: int) -> int:
        if v < 1:
            raise ValueError("min_sites_for_pwm must be >= 1")
        return v

    @field_validator("pseudocounts")
    @classmethod
    def _check_pseudocounts(cls, v: float) -> float:
        if v < 0:
            raise ValueError("pseudocounts must be >= 0")
        return float(v)


class HttpRetryConfig(StrictBaseModel):
    retries: int = Field(3, description="Number of retry attempts for transient network failures.")
    backoff_seconds: float = Field(0.5, description="Base backoff (seconds) between retries.")
    max_backoff_seconds: float = Field(8.0, description="Maximum backoff between retries.")
    retry_statuses: List[int] = Field(
        default_factory=lambda: [429, 500, 502, 503, 504],
        description="HTTP status codes that trigger retry.",
    )
    respect_retry_after: bool = Field(True, description="Respect Retry-After header when provided.")

    @field_validator("retries")
    @classmethod
    def _check_retries(cls, v: int) -> int:
        if v < 0:
            raise ValueError("retries must be >= 0")
        return v

    @field_validator("backoff_seconds", "max_backoff_seconds")
    @classmethod
    def _check_backoff(cls, v: float) -> float:
        if v < 0:
            raise ValueError("backoff_seconds must be >= 0")
        return float(v)
