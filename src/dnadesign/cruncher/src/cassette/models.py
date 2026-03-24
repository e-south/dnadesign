"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/cassette/models.py

Schema contracts for the dual-context cassette workflow.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_DNA_RE = re.compile(r"^[ACGT]+$")


class StrictCassetteModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _normalize_dna(value: str, *, allow_empty: bool = False) -> str:
    text = str(value or "").strip().upper()
    if not text:
        if allow_empty:
            return ""
        raise ValueError("DNA sequence cannot be empty.")
    if not _DNA_RE.fullmatch(text):
        raise ValueError(f"DNA sequence must contain only A/C/G/T: {value!r}")
    return text


def reverse_complement(sequence: str) -> str:
    return sequence.upper().translate(str.maketrans("ACGT", "TGCA"))[::-1]


class NickWindow(StrictCassetteModel):
    start: int = Field(ge=0)
    end: int = Field(ge=0)

    @model_validator(mode="after")
    def _validate_bounds(self) -> "NickWindow":
        if self.end < self.start:
            raise ValueError("nick_window.end must be >= nick_window.start.")
        return self


class FlankNickingRequest(StrictCassetteModel):
    nickase: str
    nick_window: NickWindow

    @field_validator("nickase")
    @classmethod
    def _validate_nickase(cls, value: str) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError("nicking nickase ids must be non-empty.")
        return text


class HairpinTopologySpec(StrictCassetteModel):
    stem5p_arm: str
    loop: str
    stem3p_arm_mode: Literal["derive_reverse_complement"] = "derive_reverse_complement"

    @field_validator("stem5p_arm", "loop")
    @classmethod
    def _validate_dna(cls, value: str, info) -> str:
        allow_empty = info.field_name == "loop" and False
        return _normalize_dna(value, allow_empty=allow_empty)


class DuplexContextSpec(StrictCassetteModel):
    upstream: str = ""
    downstream: str = ""

    @field_validator("upstream", "downstream")
    @classmethod
    def _validate_dna(cls, value: str) -> str:
        return _normalize_dna(value, allow_empty=True)


class DuplexNickingPlanSpec(StrictCassetteModel):
    designated_strand: Literal["primary_strand", "complement_strand"] = "primary_strand"
    left: FlankNickingRequest
    right: FlankNickingRequest
    forbid_additional_designated_strand_nicks: bool = False


class CassetteCatalogRef(StrictCassetteModel):
    path: Path


class CassetteOutputConfig(StrictCassetteModel):
    run_dir: Path = Path("outputs/cassettes")
    write_render_contract: bool = True

    @field_validator("run_dir")
    @classmethod
    def _validate_run_dir(cls, value: Path) -> Path:
        path = Path(value)
        if path.is_absolute():
            raise ValueError("output.run_dir must be a relative path inside the workspace.")
        if any(part == ".." for part in path.parts):
            raise ValueError("output.run_dir must not traverse outside the workspace.")
        if not str(path).strip():
            raise ValueError("output.run_dir must be non-empty.")
        return path


class HairpinCassetteSpec(StrictCassetteModel):
    schema_version: int = 1
    name: str
    topology: HairpinTopologySpec
    duplex_context: DuplexContextSpec = Field(default_factory=DuplexContextSpec)
    nicking: DuplexNickingPlanSpec
    catalog: CassetteCatalogRef
    output: CassetteOutputConfig = Field(default_factory=CassetteOutputConfig)

    @field_validator("schema_version")
    @classmethod
    def _validate_schema_version(cls, value: int) -> int:
        if int(value) != 1:
            raise ValueError("cassette.schema_version must be 1.")
        return int(value)

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError("cassette.name must be non-empty.")
        return text


class HairpinCassetteSpecDocument(StrictCassetteModel):
    cassette: HairpinCassetteSpec


class NickaseCatalogEntry(StrictCassetteModel):
    id: str
    recognition_sequence: str
    nicked_site_strand: Literal["forward", "reverse"]
    cut_offset: int = Field(ge=0)
    source: Optional[str] = None
    vendor: Optional[str] = None
    notes: Optional[str] = None
    tags: Dict[str, str] = Field(default_factory=dict)

    @field_validator("id")
    @classmethod
    def _validate_id(cls, value: str) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError("nickase id must be non-empty.")
        return text

    @field_validator("recognition_sequence")
    @classmethod
    def _validate_recognition_sequence(cls, value: str) -> str:
        text = _normalize_dna(value)
        if reverse_complement(text) == text:
            raise ValueError(
                "recognition_sequence must be asymmetric for v1 nickase catalogs; palindromic sites are ambiguous."
            )
        return text

    @model_validator(mode="after")
    def _validate_cut_offset(self) -> "NickaseCatalogEntry":
        site_len = len(self.recognition_sequence)
        if self.cut_offset > site_len:
            raise ValueError("cut_offset must be between 0 and the recognition sequence length.")
        return self


class NickaseCatalog(StrictCassetteModel):
    schema_version: int = 1
    entries: list[NickaseCatalogEntry]

    @field_validator("schema_version")
    @classmethod
    def _validate_schema_version(cls, value: int) -> int:
        if int(value) != 1:
            raise ValueError("nickases.schema_version must be 1.")
        return int(value)

    @field_validator("entries")
    @classmethod
    def _validate_entries(cls, value: list[NickaseCatalogEntry]) -> list[NickaseCatalogEntry]:
        if not value:
            raise ValueError("nickase catalog must define at least one entry.")
        ids = [entry.id for entry in value]
        if len(set(ids)) != len(ids):
            raise ValueError("nickase catalog ids must be unique.")
        return value

    def by_id(self) -> Dict[str, NickaseCatalogEntry]:
        return {entry.id: entry for entry in self.entries}


class NickaseCatalogDocument(StrictCassetteModel):
    nickases: NickaseCatalog


class SpanContract(StrictCassetteModel):
    start: int = Field(ge=0)
    end: int = Field(ge=0)

    @model_validator(mode="after")
    def _validate_bounds(self) -> "SpanContract":
        if self.end < self.start:
            raise ValueError("span.end must be >= span.start.")
        return self


class PairContract(StrictCassetteModel):
    left: int = Field(ge=0)
    right: int = Field(ge=0)

    @model_validator(mode="after")
    def _validate_pair(self) -> "PairContract":
        if self.right <= self.left:
            raise ValueError("pair.right must be > pair.left.")
        return self


class PlannedNick(StrictCassetteModel):
    nickase: str
    recognition_sequence: str
    site_start: int
    site_end: int
    site_orientation: Literal["forward", "reverse"]
    nicked_strand: Literal["primary_strand", "complement_strand"]
    nick_coordinate: int
    nick_coordinate_context: int


class BoundedSegment(StrictCassetteModel):
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    length: int = Field(ge=0)

    @model_validator(mode="after")
    def _validate_bounds(self) -> "BoundedSegment":
        if self.end < self.start:
            raise ValueError("bounded segment end must be >= start.")
        if self.length != self.end - self.start:
            raise ValueError("bounded segment length must equal end - start.")
        return self


class UnsatReason(StrictCassetteModel):
    code: str
    message: str
    details: Dict[str, Any] = Field(default_factory=dict)


class CassetteCandidateDesign(StrictCassetteModel):
    cassette_sequence: str
    context_sequence: str
    complement_sequence: str
    cassette_length: int = Field(ge=1)
    context_offset: int = Field(ge=0)
    stem5p_span: SpanContract
    loop_span: SpanContract
    stem3p_span: SpanContract
    pair_map: list[PairContract]
    left_nick: PlannedNick
    right_nick: PlannedNick
    bounded_segment: BoundedSegment
    additional_designated_strand_nicks: list[PlannedNick] = Field(default_factory=list)

    @field_validator("cassette_sequence", "context_sequence", "complement_sequence")
    @classmethod
    def _validate_sequence(cls, value: str) -> str:
        return _normalize_dna(value)


class CassetteEvaluationReport(StrictCassetteModel):
    schema_version: int = 1
    workflow: Literal["cassette"] = "cassette"
    status: Literal["satisfied", "unsatisfied"]
    spec_name: str
    designated_strand: Literal["primary_strand", "complement_strand"]
    workspace_root: str
    spec_path: str
    catalog_path: str
    issues: list[UnsatReason] = Field(default_factory=list)
    candidate: CassetteCandidateDesign | None = None
    render_contract: Dict[str, Any] | None = None
    run_dir: str | None = None
