"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/cassette/models.py

Schema and normalized planning contracts for the cassette workflow.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from dnadesign.cruncher.nickases.models import (
    NickaseCatalog,
    NickaseCatalogDocument,
    NickaseCatalogEntry,
    NickEvent,
    RecognitionSiteInstance,
)
from dnadesign.cruncher.nickases.models import (
    iupac_bases_for_symbol as _shared_iupac_bases_for_symbol,
)
from dnadesign.cruncher.nickases.models import (
    motif_matches as _shared_motif_matches,
)
from dnadesign.cruncher.nickases.models import (
    normalize_dna as _shared_normalize_dna,
)
from dnadesign.cruncher.nickases.models import (
    normalize_iupac as _shared_normalize_iupac,
)
from dnadesign.cruncher.nickases.models import (
    reverse_complement as _shared_reverse_complement,
)
from dnadesign.cruncher.nickases.models import (
    reverse_complement_iupac as _shared_reverse_complement_iupac,
)


class StrictCassetteModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


def normalize_dna(value: str, *, allow_empty: bool = False) -> str:
    return _shared_normalize_dna(value, allow_empty=allow_empty)


def normalize_iupac(value: str) -> str:
    return _shared_normalize_iupac(value)


def reverse_complement(sequence: str) -> str:
    return _shared_reverse_complement(sequence)


def reverse_complement_iupac(sequence: str) -> str:
    return _shared_reverse_complement_iupac(sequence)


def iupac_bases_for_symbol(symbol: str) -> set[str]:
    return _shared_iupac_bases_for_symbol(symbol)


def motif_matches(sequence: str, motif: str) -> bool:
    return _shared_motif_matches(sequence, motif)


_normalize_dna = normalize_dna
_normalize_iupac = normalize_iupac

__all__ = [
    "NickEvent",
    "NickWindow",
    "NickaseCatalog",
    "NickaseCatalogDocument",
    "NickaseCatalogEntry",
    "RecognitionSiteInstance",
    "iupac_bases_for_symbol",
    "motif_matches",
    "normalize_dna",
    "normalize_iupac",
    "reverse_complement",
    "reverse_complement_iupac",
]


class NickWindow(StrictCassetteModel):
    start: int = Field(ge=0)
    end: int = Field(ge=0)

    @model_validator(mode="after")
    def _validate_bounds(self) -> "NickWindow":
        if self.end < self.start:
            raise ValueError("nick_window.end must be >= nick_window.start.")
        return self


class BoundedSegmentLength(StrictCassetteModel):
    min: int = Field(ge=0)
    max: int = Field(ge=0)

    @model_validator(mode="after")
    def _validate_bounds(self) -> "BoundedSegmentLength":
        if self.max < self.min:
            raise ValueError("bounded_segment_length.max must be >= bounded_segment_length.min.")
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
    stem3p_arm_mode: Literal["derived_reverse_complement"] = "derived_reverse_complement"

    @field_validator("stem5p_arm", "loop")
    @classmethod
    def _validate_dna(cls, value: str) -> str:
        return _normalize_dna(value)


class ConstructContextSpec(StrictCassetteModel):
    left_flank: str = ""
    right_flank: str = ""

    @field_validator("left_flank", "right_flank")
    @classmethod
    def _validate_dna(cls, value: str) -> str:
        return _normalize_dna(value, allow_empty=True)


class DuplexNickingPlanSpec(StrictCassetteModel):
    target_strand: Literal["primary", "complement"] = "primary"
    left: FlankNickingRequest
    right: FlankNickingRequest
    require_exactly_two_intended_nicks: bool = True
    bounded_segment_length: BoundedSegmentLength | None = None

    @model_validator(mode="after")
    def _validate_tracer_bullet_mode(self) -> "DuplexNickingPlanSpec":
        if not self.require_exactly_two_intended_nicks:
            raise ValueError(
                "UNSUPPORTED_INTENDED_NICK_COUNT_MODE: "
                "nicking.require_exactly_two_intended_nicks must be true in the cassette tracer bullet."
            )
        return self


class SitePolicySpec(StrictCassetteModel):
    forbid_additional_designated_strand_nicks: bool = False
    scan_scope: Literal["requested_variants", "catalog"] = "requested_variants"


class HairpinValidationSpec(StrictCassetteModel):
    require_topological_hairpin: bool = True
    require_energetic_hairpin: bool = False

    @model_validator(mode="after")
    def _validate_tracer_bullet_mode(self) -> "HairpinValidationSpec":
        if not self.require_topological_hairpin:
            raise ValueError(
                "UNSUPPORTED_TOPOLOGICAL_HAIRPIN_MODE: "
                "hairpin_validation.require_topological_hairpin must be true in the cassette tracer bullet."
            )
        if self.require_energetic_hairpin:
            raise ValueError(
                "ENERGETIC_HAIRPIN_VALIDATION_NOT_SUPPORTED: "
                "hairpin_validation.require_energetic_hairpin=true is not supported in the cassette tracer bullet."
            )
        return self


class CassetteCatalogRef(StrictCassetteModel):
    path: Path


class CassetteOutputConfig(StrictCassetteModel):
    run_dir: Path = Path("outputs/cassettes")
    emit_visual_contracts: bool = True
    emit_baserender_jobs: bool = True
    baserender_profiles: list[Literal["duplex_qa", "hairpin_qa"]] = Field(
        default_factory=lambda: ["duplex_qa", "hairpin_qa"]
    )

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

    @field_validator("baserender_profiles")
    @classmethod
    def _validate_profiles(cls, value: list[str]) -> list[str]:
        normalized = [str(item).strip() for item in value]
        if any(not item for item in normalized):
            raise ValueError("output.baserender_profiles must not contain blank values.")
        if len(set(normalized)) != len(normalized):
            raise ValueError("output.baserender_profiles must not repeat values.")
        return normalized

    @model_validator(mode="after")
    def _validate_visual_output_dependencies(self) -> "CassetteOutputConfig":
        if self.emit_baserender_jobs and not self.emit_visual_contracts:
            raise ValueError("output.emit_baserender_jobs requires output.emit_visual_contracts=true.")
        return self


class HairpinCassetteSpec(StrictCassetteModel):
    schema_version: int
    name: str
    topology: HairpinTopologySpec
    construct_context: ConstructContextSpec = Field(default_factory=ConstructContextSpec)
    nicking: DuplexNickingPlanSpec
    site_policy: SitePolicySpec = Field(default_factory=SitePolicySpec)
    hairpin_validation: HairpinValidationSpec = Field(default_factory=HairpinValidationSpec)
    catalog: CassetteCatalogRef
    output: CassetteOutputConfig = Field(default_factory=CassetteOutputConfig)

    @field_validator("schema_version")
    @classmethod
    def _validate_schema_version(cls, value: int) -> int:
        version = int(value)
        if version not in {1, 2}:
            raise ValueError("cassette.schema_version must be 1 or 2.")
        return version

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError("cassette.name must be non-empty.")
        return text

    @property
    def coordinate_semantics(self) -> Literal["legacy_v1", "boundary_inclusive_v2"]:
        return "legacy_v1" if self.schema_version == 1 else "boundary_inclusive_v2"

    @property
    def target_strand_legacy(self) -> Literal["primary_strand", "complement_strand"]:
        return "primary_strand" if self.nicking.target_strand == "primary" else "complement_strand"

    def normalize(self) -> "NormalizedCassetteSpec":
        stem5p = self.topology.stem5p_arm
        loop = self.topology.loop
        stem3p = reverse_complement(stem5p)
        cassette_sequence = f"{stem5p}{loop}{stem3p}"
        pair_map = [PairContract(left=index, right=len(cassette_sequence) - 1 - index) for index in range(len(stem5p))]
        return NormalizedCassetteSpec(
            schema_version=self.schema_version,
            coordinate_semantics=self.coordinate_semantics,
            topology=NormalizedTopology(
                stem5p_arm=stem5p,
                loop=loop,
                stem3p_arm=stem3p,
                pair_map=pair_map,
                cassette_sequence=cassette_sequence,
                cassette_length_nt=len(cassette_sequence),
                stem_length_nt=len(stem5p),
                loop_length_nt=len(loop),
            ),
            construct_context=NormalizedConstructContext(
                left_flank=self.construct_context.left_flank,
                right_flank=self.construct_context.right_flank,
                evaluation_primary_sequence=(
                    f"{self.construct_context.left_flank}{cassette_sequence}{self.construct_context.right_flank}"
                ),
            ),
            nicking=NormalizedNickingSpec(
                target_strand=self.nicking.target_strand,
                left=NormalizedNickingRequest(
                    variant_id=self.nicking.left.nickase,
                    window_start=self.nicking.left.nick_window.start,
                    window_end=self.nicking.left.nick_window.end,
                ),
                right=NormalizedNickingRequest(
                    variant_id=self.nicking.right.nickase,
                    window_start=self.nicking.right.nick_window.start,
                    window_end=self.nicking.right.nick_window.end,
                ),
                require_exactly_two_intended_nicks=self.nicking.require_exactly_two_intended_nicks,
                bounded_segment_length=self.nicking.bounded_segment_length,
            ),
            site_policy=self.site_policy,
            hairpin_validation=self.hairpin_validation,
            output=NormalizedOutputConfig(
                run_dir=self.output.run_dir,
                emit_visual_contracts=self.output.emit_visual_contracts,
                emit_baserender_jobs=self.output.emit_baserender_jobs,
                baserender_profiles=self.output.baserender_profiles,
            ),
        )


class HairpinCassetteSpecDocument(StrictCassetteModel):
    cassette: HairpinCassetteSpec


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


class BoundedNickedSegment(StrictCassetteModel):
    strand: Literal["primary", "complement"]
    start_boundary: int = Field(ge=0)
    end_boundary: int = Field(ge=0)
    length_nt: int = Field(ge=0)

    @model_validator(mode="after")
    def _validate_bounds(self) -> "BoundedNickedSegment":
        if self.end_boundary < self.start_boundary:
            raise ValueError("bounded nicked segment end_boundary must be >= start_boundary.")
        if self.length_nt != self.end_boundary - self.start_boundary:
            raise ValueError("bounded nicked segment length must equal end_boundary - start_boundary.")
        return self


class ValidationIssue(StrictCassetteModel):
    code: str
    message: str
    details: Dict[str, Any] = Field(default_factory=dict)


UnsatReason = ValidationIssue


class CatalogNormalizationInfo(StrictCassetteModel):
    variant_id: str
    specificity_id: str
    motif_top_5to3: str
    motif_len: int = Field(ge=1)
    top_cut_offset: int | None = None
    bottom_cut_offset: int | None = None
    source: str | None = None
    raw_cut_notation: str | None = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CassetteReportMetadata(StrictCassetteModel):
    spec_schema_version: int
    coordinate_semantics: Literal["legacy_v1", "boundary_inclusive_v2"]
    left_flank_length: int = Field(ge=0)
    right_flank_length: int = Field(ge=0)
    evaluation_primary_length: int = Field(ge=0)
    catalog_variants: list[CatalogNormalizationInfo] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    bounded_segment_statement: str = "Reports a bounded nicked segment, not excision."


class CassetteCandidateDesign(StrictCassetteModel):
    cassette_sequence: str
    stem5p_arm: str
    loop: str
    stem3p_arm: str
    target_strand: Literal["primary", "complement"]
    intended_left_site: RecognitionSiteInstance
    intended_right_site: RecognitionSiteInstance
    intended_left_nick: NickEvent
    intended_right_nick: NickEvent
    bounded_nicked_segment: BoundedNickedSegment
    extra_designated_strand_nicks: list[NickEvent] = Field(default_factory=list)
    evaluation_primary_sequence: str
    evaluation_complement_sequence: str
    cassette_length_nt: int = Field(ge=1)
    context_offset: int = Field(ge=0)
    stem5p_span: SpanContract
    loop_span: SpanContract
    stem3p_span: SpanContract
    pair_map: list[PairContract]

    @field_validator(
        "cassette_sequence",
        "stem5p_arm",
        "loop",
        "stem3p_arm",
        "evaluation_primary_sequence",
        "evaluation_complement_sequence",
    )
    @classmethod
    def _validate_sequence(cls, value: str) -> str:
        return _normalize_dna(value)


class CassetteEvaluationReport(StrictCassetteModel):
    schema_version: int = 2
    workflow: Literal["cassette"] = "cassette"
    status: Literal["satisfied", "unsatisfied"]
    spec_name: str
    target_strand: Literal["primary", "complement"]
    workspace_root: str
    spec_path: str
    catalog_path: str
    metadata: CassetteReportMetadata
    issues: list[ValidationIssue] = Field(default_factory=list)
    candidate: CassetteCandidateDesign | None = None
    run_dir: str | None = None


class NormalizedTopology(StrictCassetteModel):
    stem5p_arm: str
    loop: str
    stem3p_arm: str
    pair_map: list[PairContract]
    cassette_sequence: str
    cassette_length_nt: int = Field(ge=1)
    stem_length_nt: int = Field(ge=1)
    loop_length_nt: int = Field(ge=1)


class NormalizedConstructContext(StrictCassetteModel):
    left_flank: str
    right_flank: str
    evaluation_primary_sequence: str

    @property
    def evaluation_complement_sequence(self) -> str:
        return reverse_complement(self.evaluation_primary_sequence)

    @property
    def cassette_start_offset(self) -> int:
        return len(self.left_flank)


class NormalizedNickingRequest(StrictCassetteModel):
    variant_id: str
    window_start: int = Field(ge=0)
    window_end: int = Field(ge=0)

    @model_validator(mode="after")
    def _validate_bounds(self) -> "NormalizedNickingRequest":
        if self.window_end < self.window_start:
            raise ValueError("nicking window_end must be >= window_start.")
        return self


class NormalizedNickingSpec(StrictCassetteModel):
    target_strand: Literal["primary", "complement"]
    left: NormalizedNickingRequest
    right: NormalizedNickingRequest
    require_exactly_two_intended_nicks: bool = True
    bounded_segment_length: BoundedSegmentLength | None = None


class NormalizedOutputConfig(StrictCassetteModel):
    run_dir: Path
    emit_visual_contracts: bool
    emit_baserender_jobs: bool
    baserender_profiles: list[Literal["duplex_qa", "hairpin_qa"]] = Field(default_factory=list)


class NormalizedCassetteSpec(StrictCassetteModel):
    schema_version: int
    coordinate_semantics: Literal["legacy_v1", "boundary_inclusive_v2"]
    topology: NormalizedTopology
    construct_context: NormalizedConstructContext
    nicking: NormalizedNickingSpec
    site_policy: SitePolicySpec
    hairpin_validation: HairpinValidationSpec
    output: NormalizedOutputConfig
