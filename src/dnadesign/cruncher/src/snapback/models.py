"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/snapback/models.py

Schema and reporting contracts for v2 explicit snapback workflows.

Module Author(s): Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from dnadesign.cruncher.nickases.models import (
    NickaseCatalogEntry,
    NickaseOperationalProfile,
    NickaseSelectionProfile,
    NickEvent,
    RecognitionSiteInstance,
    normalize_dna,
)

EFFECTIVE_CAP_LOOP_NT = 3


class StrictSnapbackModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _normalize_dna(value: str, *, allow_empty: bool = False) -> str:
    return normalize_dna(value, allow_empty=allow_empty)


class CoordinateSpan(StrictSnapbackModel):
    start: int = Field(ge=0)
    end: int = Field(ge=0)

    @model_validator(mode="after")
    def _validate_bounds(self) -> "CoordinateSpan":
        if self.end < self.start:
            raise ValueError("span.end must be >= span.start.")
        return self

    @property
    def length(self) -> int:
        return self.end - self.start


class BoundaryRange(StrictSnapbackModel):
    min: int = Field(ge=0)
    max: int = Field(ge=0)

    @model_validator(mode="after")
    def _validate_bounds(self) -> "BoundaryRange":
        if self.max < self.min:
            raise ValueError("boundary range max must be >= min.")
        return self

    def contains(self, boundary: int) -> bool:
        return self.min <= boundary <= self.max


class BoundedIntRange(StrictSnapbackModel):
    min: int = Field(ge=0)
    max: int = Field(ge=0)

    @model_validator(mode="after")
    def _validate_bounds(self) -> "BoundedIntRange":
        if self.max < self.min:
            raise ValueError("bounded range max must be >= min.")
        return self

    def contains(self, value: int) -> bool:
        return self.min <= value <= self.max


class FractionRange(StrictSnapbackModel):
    min: float = Field(ge=0.0, le=1.0)
    max: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _validate_bounds(self) -> "FractionRange":
        if self.max < self.min:
            raise ValueError("fraction range max must be >= min.")
        return self


class PairContract(StrictSnapbackModel):
    left: int = Field(ge=0)
    right: int = Field(ge=0)

    @model_validator(mode="after")
    def _validate_pair(self) -> "PairContract":
        if self.right <= self.left:
            raise ValueError("pair.right must be > pair.left.")
        return self


class SnapbackHeader(StrictSnapbackModel):
    schema_version: Literal[2] = 2
    contract: Literal["single_nick_snapback_v2"] = "single_nick_snapback_v2"
    name: str

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError("snapback.name must be non-empty.")
        return text


class CanonicalTopStrandSpec(StrictSnapbackModel):
    sequence: str
    protected_region: CoordinateSpan
    pre_nick_duplex_window: CoordinateSpan

    @field_validator("sequence")
    @classmethod
    def _validate_sequence(cls, value: str) -> str:
        return _normalize_dna(value)

    @model_validator(mode="after")
    def _validate_windows(self) -> "CanonicalTopStrandSpec":
        sequence_len = len(self.sequence)
        if self.protected_region.end > sequence_len:
            raise ValueError(
                "input.canonical_top_strand.protected_region must stay inside input.canonical_top_strand.sequence."
            )
        if self.pre_nick_duplex_window.end > sequence_len:
            raise ValueError(
                "input.canonical_top_strand.pre_nick_duplex_window must stay "
                "inside input.canonical_top_strand.sequence."
            )
        return self


class SnapbackInputSpec(StrictSnapbackModel):
    canonical_top_strand: CanonicalTopStrandSpec


class CatalogSources(StrictSnapbackModel):
    preset: str | None = None
    additional_presets: list[str] = Field(default_factory=list)
    additional_paths: list[Path] = Field(default_factory=list)

    @field_validator("preset")
    @classmethod
    def _validate_preset(cls, value: str | None) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            raise ValueError("catalog.preset must be non-empty when provided.")
        return text

    @field_validator("additional_presets")
    @classmethod
    def _validate_additional_presets(cls, value: list[str]) -> list[str]:
        normalized = [str(item or "").strip() for item in value]
        if any(not item for item in normalized):
            raise ValueError("catalog.additional_presets must not contain blank values.")
        if len(set(normalized)) != len(normalized):
            raise ValueError("catalog.additional_presets must not repeat values.")
        return normalized

    @field_validator("additional_paths")
    @classmethod
    def _validate_additional_paths(cls, value: list[Path]) -> list[Path]:
        normalized = [Path(path) for path in value]
        if len({str(path) for path in normalized}) != len(normalized):
            raise ValueError("catalog.additional_paths must not repeat values.")
        return normalized

    @model_validator(mode="after")
    def _validate_sources(self) -> "CatalogSources":
        if self.preset is None and not self.additional_presets and not self.additional_paths:
            raise ValueError("nickase catalog must define a preset, an additional preset, or an additional path.")
        preset_ids = self.resolved_preset_ids()
        if len(set(preset_ids)) != len(preset_ids):
            raise ValueError("catalog presets must not repeat values across preset and additional_presets.")
        return self

    def resolved_preset_ids(self) -> list[str]:
        preset_ids: list[str] = []
        if self.preset is not None:
            preset_ids.append(self.preset)
        preset_ids.extend(self.additional_presets)
        return preset_ids


class SnapbackNickaseSpec(StrictSnapbackModel):
    variant_id: str
    catalog: CatalogSources

    @field_validator("variant_id")
    @classmethod
    def _validate_variant_id(cls, value: str) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError("design.nickase.variant_id must be non-empty.")
        return text


class OrientationPolicySpec(StrictSnapbackModel):
    normalize_to_top_strand_nick: bool = True
    release_direction: Literal["left_to_right_from_nick"] = "left_to_right_from_nick"


class SingleNickGoalSpec(StrictSnapbackModel):
    nick_boundary_window: BoundaryRange


class HomologyPolicySpec(StrictSnapbackModel):
    max_mismatches: int = Field(ge=0, le=2)
    min_paired_bp: int = Field(default=3, ge=1)
    max_paired_bp: int = Field(default=64, ge=1)

    @model_validator(mode="after")
    def _validate_bounds(self) -> "HomologyPolicySpec":
        if self.max_paired_bp < self.min_paired_bp:
            raise ValueError("homology_policy.max_paired_bp must be >= min_paired_bp.")
        return self


class SnapbackTopologySpec(StrictSnapbackModel):
    retained_homology_window: CoordinateSpan
    cap_sequence: str
    foldback_arm: str
    homology_policy: HomologyPolicySpec

    @field_validator("cap_sequence")
    @classmethod
    def _validate_cap_sequence(cls, value: str) -> str:
        normalized = _normalize_dna(value, allow_empty=True)
        if len(normalized) > EFFECTIVE_CAP_LOOP_NT:
            raise ValueError(
                f"design.topology.cap_sequence must be <= {EFFECTIVE_CAP_LOOP_NT} nt because the snapback "
                "effective cap loop is fixed to 3 nt."
            )
        return normalized

    @field_validator("foldback_arm")
    @classmethod
    def _validate_foldback_arm(cls, value: str) -> str:
        return _normalize_dna(value)


class SnapbackConstraintsSpec(StrictSnapbackModel):
    terminal_ligatable_duplex_bp: BoundedIntRange
    max_uninterrupted_duplex_bp: int = Field(ge=0)
    max_added_nt: int = Field(ge=0)
    forbid_additional_target_strand_nicks: bool = False
    forbid_any_additional_nicks: bool = False

    @model_validator(mode="after")
    def _validate_bounds(self) -> "SnapbackConstraintsSpec":
        if self.max_uninterrupted_duplex_bp < self.terminal_ligatable_duplex_bp.min:
            raise ValueError("max_uninterrupted_duplex_bp must be >= terminal_ligatable_duplex_bp.min.")
        return self


class SnapbackSequenceQualitySpec(StrictSnapbackModel):
    gc_fraction: FractionRange | None = None
    max_homopolymer_run: int | None = Field(default=None, ge=1)


class SnapbackDesignSpec(StrictSnapbackModel):
    nickase: SnapbackNickaseSpec
    orientation_policy: OrientationPolicySpec = Field(default_factory=OrientationPolicySpec)
    single_nick_goal: SingleNickGoalSpec
    topology: SnapbackTopologySpec
    constraints: SnapbackConstraintsSpec
    sequence_quality: SnapbackSequenceQualitySpec = Field(default_factory=SnapbackSequenceQualitySpec)


class SnapbackOutputConfig(StrictSnapbackModel):
    run_dir: Path = Path("outputs/design")
    emit_visual_contracts: bool = True
    emit_baserender_jobs: bool = True
    render_format: Literal["png", "svg", "pdf"] = "png"

    @field_validator("run_dir", mode="before")
    @classmethod
    def _validate_run_dir(cls, value: Path | str) -> Path | str:
        raw_text = str(value or "").strip()
        if not raw_text:
            raise ValueError("output.run_dir must be non-empty.")
        path = Path(raw_text)
        if path.is_absolute():
            raise ValueError("output.run_dir must be a relative path inside the workspace.")
        if any(part == ".." for part in path.parts):
            raise ValueError("output.run_dir must not traverse outside the workspace.")
        return raw_text


class SingleNickSnapbackSpec(StrictSnapbackModel):
    snapback: SnapbackHeader
    input: SnapbackInputSpec
    design: SnapbackDesignSpec
    output: SnapbackOutputConfig = Field(default_factory=SnapbackOutputConfig)

    @model_validator(mode="after")
    def _validate_coordinate_bounds(self) -> "SingleNickSnapbackSpec":
        input_length = len(self.input.canonical_top_strand.sequence)
        retained = self.design.topology.retained_homology_window
        if retained.end > input_length:
            raise ValueError(
                "design.topology.retained_homology_window must stay inside input.canonical_top_strand.sequence."
            )
        paired_bp = len(self.design.topology.foldback_arm)
        if paired_bp != retained.length:
            raise ValueError(
                "design.topology.foldback_arm length must equal design.topology.retained_homology_window length."
            )
        homology_policy = self.design.topology.homology_policy
        if not homology_policy.min_paired_bp <= paired_bp <= homology_policy.max_paired_bp:
            raise ValueError("design.topology.foldback_arm length must satisfy homology_policy min/max paired bp.")
        if self.added_nt > self.design.constraints.max_added_nt:
            raise ValueError("design.topology.cap_sequence + foldback_arm exceeds constraints.max_added_nt.")
        if self.output.emit_baserender_jobs and not self.output.emit_visual_contracts:
            raise ValueError("output.emit_baserender_jobs requires output.emit_visual_contracts: true.")
        return self

    @property
    def name(self) -> str:
        return self.snapback.name

    @property
    def input_sequence(self) -> str:
        return self.input.canonical_top_strand.sequence

    @property
    def designed_sequence(self) -> str:
        return f"{self.input_sequence}{self.design.topology.cap_sequence}{self.design.topology.foldback_arm}"

    @property
    def added_nt(self) -> int:
        return len(self.design.topology.cap_sequence) + len(self.design.topology.foldback_arm)

    @property
    def retained_homology_sequence(self) -> str:
        span = self.design.topology.retained_homology_window
        return self.input_sequence[span.start : span.end]


class CatalogNormalizationInfo(StrictSnapbackModel):
    variant_id: str
    specificity_id: str
    motif_top_5to3: str
    motif_len: int = Field(ge=1)
    nicked_strand: Literal["top", "bottom"]
    active_cut_offset: int
    top_cut_offset: int | None = None
    bottom_cut_offset: int | None = None
    source: str | None = None
    vendor: str | None = None
    vendor_catalog_number: str | None = None
    origin_class: str | None = None
    source_family: str | None = None
    notes: list[str] = Field(default_factory=list)
    selection: NickaseSelectionProfile | None = None
    operational: NickaseOperationalProfile | None = None
    raw_cut_notation: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SnapbackIssue(StrictSnapbackModel):
    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class SnapbackReportMetadata(StrictSnapbackModel):
    spec_schema_version: int = 2
    contract: Literal["single_nick_snapback_v2"] = "single_nick_snapback_v2"
    coordinate_semantics: Literal["half_open_zero_based_v1"] = "half_open_zero_based_v1"
    boundary_semantics: Literal["closed_zero_based_boundary_v1"] = "closed_zero_based_boundary_v1"
    input_length_nt: int = Field(ge=0)
    added_nt: int = Field(ge=0)
    designed_length_nt: int = Field(ge=0)
    catalog_source: str
    catalog_presets: list[str] = Field(default_factory=list)
    catalog_variants: list[CatalogNormalizationInfo] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    warning_codes: list[str] = Field(default_factory=list)


class SnapbackCandidateDesign(StrictSnapbackModel):
    designed_sequence: str
    input_sequence: str
    protected_region: CoordinateSpan
    pre_nick_duplex_window: CoordinateSpan
    retained_homology_window: CoordinateSpan
    source_cap_window: CoordinateSpan
    cap_span: CoordinateSpan
    foldback_arm_span: CoordinateSpan
    retained_homology_sequence: str
    released_prefix_sequence: str
    source_cap_sequence: str
    effective_cap_sequence: str
    cap_sequence: str
    foldback_arm: str
    intended_site: RecognitionSiteInstance
    intended_nick: NickEvent
    nick_boundary: int = Field(ge=0)
    nick_boundary_from_left: int = Field(ge=0)
    site_mutation_count: int = Field(ge=0)
    released_prefix_nt: int = Field(ge=0)
    retained_start_from_nick: int = Field(ge=0)
    cap_nt: int = Field(ge=0)
    cap_extension_nt: int = Field(ge=0)
    paired_bp: int = Field(ge=0)
    mismatch_count: int = Field(ge=0)
    mismatch_positions: list[int] = Field(default_factory=list)
    terminal_ligatable_duplex_bp: int = Field(ge=0)
    max_uninterrupted_duplex_bp: int = Field(ge=0)
    added_nt: int = Field(ge=0)
    extra_nick_event_count: int = Field(ge=0)
    gc_fraction_added: float = Field(ge=0.0, le=1.0)
    gc_distance: float = Field(ge=0.0)
    max_homopolymer_run_added: int = Field(ge=0)
    extra_target_strand_nicks: list[NickEvent] = Field(default_factory=list)
    extra_nick_events: list[NickEvent] = Field(default_factory=list)
    post_nick_sequence: str
    post_nick_released_prefix_span: CoordinateSpan
    post_nick_retained_homology_span: CoordinateSpan
    post_nick_source_cap_span: CoordinateSpan
    post_nick_cap_extension_span: CoordinateSpan
    post_nick_cap_span: CoordinateSpan
    post_nick_foldback_arm_span: CoordinateSpan
    pair_map: list[PairContract] = Field(default_factory=list)

    @field_validator(
        "designed_sequence",
        "input_sequence",
        "retained_homology_sequence",
        "released_prefix_sequence",
        "source_cap_sequence",
        "effective_cap_sequence",
        "cap_sequence",
        "foldback_arm",
        "post_nick_sequence",
    )
    @classmethod
    def _validate_sequence(cls, value: str) -> str:
        return _normalize_dna(value, allow_empty=True)

    @model_validator(mode="after")
    def _validate_consistency(self) -> "SnapbackCandidateDesign":
        input_length = len(self.input_sequence)
        designed_length = len(self.designed_sequence)
        if self.protected_region.end > input_length:
            raise ValueError("protected_region must stay inside input_sequence.")
        if self.pre_nick_duplex_window.end > input_length:
            raise ValueError("pre_nick_duplex_window must stay inside input_sequence.")
        if self.retained_homology_window.end > input_length:
            raise ValueError("retained_homology_window must stay inside input_sequence.")
        if self.intended_site.start < 0 or self.intended_site.end > input_length:
            raise ValueError("intended_site must stay inside input_sequence.")
        if self.source_cap_window.start != self.retained_homology_window.end:
            raise ValueError("source_cap_window must start at retained_homology_window.end.")
        if self.source_cap_window.end != input_length:
            raise ValueError("source_cap_window must end at input_sequence length.")
        if self.nick_boundary > input_length:
            raise ValueError("nick_boundary must stay inside input_sequence.")
        if self.site_mutation_count > len(self.intended_site.matched_span_sequence):
            raise ValueError("site_mutation_count must not exceed intended_site length.")
        if self.retained_homology_window.start != self.nick_boundary:
            raise ValueError("retained_homology_window must start exactly at nick_boundary.")
        if self.cap_span.start != input_length:
            raise ValueError("cap_span must start at the end of input_sequence.")
        if self.cap_span.end != self.cap_span.start + len(self.cap_sequence):
            raise ValueError("cap_span length must match cap_sequence.")
        if self.foldback_arm_span.start != self.cap_span.end:
            raise ValueError("foldback_arm_span must start at cap_span.end.")
        if self.foldback_arm_span.end != self.foldback_arm_span.start + len(self.foldback_arm):
            raise ValueError("foldback_arm_span length must match foldback_arm.")
        if self.foldback_arm_span.end != designed_length:
            raise ValueError("designed_sequence must end at foldback_arm_span.end.")
        expected_retained = self.input_sequence[self.retained_homology_window.start : self.retained_homology_window.end]
        if self.retained_homology_sequence != expected_retained:
            raise ValueError("retained_homology_sequence must match retained_homology_window.")
        expected_prefix = self.input_sequence[self.nick_boundary : self.retained_homology_window.start]
        if self.released_prefix_sequence != expected_prefix:
            raise ValueError("released_prefix_sequence must match nick-relative input prefix.")
        expected_source_cap = self.input_sequence[self.retained_homology_window.end : self.source_cap_window.end]
        if self.source_cap_sequence != expected_source_cap:
            raise ValueError("source_cap_sequence must match retained_homology_window.end:input_sequence.end.")
        if self.released_prefix_nt != len(self.released_prefix_sequence):
            raise ValueError("released_prefix_nt must match released_prefix_sequence length.")
        if self.retained_start_from_nick != 0:
            raise ValueError("retained_start_from_nick must be 0 because retained_homology starts at nick_boundary.")
        if self.effective_cap_sequence != f"{self.source_cap_sequence}{self.cap_sequence}":
            raise ValueError("effective_cap_sequence must match source_cap_sequence + cap_sequence.")
        if self.cap_nt != len(self.effective_cap_sequence):
            raise ValueError("cap_nt must match effective_cap_sequence length.")
        if self.cap_extension_nt != len(self.cap_sequence):
            raise ValueError("cap_extension_nt must match cap_sequence length.")
        if self.cap_nt != EFFECTIVE_CAP_LOOP_NT:
            raise ValueError(f"cap_nt must equal the fixed effective cap loop size of {EFFECTIVE_CAP_LOOP_NT}.")
        if self.paired_bp != len(self.foldback_arm):
            raise ValueError("paired_bp must match foldback_arm length.")
        if self.added_nt != len(self.cap_sequence) + len(self.foldback_arm):
            raise ValueError("added_nt must match cap_sequence + foldback_arm length.")
        expected_post_nick = build_post_nick_sequence(
            released_prefix_sequence=self.released_prefix_sequence,
            retained_homology_sequence=self.retained_homology_sequence,
            source_cap_sequence=self.source_cap_sequence,
            cap_sequence=self.cap_sequence,
            foldback_arm=self.foldback_arm,
        )
        if self.post_nick_sequence != expected_post_nick:
            raise ValueError("post_nick_sequence must match released_prefix + retained_homology + cap + foldback.")
        expected_released_span = CoordinateSpan(start=0, end=len(self.released_prefix_sequence))
        expected_retained_span = CoordinateSpan(
            start=expected_released_span.end,
            end=expected_released_span.end + len(self.retained_homology_sequence),
        )
        expected_source_cap_span = CoordinateSpan(
            start=expected_retained_span.end,
            end=expected_retained_span.end + len(self.source_cap_sequence),
        )
        expected_cap_extension_span = CoordinateSpan(
            start=expected_source_cap_span.end,
            end=expected_source_cap_span.end + len(self.cap_sequence),
        )
        expected_cap_span = CoordinateSpan(
            start=expected_retained_span.end,
            end=expected_cap_extension_span.end,
        )
        expected_foldback_span = CoordinateSpan(
            start=expected_cap_span.end,
            end=expected_cap_span.end + len(self.foldback_arm),
        )
        if self.post_nick_released_prefix_span != expected_released_span:
            raise ValueError("post_nick_released_prefix_span must match released_prefix_sequence length.")
        if self.post_nick_retained_homology_span != expected_retained_span:
            raise ValueError("post_nick_retained_homology_span must match retained_homology_sequence length.")
        if self.post_nick_source_cap_span != expected_source_cap_span:
            raise ValueError("post_nick_source_cap_span must match source_cap_sequence length.")
        if self.post_nick_cap_extension_span != expected_cap_extension_span:
            raise ValueError("post_nick_cap_extension_span must match cap_sequence length.")
        if self.post_nick_cap_span != expected_cap_span:
            raise ValueError("post_nick_cap_span must match effective_cap_sequence length.")
        if self.post_nick_foldback_arm_span != expected_foldback_span:
            raise ValueError("post_nick_foldback_arm_span must match foldback_arm length.")
        if expected_foldback_span.end != len(self.post_nick_sequence):
            raise ValueError("post_nick_sequence length must match folded topology spans.")
        if self.mismatch_count != len(self.mismatch_positions):
            raise ValueError("mismatch_count must match mismatch_positions length.")
        if any(
            position < 0 or position >= len(self.retained_homology_sequence) for position in self.mismatch_positions
        ):
            raise ValueError("mismatch_positions must be retained-homology-local indices.")
        for pair in self.pair_map:
            if not (
                self.post_nick_retained_homology_span.start <= pair.left < self.post_nick_retained_homology_span.end
            ):
                raise ValueError("pair_map left indices must stay inside post_nick_retained_homology_span.")
            if not (self.post_nick_foldback_arm_span.start <= pair.right < self.post_nick_foldback_arm_span.end):
                raise ValueError("pair_map right indices must stay inside post_nick_foldback_arm_span.")
        return self


class SnapbackEvaluationReport(StrictSnapbackModel):
    schema_version: Literal[2] = 2
    workflow: Literal["snapback"] = "snapback"
    status: Literal["satisfied", "unsatisfied", "invalid_catalog"]
    spec_name: str
    workspace_root: str
    spec_path: str
    catalog_source: str
    metadata: SnapbackReportMetadata
    issues: list[SnapbackIssue] = Field(default_factory=list)
    candidate: SnapbackCandidateDesign | None = None
    run_dir: str | None = None


def build_catalog_info(entry: NickaseCatalogEntry) -> CatalogNormalizationInfo:
    return CatalogNormalizationInfo(
        variant_id=entry.id,
        specificity_id=entry.specificity_id,
        motif_top_5to3=entry.motif_top_5to3,
        motif_len=entry.motif_len or len(entry.motif_top_5to3),
        nicked_strand=entry.nicked_strand,
        active_cut_offset=entry.active_cut_offset,
        top_cut_offset=entry.top_cut_offset,
        bottom_cut_offset=entry.bottom_cut_offset,
        source=entry.source,
        vendor=entry.vendor,
        vendor_catalog_number=entry.vendor_catalog_number,
        origin_class=entry.origin_class,
        source_family=entry.source_family,
        notes=list(entry.notes),
        selection=entry.selection,
        operational=entry.operational,
        raw_cut_notation=entry.raw_cut_notation,
        metadata=entry.metadata,
    )


def gc_fraction(sequence: str) -> float:
    if not sequence:
        return 0.0
    gc_count = sum(1 for base in sequence if base in {"G", "C"})
    return gc_count / len(sequence)


def gc_distance_for_range(sequence: str, bounds: FractionRange | None) -> float:
    observed = gc_fraction(sequence)
    if bounds is None:
        return 0.0
    if observed < bounds.min:
        return bounds.min - observed
    if observed > bounds.max:
        return observed - bounds.max
    return 0.0


def max_homopolymer_run(sequence: str) -> int:
    if not sequence:
        return 0
    longest = 1
    current = 1
    for left, right in zip(sequence, sequence[1:], strict=False):
        if left == right:
            current += 1
            longest = max(longest, current)
        else:
            current = 1
    return longest


def build_post_nick_sequence(
    *,
    released_prefix_sequence: str,
    retained_homology_sequence: str,
    source_cap_sequence: str,
    cap_sequence: str,
    foldback_arm: str,
) -> str:
    return f"{released_prefix_sequence}{retained_homology_sequence}{source_cap_sequence}{cap_sequence}{foldback_arm}"
