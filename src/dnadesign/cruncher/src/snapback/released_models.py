"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/snapback/released_models.py

Contracts for released-product snapback design and target-search workflows.

Module Author(s): Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator

from dnadesign.cruncher.nickases.models import NickEvent, RecognitionSiteInstance, normalize_dna
from dnadesign.cruncher.release_enzymes.models import (
    ReleaseCutEvent,
    ReleaseEnzymeEntry,
    ReleaseRecognitionSiteInstance,
)
from dnadesign.cruncher.snapback.models import (
    EFFECTIVE_CAP_LOOP_NT,
    CatalogNormalizationInfo,
    CatalogSources,
    SnapbackIssue,
    StrictSnapbackModel,
    build_catalog_info,
)

_DEFAULT_DISALLOWED_NICKASE_WARNING_CODES = ("FREQUENT_CUTTER",)


def _normalize_release_catalog_path_list(value: list[Path], *, label: str) -> list[Path]:
    normalized = [Path(path) for path in value]
    if len({str(path) for path in normalized}) != len(normalized):
        raise ValueError(f"{label} must not repeat values.")
    return normalized


def _normalize_warning_code_list(value: list[str], *, label: str) -> list[str]:
    normalized = [str(item or "").strip() for item in value]
    if any(not item for item in normalized):
        raise ValueError(f"{label} must not contain blank values.")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{label} must not repeat values.")
    return normalized


class ReleaseCatalogSources(StrictSnapbackModel):
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
            raise ValueError("release_sources.preset must be non-empty when provided.")
        return text

    @field_validator("additional_presets")
    @classmethod
    def _validate_additional_presets(cls, value: list[str]) -> list[str]:
        normalized = [str(item or "").strip() for item in value]
        if any(not item for item in normalized):
            raise ValueError("release_sources.additional_presets must not contain blank values.")
        if len(set(normalized)) != len(normalized):
            raise ValueError("release_sources.additional_presets must not repeat values.")
        return normalized

    @field_validator("additional_paths")
    @classmethod
    def _validate_additional_paths(cls, value: list[Path]) -> list[Path]:
        return _normalize_release_catalog_path_list(value, label="release_sources.additional_paths")

    @model_validator(mode="after")
    def _validate_sources(self) -> "ReleaseCatalogSources":
        if self.preset is None and not self.additional_presets and not self.additional_paths:
            raise ValueError("release sources must define a preset, an additional preset, or an additional path.")
        preset_ids = self.resolved_preset_ids()
        if len(set(preset_ids)) != len(preset_ids):
            raise ValueError("release sources presets must not repeat values across preset and additional_presets.")
        return self

    def resolved_preset_ids(self) -> list[str]:
        preset_ids: list[str] = []
        if self.preset is not None:
            preset_ids.append(self.preset)
        preset_ids.extend(self.additional_presets)
        return preset_ids


class ReleasedSnapbackHeader(StrictSnapbackModel):
    schema_version: Literal[1] = 1
    kind: Literal["single_nick_released_snapback_v1"] = "single_nick_released_snapback_v1"
    name: str

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError("released_snapback.name must be non-empty.")
        return text


class ReleasedSnapbackInputSpec(StrictSnapbackModel):
    precursor_top_strand: str

    @field_validator("precursor_top_strand")
    @classmethod
    def _validate_precursor_top_strand(cls, value: str) -> str:
        return normalize_dna(value)


class ReleasedNickStageSpec(StrictSnapbackModel):
    nickase_variant_id: str
    catalog: CatalogSources
    intended_site_sequence: str | None = None
    normalized_to_top_strand_nick: bool = True
    require_site_sequence_preserved_pre_nick: bool = True

    @field_validator("nickase_variant_id")
    @classmethod
    def _validate_variant_id(cls, value: str) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError("nick_stage.nickase_variant_id must be non-empty.")
        return text

    @field_validator("intended_site_sequence")
    @classmethod
    def _validate_intended_site_sequence(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return normalize_dna(value)

    @model_validator(mode="after")
    def _validate_preservation(self) -> "ReleasedNickStageSpec":
        if self.require_site_sequence_preserved_pre_nick is not True:
            raise ValueError(
                "nick_stage.require_site_sequence_preserved_pre_nick is reserved in v1 and must remain true."
            )
        return self


class ReleasedReleaseStageSpec(StrictSnapbackModel):
    release_variant_id: str
    catalog: ReleaseCatalogSources
    intended_site_sequence: str | None = None
    retained_side: Literal["upstream"] = "upstream"
    stage_order: Literal["nick_then_release"] = "nick_then_release"
    require_site_sequence_preserved_pre_release: bool = True

    @field_validator("release_variant_id")
    @classmethod
    def _validate_variant_id(cls, value: str) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError("release_stage.release_variant_id must be non-empty.")
        return text

    @field_validator("intended_site_sequence")
    @classmethod
    def _validate_intended_site_sequence(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return normalize_dna(value)

    @model_validator(mode="after")
    def _validate_supported_mode(self) -> "ReleasedReleaseStageSpec":
        if self.require_site_sequence_preserved_pre_release is not True:
            raise ValueError(
                "release_stage.require_site_sequence_preserved_pre_release is reserved in v1 and must remain true."
            )
        return self


class ReleasedFinalTargetGeometry(StrictSnapbackModel):
    nick_boundary_from_left: int = Field(ge=0)
    paired_bp: int = Field(ge=1)
    cap_nt: int = Field(ge=0)

    @model_validator(mode="after")
    def _validate_cap_nt(self) -> "ReleasedFinalTargetGeometry":
        if self.cap_nt != EFFECTIVE_CAP_LOOP_NT:
            raise ValueError(
                f"final_target.cap_nt must equal the fixed snapback effective cap loop size of {EFFECTIVE_CAP_LOOP_NT}."
            )
        return self


class ReleasedSnapbackConstraintsSpec(StrictSnapbackModel):
    allow_post_release_loss_of_nickase_site: bool = True
    allow_post_release_loss_of_release_site: bool = True
    require_nick_survives_in_retained_product: bool = False
    require_release_site_downstream_of_nick: bool = True
    require_complete_downstream_fragment_separation: bool = True
    disallowed_nickase_warning_codes: list[str] = Field(
        default_factory=lambda: list(_DEFAULT_DISALLOWED_NICKASE_WARNING_CODES)
    )

    @field_validator("disallowed_nickase_warning_codes")
    @classmethod
    def _validate_disallowed_nickase_warning_codes(cls, value: list[str]) -> list[str]:
        return _normalize_warning_code_list(value, label="constraints.disallowed_nickase_warning_codes")


class ReleasedSnapbackOutputConfig(StrictSnapbackModel):
    run_dir: Path = Path("outputs/released_design")

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


class SingleNickReleasedSnapbackSpec(StrictSnapbackModel):
    released_snapback: ReleasedSnapbackHeader
    input: ReleasedSnapbackInputSpec
    nick_stage: ReleasedNickStageSpec
    release_stage: ReleasedReleaseStageSpec
    final_target: ReleasedFinalTargetGeometry
    constraints: ReleasedSnapbackConstraintsSpec = Field(default_factory=ReleasedSnapbackConstraintsSpec)
    output: ReleasedSnapbackOutputConfig = Field(default_factory=ReleasedSnapbackOutputConfig)

    @property
    def name(self) -> str:
        return self.released_snapback.name


class ReleaseCatalogNormalizationInfo(StrictSnapbackModel):
    variant_id: str
    display_name: str
    recognition_sequence: str
    recognition_len: int = Field(ge=1)
    top_cut_offset: int
    bottom_cut_offset: int
    class_label: str
    outside_site: bool
    commercial_confidence: str
    warning_codes: list[str] = Field(default_factory=list)
    recommended_5prime_flanking_bases: int | None = Field(default=None, ge=0)
    source_catalog_id: str
    source_url: str | None = None


class ReleasedProductProjection(StrictSnapbackModel):
    final_geometry_source: Literal["exposed_bottom_strand"] = "exposed_bottom_strand"
    precursor_top_strand: str
    precursor_length: int = Field(ge=0)
    nick_coordinate_precursor: int = Field(ge=0)
    release_top_cut_precursor: int = Field(ge=0)
    release_bottom_cut_precursor: int = Field(ge=0)
    retained_top_strand: str
    retained_top_length_nt: int = Field(ge=0)
    active_bottom_strand: str
    active_bottom_strand_span: tuple[int, int]
    active_bottom_length_nt: int = Field(ge=0)
    rebased_nick_boundary: int = Field(ge=0)
    nickase_site_survives_post_release: bool
    release_site_survives_post_release: bool

    @field_validator("precursor_top_strand", "retained_top_strand", "active_bottom_strand")
    @classmethod
    def _validate_sequence(cls, value: str) -> str:
        return normalize_dna(value, allow_empty=True)

    @model_validator(mode="after")
    def _validate_lengths(self) -> "ReleasedProductProjection":
        if self.precursor_length != len(self.precursor_top_strand):
            raise ValueError("projection.precursor_length must match precursor_top_strand length.")
        if self.retained_top_length_nt != len(self.retained_top_strand):
            raise ValueError("projection.retained_top_length_nt must match retained_top_strand length.")
        if self.active_bottom_length_nt != len(self.active_bottom_strand):
            raise ValueError("projection.active_bottom_length_nt must match active_bottom_strand length.")
        start, end = self.active_bottom_strand_span
        if start < 0 or end < start:
            raise ValueError("projection.active_bottom_strand_span must be an ordered non-negative span.")
        if end - start != self.active_bottom_length_nt:
            raise ValueError("projection.active_bottom_strand_span must match active_bottom_length_nt.")
        if (
            self.release_top_cut_precursor > self.precursor_length
            or self.release_bottom_cut_precursor > self.precursor_length
        ):
            raise ValueError("projection release cut boundaries must stay inside precursor length.")
        if self.rebased_nick_boundary > self.active_bottom_length_nt:
            raise ValueError("projection rebased nick boundary must stay inside active bottom length.")
        return self


class ReleasedFinalCandidate(StrictSnapbackModel):
    final_geometry_source: Literal["exposed_bottom_strand"] = "exposed_bottom_strand"
    designed_sequence: str
    input_sequence: str
    foldback_arm: str
    nick_boundary_from_left: int = Field(ge=0)
    paired_bp: int = Field(ge=1)
    cap_nt: int = Field(ge=0)
    source_cap_nt: int = Field(ge=0)
    cap_extension_nt: int = Field(ge=0)
    active_bottom_length_nt: int = Field(ge=0)
    active_bottom_input_length_nt: int = Field(ge=0)
    mismatch_count: int = Field(ge=0)
    mismatch_positions: list[int] = Field(default_factory=list)
    terminal_ligatable_duplex_bp: int = Field(ge=0)
    max_uninterrupted_duplex_bp: int = Field(ge=0)
    extra_nick_event_count: int = Field(ge=0)
    extra_target_strand_nick_count: int = Field(ge=0)
    gc_fraction_added: float = Field(ge=0.0, le=1.0)
    max_homopolymer_run_added: int = Field(ge=0)
    projected_origin_event: NickEvent
    extra_target_strand_nicks: list[NickEvent] = Field(default_factory=list)
    extra_nick_events: list[NickEvent] = Field(default_factory=list)
    post_nick_sequence: str
    nickase_site_survives_post_release: bool
    release_site_survives_post_release: bool

    @field_validator("designed_sequence", "input_sequence", "foldback_arm", "post_nick_sequence")
    @classmethod
    def _validate_sequence(cls, value: str) -> str:
        return normalize_dna(value, allow_empty=True)

    @model_validator(mode="after")
    def _validate_lengths(self) -> "ReleasedFinalCandidate":
        if self.active_bottom_length_nt != len(self.designed_sequence):
            raise ValueError("final candidate active_bottom_length_nt must match designed_sequence length.")
        if self.active_bottom_input_length_nt != len(self.input_sequence):
            raise ValueError("final candidate active_bottom_input_length_nt must match input_sequence length.")
        if self.input_sequence + self.foldback_arm != self.designed_sequence:
            raise ValueError("final candidate designed_sequence must equal input_sequence + foldback_arm.")
        if self.active_bottom_input_length_nt + self.paired_bp != self.active_bottom_length_nt:
            raise ValueError("final candidate active bottom length must equal input length + paired bp.")
        if self.cap_nt != EFFECTIVE_CAP_LOOP_NT:
            raise ValueError(
                f"final candidate cap_nt must equal the fixed effective cap loop size of {EFFECTIVE_CAP_LOOP_NT}."
            )
        if self.cap_extension_nt != 0:
            raise ValueError("released-product v1 keeps cap_extension_nt fixed at 0.")
        if self.source_cap_nt != self.cap_nt:
            raise ValueError("released-product v1 source_cap_nt must equal cap_nt.")
        return self


class ReleasedSnapbackReportMetadata(StrictSnapbackModel):
    schema_version: int = 1
    kind: Literal["single_nick_released_snapback_v1"] = "single_nick_released_snapback_v1"
    coordinate_semantics: Literal["half_open_zero_based_v1"] = "half_open_zero_based_v1"
    boundary_semantics: Literal["closed_zero_based_boundary_v1"] = "closed_zero_based_boundary_v1"
    final_geometry_source: Literal["exposed_bottom_strand"] = "exposed_bottom_strand"
    nick_catalog_source: str
    release_catalog_source: str
    disallowed_nickase_warning_codes: list[str] = Field(default_factory=list)
    final_target: ReleasedFinalTargetGeometry
    nickase_catalog_variants: list[CatalogNormalizationInfo] = Field(default_factory=list)
    release_catalog_variants: list[ReleaseCatalogNormalizationInfo] = Field(default_factory=list)

    @field_validator("disallowed_nickase_warning_codes")
    @classmethod
    def _validate_report_disallowed_nickase_warning_codes(cls, value: list[str]) -> list[str]:
        return _normalize_warning_code_list(value, label="metadata.disallowed_nickase_warning_codes")


class ReleasedSnapbackEvaluationReport(StrictSnapbackModel):
    schema_version: Literal[1] = 1
    workflow: Literal["snapback_released_design"] = "snapback_released_design"
    status: Literal[
        "satisfied",
        "unsatisfied",
        "invalid_catalog",
        "invalid_precursor",
        "no_release_path",
        "post_release_projection_failed",
    ]
    spec_name: str
    workspace_root: str
    spec_path: str
    metadata: ReleasedSnapbackReportMetadata
    issues: list[SnapbackIssue] = Field(default_factory=list)
    pre_nick_site: RecognitionSiteInstance | None = None
    pre_nick_event: NickEvent | None = None
    release_site: ReleaseRecognitionSiteInstance | None = None
    release_event: ReleaseCutEvent | None = None
    projection: ReleasedProductProjection | None = None
    candidate: ReleasedFinalCandidate | None = None
    run_dir: str | None = None


class ReleasedTargetSearchConfig(StrictSnapbackModel):
    require_pre_nick_site_sequence_preserved: bool = True
    require_pre_release_site_sequence_preserved: bool = True
    allow_post_release_loss_of_nickase_site: bool = True
    allow_demo_hits: bool = False
    disallowed_nickase_warning_codes: list[str] = Field(
        default_factory=lambda: list(_DEFAULT_DISALLOWED_NICKASE_WARNING_CODES)
    )
    retained_side: Literal["upstream"] = "upstream"
    stage_order: Literal["nick_then_release"] = "nick_then_release"
    max_results: int = Field(default=8, ge=1, le=64)
    near_boundary_search_limit: int = Field(default=8, ge=0, le=64)

    @model_validator(mode="after")
    def _validate_supported_mode(self) -> "ReleasedTargetSearchConfig":
        if self.require_pre_nick_site_sequence_preserved is not True:
            raise ValueError("search.require_pre_nick_site_sequence_preserved is reserved in v1 and must remain true.")
        if self.require_pre_release_site_sequence_preserved is not True:
            raise ValueError(
                "search.require_pre_release_site_sequence_preserved is reserved in v1 and must remain true."
            )
        return self

    @field_validator("disallowed_nickase_warning_codes")
    @classmethod
    def _validate_search_disallowed_nickase_warning_codes(cls, value: list[str]) -> list[str]:
        return _normalize_warning_code_list(value, label="search.disallowed_nickase_warning_codes")


class SingleNickReleasedTargetSearchRequest(StrictSnapbackModel):
    schema_version: Literal[1] = 1
    kind: Literal["single_nick_released_target_v1"] = "single_nick_released_target_v1"
    target: ReleasedFinalTargetGeometry
    nick_sources: CatalogSources
    release_sources: ReleaseCatalogSources
    search: ReleasedTargetSearchConfig = Field(default_factory=ReleasedTargetSearchConfig)


class ReleasedTargetSearchHit(StrictSnapbackModel):
    rank: int = Field(ge=1)
    hit_kind: Literal["exact", "nearest"]
    nickase_variant_id: str
    release_variant_id: str
    intended_nick_site_orientation: Literal["forward", "reverse"]
    intended_nick_site_sequence: str
    release_site_orientation: Literal["forward", "reverse"]
    release_site_sequence: str
    nick_boundary_from_left: int = Field(ge=0)
    active_bottom_input_length_nt: int = Field(ge=0)
    active_bottom_length_nt: int = Field(ge=0)
    precursor_length_nt: int = Field(ge=0)
    sacrificial_downstream_tail_nt: int = Field(ge=0)
    extra_nick_event_count: int = Field(ge=0)
    extra_target_strand_nick_count: int = Field(ge=0)
    precursor_top_strand: str
    pre_nick_site: RecognitionSiteInstance
    pre_nick_event: NickEvent
    release_site: ReleaseRecognitionSiteInstance
    release_event: ReleaseCutEvent
    nickase: CatalogNormalizationInfo
    release_enzyme: ReleaseCatalogNormalizationInfo
    projection: ReleasedProductProjection
    final_candidate: ReleasedFinalCandidate

    @field_validator("precursor_top_strand")
    @classmethod
    def _validate_precursor_top_strand(cls, value: str) -> str:
        return normalize_dna(value)


class ReleasedTargetSearchMetadata(StrictSnapbackModel):
    schema_version: int = 1
    kind: Literal["single_nick_released_target_v1"] = "single_nick_released_target_v1"
    coordinate_semantics: Literal["half_open_zero_based_v1"] = "half_open_zero_based_v1"
    boundary_semantics: Literal["closed_zero_based_boundary_v1"] = "closed_zero_based_boundary_v1"
    final_geometry_source: Literal["exposed_bottom_strand"] = "exposed_bottom_strand"
    target: ReleasedFinalTargetGeometry
    nick_catalog_source: str
    release_catalog_source: str
    disallowed_nickase_warning_codes: list[str] = Field(default_factory=list)
    evaluated_pair_count: int = Field(ge=0)
    pre_truncation_exact_hit_count: int = Field(ge=0)
    post_truncation_exact_hit_count: int = Field(ge=0)
    pre_truncation_near_hit_count: int = Field(ge=0)
    post_truncation_near_hit_count: int = Field(ge=0)
    blocker_counts: dict[str, int] = Field(default_factory=dict)

    @field_validator("disallowed_nickase_warning_codes")
    @classmethod
    def _validate_target_search_disallowed_nickase_warning_codes(cls, value: list[str]) -> list[str]:
        return _normalize_warning_code_list(value, label="metadata.disallowed_nickase_warning_codes")


class ReleasedTargetSearchReport(StrictSnapbackModel):
    schema_version: Literal[1] = 1
    workflow: Literal["snapback_released_target_search"] = "snapback_released_target_search"
    status: Literal["exact_hits_found", "near_hits_only", "no_hits", "invalid_catalog"]
    workspace_root: str | None = None
    metadata: ReleasedTargetSearchMetadata
    issues: list[SnapbackIssue] = Field(default_factory=list)
    exact_hits: list[ReleasedTargetSearchHit] = Field(default_factory=list)
    near_hits: list[ReleasedTargetSearchHit] = Field(default_factory=list)


class ReleasedSolveOutputConfig(StrictSnapbackModel):
    run_dir: Path = Path("outputs/released_solve")
    materialize_top_k: int = Field(default=8, ge=1, le=64)
    render_format: Literal["png", "svg", "pdf"] = "pdf"
    emit_renders: bool = False

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


class ReleasedSolveHit(StrictSnapbackModel):
    rank: int = Field(ge=1)
    hit_kind: Literal["exact", "nearest"]
    nickase_variant_id: str
    release_variant_id: str
    materialized_run_dir: str
    render_job_path: str | None = None
    rendered_plot_path: str | None = None
    target_search_hit: ReleasedTargetSearchHit


class ReleasedSolveReportMetadata(StrictSnapbackModel):
    schema_version: int = 1
    kind: Literal["single_nick_released_solve_v1"] = "single_nick_released_solve_v1"
    coordinate_semantics: Literal["half_open_zero_based_v1"] = "half_open_zero_based_v1"
    boundary_semantics: Literal["closed_zero_based_boundary_v1"] = "closed_zero_based_boundary_v1"
    final_geometry_source: Literal["exposed_bottom_strand"] = "exposed_bottom_strand"
    target: ReleasedFinalTargetGeometry
    nick_catalog_source: str
    release_catalog_source: str
    disallowed_nickase_warning_codes: list[str] = Field(default_factory=list)
    evaluated_pair_count: int = Field(ge=0)
    available_exact_hit_count: int = Field(ge=0)
    available_near_hit_count: int = Field(ge=0)
    selected_hit_kind: Literal["exact", "nearest"] | None = None
    materialized_hit_count: int = Field(ge=0)
    requested_materialize_top_k: int = Field(ge=1, le=64)
    render_format: Literal["png", "svg", "pdf"]
    emit_renders: bool = False
    blocker_counts: dict[str, int] = Field(default_factory=dict)

    @field_validator("disallowed_nickase_warning_codes")
    @classmethod
    def _validate_solve_disallowed_nickase_warning_codes(cls, value: list[str]) -> list[str]:
        return _normalize_warning_code_list(value, label="metadata.disallowed_nickase_warning_codes")


class ReleasedSolveReport(StrictSnapbackModel):
    schema_version: Literal[1] = 1
    workflow: Literal["snapback_released_solve"] = "snapback_released_solve"
    status: Literal["exact_hits_materialized", "near_hits_materialized", "no_hits", "invalid_catalog"]
    workspace_root: str | None = None
    run_dir: str | None = None
    metadata: ReleasedSolveReportMetadata
    issues: list[SnapbackIssue] = Field(default_factory=list)
    search_report: ReleasedTargetSearchReport
    hits: list[ReleasedSolveHit] = Field(default_factory=list)


def build_release_catalog_info(entry: ReleaseEnzymeEntry) -> ReleaseCatalogNormalizationInfo:
    return ReleaseCatalogNormalizationInfo(
        variant_id=entry.variant_id,
        display_name=entry.display_name,
        recognition_sequence=entry.recognition_sequence,
        recognition_len=entry.recognition_len,
        top_cut_offset=entry.top_cut_offset,
        bottom_cut_offset=entry.bottom_cut_offset,
        class_label=entry.class_label,
        outside_site=entry.outside_site,
        commercial_confidence=entry.commercial_confidence,
        warning_codes=list(entry.warning_codes),
        recommended_5prime_flanking_bases=entry.recommended_5prime_flanking_bases,
        source_catalog_id=entry.source_catalog_id,
        source_url=entry.source_url,
    )


def build_released_nickase_catalog_info(entry) -> CatalogNormalizationInfo:
    return build_catalog_info(entry)


__all__ = [
    "ReleaseCatalogNormalizationInfo",
    "ReleaseCatalogSources",
    "ReleasedFinalCandidate",
    "ReleasedFinalTargetGeometry",
    "ReleasedNickStageSpec",
    "ReleasedProductProjection",
    "ReleasedReleaseStageSpec",
    "ReleasedSnapbackConstraintsSpec",
    "ReleasedSnapbackEvaluationReport",
    "ReleasedSnapbackHeader",
    "ReleasedSnapbackInputSpec",
    "ReleasedSnapbackOutputConfig",
    "ReleasedSnapbackReportMetadata",
    "ReleasedSolveHit",
    "ReleasedSolveOutputConfig",
    "ReleasedSolveReport",
    "ReleasedSolveReportMetadata",
    "ReleasedTargetSearchConfig",
    "ReleasedTargetSearchHit",
    "ReleasedTargetSearchMetadata",
    "ReleasedTargetSearchReport",
    "SingleNickReleasedSnapbackSpec",
    "SingleNickReleasedTargetSearchRequest",
    "build_release_catalog_info",
    "build_released_nickase_catalog_info",
]
