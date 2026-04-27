"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/snapback/released_projection_models.py

Projection and final-candidate contracts for released-product snapback.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator, model_validator

from dnadesign.cruncher.nickases.models import NickEvent, normalize_dna
from dnadesign.cruncher.release_enzymes.models import ReleaseEnzymeEntry
from dnadesign.cruncher.snapback.models import (
    EFFECTIVE_CAP_LOOP_NT,
    CatalogNormalizationInfo,
    StrictSnapbackModel,
    build_catalog_info,
)
from dnadesign.cruncher.snapback.released_route_policy import (
    ReleasedActiveStrand,
    ReleasedFinalGeometrySource,
    ReleasedRouteFamily,
    route_family_active_strand,
    route_family_final_geometry_source,
    route_family_physical_nicked_strand,
)


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


class ReleasedProductBaseProvenance(StrictSnapbackModel):
    active_index: int = Field(ge=0)
    precursor_strand: ReleasedActiveStrand
    precursor_index: int = Field(ge=0)
    source_constraint: Literal["fixed_motif_base", "degenerate_motif_base", "user_sequence"]


class ReleasedProductProjection(StrictSnapbackModel):
    final_geometry_source: ReleasedFinalGeometrySource = "exposed_bottom_strand"
    route_family: ReleasedRouteFamily = "bottom_active_from_top_nick"
    physical_nicked_strand: ReleasedActiveStrand = "top"
    active_strand: ReleasedActiveStrand = "bottom"
    retained_partner_strand: ReleasedActiveStrand = "top"
    precursor_top_strand: str
    precursor_length: int = Field(ge=0)
    nick_coordinate_precursor: int = Field(ge=0)
    release_top_cut_precursor: int = Field(ge=0)
    release_bottom_cut_precursor: int = Field(ge=0)
    retained_partner_sequence: str = ""
    retained_partner_length_nt: int = Field(default=0, ge=0)
    active_product_sequence: str = ""
    active_product_span: tuple[int, int] = (0, 0)
    active_product_length_nt: int = Field(default=0, ge=0)
    active_product_provenance: list[ReleasedProductBaseProvenance] = Field(default_factory=list)
    rebased_nick_boundary: int = Field(ge=0)
    nickase_site_survives_post_release: bool
    release_site_survives_post_release: bool

    @field_validator("precursor_top_strand", "retained_partner_sequence", "active_product_sequence")
    @classmethod
    def _validate_sequence(cls, value: str) -> str:
        return normalize_dna(value, allow_empty=True)

    @model_validator(mode="after")
    def _validate_lengths(self) -> "ReleasedProductProjection":
        if self.precursor_length != len(self.precursor_top_strand):
            raise ValueError("projection.precursor_length must match precursor_top_strand length.")
        if self.retained_partner_length_nt != len(self.retained_partner_sequence):
            raise ValueError("projection.retained_partner_length_nt must match retained_partner_sequence length.")
        if self.active_product_length_nt != len(self.active_product_sequence):
            raise ValueError("projection.active_product_length_nt must match active_product_sequence length.")
        active_start, active_end = self.active_product_span
        if active_start < 0 or active_end < active_start:
            raise ValueError("projection.active_product_span must be an ordered non-negative span.")
        if active_end - active_start != self.active_product_length_nt:
            raise ValueError("projection.active_product_span must match active_product_length_nt.")
        if (
            self.release_top_cut_precursor > self.precursor_length
            or self.release_bottom_cut_precursor > self.precursor_length
        ):
            raise ValueError("projection release cut boundaries must stay inside precursor length.")
        if self.rebased_nick_boundary > self.active_product_length_nt:
            raise ValueError("projection rebased nick boundary must stay inside active product length.")
        if route_family_active_strand(self.route_family) != self.active_strand:
            raise ValueError("projection route_family must match active_strand.")
        if route_family_physical_nicked_strand(self.route_family) != self.physical_nicked_strand:
            raise ValueError("projection route_family must match physical_nicked_strand.")
        if route_family_final_geometry_source(self.route_family) != self.final_geometry_source:
            raise ValueError("projection route_family must match final_geometry_source.")
        if self.active_strand == self.retained_partner_strand:
            raise ValueError("projection active_strand and retained_partner_strand must differ.")
        if self.active_strand == "bottom":
            if self.final_geometry_source != "exposed_bottom_strand":
                raise ValueError("projection final_geometry_source drift detected for bottom-active route.")
        elif self.final_geometry_source != "retained_active_strand":
            raise ValueError("projection final_geometry_source drift detected for top-active route.")
        return self


class ReleasedFinalCandidate(StrictSnapbackModel):
    final_geometry_source: ReleasedFinalGeometrySource = "exposed_bottom_strand"
    route_family: ReleasedRouteFamily = "bottom_active_from_top_nick"
    physical_nicked_strand: ReleasedActiveStrand = "top"
    active_strand: ReleasedActiveStrand = "bottom"
    designed_sequence: str
    input_sequence: str
    foldback_arm: str
    nick_boundary_from_left: int = Field(ge=0)
    paired_bp: int = Field(ge=1)
    cap_nt: int = Field(ge=0)
    source_cap_nt: int = Field(ge=0)
    cap_extension_nt: int = Field(ge=0)
    active_product_length_nt: int = Field(default=0, ge=0)
    active_product_input_length_nt: int = Field(default=0, ge=0)
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
    active_product_provenance: list[ReleasedProductBaseProvenance] = Field(default_factory=list)

    @field_validator("designed_sequence", "input_sequence", "foldback_arm", "post_nick_sequence")
    @classmethod
    def _validate_sequence(cls, value: str) -> str:
        return normalize_dna(value, allow_empty=True)

    @model_validator(mode="after")
    def _validate_lengths(self) -> "ReleasedFinalCandidate":
        if self.active_product_length_nt != len(self.designed_sequence):
            raise ValueError("final candidate active_product_length_nt must match designed_sequence length.")
        if self.active_product_input_length_nt != len(self.input_sequence):
            raise ValueError("final candidate active_product_input_length_nt must match input_sequence length.")
        if self.input_sequence + self.foldback_arm != self.designed_sequence:
            raise ValueError("final candidate designed_sequence must equal input_sequence + foldback_arm.")
        if self.active_product_input_length_nt + self.paired_bp != self.active_product_length_nt:
            raise ValueError("final candidate active product length must equal input length + paired bp.")
        if self.cap_nt != EFFECTIVE_CAP_LOOP_NT:
            raise ValueError(
                f"final candidate cap_nt must equal the fixed effective cap loop size of {EFFECTIVE_CAP_LOOP_NT}."
            )
        if self.cap_extension_nt != 0:
            raise ValueError("released-product v1 keeps cap_extension_nt fixed at 0.")
        if self.source_cap_nt != self.cap_nt:
            raise ValueError("released-product v1 source_cap_nt must equal cap_nt.")
        if route_family_active_strand(self.route_family) != self.active_strand:
            raise ValueError("final candidate route_family must match active_strand.")
        if route_family_physical_nicked_strand(self.route_family) != self.physical_nicked_strand:
            raise ValueError("final candidate route_family must match physical_nicked_strand.")
        if route_family_final_geometry_source(self.route_family) != self.final_geometry_source:
            raise ValueError("final candidate route_family must match final_geometry_source.")
        if self.active_strand == "bottom":
            if self.final_geometry_source != "exposed_bottom_strand":
                raise ValueError("final candidate final_geometry_source drift detected for bottom-active route.")
        elif self.final_geometry_source != "retained_active_strand":
            raise ValueError("final candidate final_geometry_source drift detected for top-active route.")
        return self


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
    "ReleasedFinalCandidate",
    "ReleasedProductBaseProvenance",
    "ReleasedProductProjection",
    "build_release_catalog_info",
    "build_released_nickase_catalog_info",
]
