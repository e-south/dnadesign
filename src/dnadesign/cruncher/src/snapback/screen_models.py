"""
Screen-level ontology for the released-product Snapback study objective.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator

from dnadesign.cruncher.snapback.models import CoordinateSpan, StrictSnapbackModel
from dnadesign.cruncher.snapback.released_route_policy import ReleasedActiveStrand, ReleasedRouteFamily
from dnadesign.cruncher.snapback.released_search_models import ReleasedTargetSearchReport

SnapbackMechanismClass = Literal[
    "degenerate_footprint_snapback",
    "fixed_footprint_plus_release_trim",
    "mixed_footprint_payload",
    "comparison_visual_only",
]


class CoordinateFrameTransform(StrictSnapbackModel):
    source_frame: str
    target_frame: str
    source_span: CoordinateSpan
    target_span: CoordinateSpan
    orientation: Literal["forward", "reverse"] = "forward"
    label: str

    @field_validator("source_frame", "target_frame", "label")
    @classmethod
    def _validate_text(cls, value: str) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError("coordinate frame transform text fields must be non-empty.")
        return text


class SnapbackScreenTargetTopology(StrictSnapbackModel):
    logical_origin: int = Field(default=0, ge=0)
    stem_bp: int = Field(default=3, ge=1)
    cap_nt: int = Field(default=3, ge=0)
    retained_product_strands: list[ReleasedActiveStrand] = Field(default_factory=lambda: ["top", "bottom"])
    allow_oriented_vendor_footprints: bool = True
    allow_degenerate_motif_assignment: bool = True
    allow_release_trim_after_foldback_return: bool = True
    require_wc_stem_pairing: bool = True
    require_complete_downstream_fragment_separation: bool = True

    @field_validator("retained_product_strands")
    @classmethod
    def _validate_retained_product_strands(cls, value: list[ReleasedActiveStrand]) -> list[ReleasedActiveStrand]:
        if not value:
            raise ValueError("target_topology.retained_product_strands must not be empty.")
        if len(set(value)) != len(value):
            raise ValueError("target_topology.retained_product_strands must not repeat values.")
        return value


class SnapbackMechanismLedgerEntry(StrictSnapbackModel):
    rank: int = Field(ge=1)
    hit_kind: Literal["exact", "nearest"]
    nickase_variant_id: str
    release_variant_id: str
    route_family: ReleasedRouteFamily
    physical_nicked_strand: ReleasedActiveStrand
    retained_product_strand: ReleasedActiveStrand
    oriented_nick_footprint: str
    oriented_nick_footprint_orientation: Literal["forward", "reverse"]
    oriented_release_footprint: str
    oriented_release_footprint_orientation: Literal["forward", "reverse"]
    logical_origin: int = Field(ge=0)
    logical_stem_bp: int = Field(ge=1)
    cap_nt: int = Field(ge=0)
    logical_stem_span: CoordinateSpan
    logical_cap_span: CoordinateSpan
    logical_foldback_return_span: CoordinateSpan
    upstream_retained_duplex_bp: int = Field(ge=0)
    effective_foldback_pairing_bp: int = Field(ge=0)
    release_terminal_boundary: int = Field(ge=0)
    mechanism_class: SnapbackMechanismClass
    provenance_counts: dict[str, int] = Field(default_factory=dict)
    foldback_mismatch_count: int = Field(ge=0)
    frame_transforms: list[CoordinateFrameTransform] = Field(default_factory=list)

    @field_validator(
        "nickase_variant_id", "release_variant_id", "oriented_nick_footprint", "oriented_release_footprint"
    )
    @classmethod
    def _validate_text(cls, value: str) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError("mechanism ledger text fields must be non-empty.")
        return text

    @field_validator("provenance_counts")
    @classmethod
    def _validate_provenance_counts(cls, value: dict[str, int]) -> dict[str, int]:
        for key, count in value.items():
            if not str(key).strip():
                raise ValueError("provenance_counts keys must be non-empty.")
            if count < 0:
                raise ValueError("provenance_counts values must be non-negative.")
        return value


class SnapbackScreenReport(StrictSnapbackModel):
    kind: Literal["snapback_screen_report_v1"] = "snapback_screen_report_v1"
    status: Literal["exact_hits_found", "near_hits_only", "no_hits", "invalid_catalog"]
    workspace_root: str | None = None
    target_topology: SnapbackScreenTargetTopology
    exact_hit_count: int = Field(ge=0)
    near_hit_count: int = Field(ge=0)
    mechanism_ledger: list[SnapbackMechanismLedgerEntry] = Field(default_factory=list)
    search_report: ReleasedTargetSearchReport


__all__ = [
    "CoordinateFrameTransform",
    "SnapbackMechanismClass",
    "SnapbackMechanismLedgerEntry",
    "SnapbackScreenReport",
    "SnapbackScreenTargetTopology",
]
