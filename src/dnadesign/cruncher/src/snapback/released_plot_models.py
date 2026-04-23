"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/snapback/released_plot_models.py

Typed plot-context contracts for released-product snapback hit rendering.

Module Author(s): Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from dnadesign.cruncher.nickases.models import normalize_dna
from dnadesign.cruncher.snapback.models import StrictSnapbackModel
from dnadesign.cruncher.snapback.released_route_policy import ReleasedActiveStrand


class PlotSpan(StrictSnapbackModel):
    start: int
    end: int

    @model_validator(mode="after")
    def _validate_bounds(self) -> "PlotSpan":
        if self.end < self.start:
            raise ValueError("plot span end must be greater than or equal to start.")
        return self

    @property
    def width(self) -> int:
        return self.end - self.start


class PlotFragmentRow(StrictSnapbackModel):
    role: Literal["active_product", "retained_partner"]
    strand: ReleasedActiveStrand
    label: str
    sequence: str
    span: PlotSpan
    start_terminal: str | None = None
    end_terminal: str | None = None

    @field_validator("sequence")
    @classmethod
    def _validate_sequence(cls, value: str) -> str:
        return normalize_dna(value, allow_empty=True)

    @model_validator(mode="after")
    def _validate_row(self) -> "PlotFragmentRow":
        if len(self.sequence) != self.span.width:
            raise ValueError("plot fragment row sequence length must match the visible span width.")
        return self


class PlotFoldbackRow(StrictSnapbackModel):
    role: Literal["active_stem", "foldback_return"]
    label: str
    sequence: str
    span: PlotSpan
    left_terminal: str | None = None

    @field_validator("sequence")
    @classmethod
    def _validate_sequence(cls, value: str) -> str:
        return normalize_dna(value, allow_empty=True)

    @model_validator(mode="after")
    def _validate_row(self) -> "PlotFoldbackRow":
        if len(self.sequence) != self.span.width:
            raise ValueError("plot foldback row sequence length must match the visible span width.")
        return self


class PlotLabels(StrictSnapbackModel):
    active_label: str
    partner_label: str
    active_start_terminal: str
    active_end_terminal: str
    partner_start_terminal: str
    partner_end_terminal: str
    orientation_note: str


class PlotTarget(StrictSnapbackModel):
    nick_boundary_from_left: int
    paired_bp: int
    cap_nt: int


class PlotPrecursorPanelContext(StrictSnapbackModel):
    top_sequence: str
    bottom_sequence: str
    nick_site: dict[str, Any]
    nick_event: dict[str, Any]
    nicked_strand: ReleasedActiveStrand
    release_site: dict[str, Any]
    release_event: dict[str, Any]
    top_span: PlotSpan
    bottom_span: PlotSpan
    nick_boundary: int
    nick_site_span: PlotSpan
    release_site_span: PlotSpan
    retained_partner_span: PlotSpan
    active_product_span: PlotSpan
    sacrificial_top_tail_span: PlotSpan
    sacrificial_bottom_tail_span: PlotSpan

    @field_validator("top_sequence", "bottom_sequence")
    @classmethod
    def _validate_sequence(cls, value: str) -> str:
        return normalize_dna(value, allow_empty=True)

    @model_validator(mode="after")
    def _validate_sequences(self) -> "PlotPrecursorPanelContext":
        if len(self.top_sequence) != self.top_span.width:
            raise ValueError("precursor top sequence length must match top span width.")
        if len(self.bottom_sequence) != self.bottom_span.width:
            raise ValueError("precursor bottom sequence length must match bottom span width.")
        return self


class PlotReleasedProductContext(StrictSnapbackModel):
    retained_partner_sequence: str
    active_product_sequence: str
    nick_boundary: int
    release_top_cut_boundary: int
    release_bottom_cut_boundary: int
    upstream_context_span: PlotSpan
    retained_partner_span: PlotSpan
    active_product_span: PlotSpan
    nicked_strand: ReleasedActiveStrand
    top_row: PlotFragmentRow
    bottom_row: PlotFragmentRow
    duplex_overlap_span: PlotSpan | None = None
    duplex_top_sequence: str = ""
    duplex_bottom_sequence: str = ""
    duplex_mismatch_positions: list[int] = Field(default_factory=list)
    top_only_overhang_span: PlotSpan | None = None
    bottom_only_overhang_span: PlotSpan | None = None
    active_prefix_span: PlotSpan
    stem_span: PlotSpan
    cap_span: PlotSpan
    foldback_span: PlotSpan
    nickase_site_survives_post_release: bool
    release_site_survives_post_release: bool

    @field_validator(
        "retained_partner_sequence",
        "active_product_sequence",
        "duplex_top_sequence",
        "duplex_bottom_sequence",
    )
    @classmethod
    def _validate_sequence(cls, value: str) -> str:
        return normalize_dna(value, allow_empty=True)

    @model_validator(mode="after")
    def _validate_panel(self) -> "PlotReleasedProductContext":
        if self.top_row.strand != "top":
            raise ValueError("released-product top_row must stay on the physical top lane.")
        if self.bottom_row.strand != "bottom":
            raise ValueError("released-product bottom_row must stay on the physical bottom lane.")
        if self.duplex_overlap_span is None:
            if self.duplex_top_sequence or self.duplex_bottom_sequence or self.duplex_mismatch_positions:
                raise ValueError("released-product duplex payload must be empty when there is no overlap span.")
            return self
        if len(self.duplex_top_sequence) != self.duplex_overlap_span.width:
            raise ValueError("released-product duplex_top_sequence must match duplex_overlap_span width.")
        if len(self.duplex_bottom_sequence) != self.duplex_overlap_span.width:
            raise ValueError("released-product duplex_bottom_sequence must match duplex_overlap_span width.")
        return self


class PlotFoldbackPanelContext(StrictSnapbackModel):
    origin_boundary_from_left: int
    stem_sequence: str
    cap_sequence: str
    foldback_sequence: str
    foldback_partner_sequence: str
    upstream_context_span: PlotSpan
    nicked_strand: ReleasedActiveStrand
    top_row: PlotFoldbackRow
    bottom_row: PlotFoldbackRow
    foldback_mismatch_positions: list[int] = Field(default_factory=list)

    @field_validator("stem_sequence", "cap_sequence", "foldback_sequence", "foldback_partner_sequence")
    @classmethod
    def _validate_sequence(cls, value: str) -> str:
        return normalize_dna(value, allow_empty=True)

    @model_validator(mode="after")
    def _validate_panel(self) -> "PlotFoldbackPanelContext":
        return self


class ReleasedHitPlotContext(StrictSnapbackModel):
    kind: Literal["released_hit_plot_v1"] = "released_hit_plot_v1"
    labels: PlotLabels
    target: PlotTarget
    nickase_variant_id: str
    release_variant_id: str
    precursor: PlotPrecursorPanelContext
    released_product: PlotReleasedProductContext
    foldback: PlotFoldbackPanelContext


__all__ = [
    "PlotFoldbackPanelContext",
    "PlotFoldbackRow",
    "PlotFragmentRow",
    "PlotLabels",
    "PlotPrecursorPanelContext",
    "PlotReleasedProductContext",
    "PlotSpan",
    "PlotTarget",
    "ReleasedHitPlotContext",
]
