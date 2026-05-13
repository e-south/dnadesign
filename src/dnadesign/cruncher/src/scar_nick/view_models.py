"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/scar_nick/view_models.py

Producer-owned QA view contracts for scar-nick visual artifacts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, model_validator

from dnadesign.cruncher.scar_nick.models import StrictScarNickModel


class ScarNickCoordinateSpan(StrictScarNickModel):
    start: int = Field(ge=0)
    end: int = Field(ge=0)

    @model_validator(mode="after")
    def _validate_span(self) -> "ScarNickCoordinateSpan":
        if self.end < self.start:
            raise ValueError("span end must be >= start")
        return self


class ScarNickRawCoordinateSpan(StrictScarNickModel):
    start: int
    end: int

    @model_validator(mode="after")
    def _validate_span(self) -> "ScarNickRawCoordinateSpan":
        if self.end <= self.start:
            raise ValueError("raw span end must be > start")
        return self


class ScarNickTerminalNickViewV1(StrictScarNickModel):
    version: Literal[1] = 1
    kind: Literal["scar_nick_terminal_nick_v1"] = "scar_nick_terminal_nick_v1"
    view_id: str
    solution_id: str
    candidate_id: str
    rank: int | None = None
    title: str
    state_kind: Literal["pre_terminal_nick", "post_terminal_nick"]
    event_scope: Literal["terminal_nick"] = "terminal_nick"
    coordinate_semantics: Literal["half_open_zero_based_v1"] = "half_open_zero_based_v1"
    boundary_semantics: Literal["closed_zero_based_boundary_v1"] = "closed_zero_based_boundary_v1"
    primary_sequence_5to3: str
    complement_sequence_3to5: str
    terminal_boundary: int = Field(ge=0)
    nick_boundary: int = Field(ge=0)
    retained_product_span: ScarNickCoordinateSpan
    release_site_span: ScarNickCoordinateSpan
    type_iis_offset_span: ScarNickCoordinateSpan | None = None
    retained_scar_span: ScarNickCoordinateSpan
    junction_partner_span: ScarNickCoordinateSpan | None = None
    nickase_site_span: ScarNickCoordinateSpan
    nickase_site_source_span: ScarNickRawCoordinateSpan | None = None
    nickase_site_span_clipped: bool = False
    nick_state: Literal["intact", "nicked"]
    profile_s3s2s1s0: str
    profile_payload_outward: str
    pair_classes: list[dict[str, Any]] = Field(default_factory=list)
    release_placement: dict[str, Any] | None = None
    nickase_placement: dict[str, Any] | None = None
    meta: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_view(self) -> "ScarNickTerminalNickViewV1":
        sequence_length = len(self.primary_sequence_5to3)
        if sequence_length == 0:
            raise ValueError("primary_sequence_5to3 must be non-empty")
        if len(self.complement_sequence_3to5) != sequence_length:
            raise ValueError("complement_sequence_3to5 must match primary sequence length")
        if self.nick_boundary != self.terminal_boundary:
            raise ValueError("scar-nick terminal view requires nick at terminal boundary")
        downstream = self.primary_sequence_5to3[self.terminal_boundary :]
        if any(symbol.upper() != "N" for symbol in downstream):
            raise ValueError("scar-nick terminal view allows only degenerate N symbols downstream of the nick")
        if self.retained_scar_span.end - self.retained_scar_span.start != 4:
            raise ValueError("retained_scar_span must mark the 4-nt retained Type IIS scar")
        if self.terminal_boundary != self.retained_scar_span.end:
            raise ValueError("terminal_boundary must equal retained_scar_span.end")
        if self.junction_partner_span is not None:
            raise ValueError("scar-nick terminal view must not place partner sequence downstream of the nick")
        if self.retained_product_span.start != self.retained_scar_span.start:
            raise ValueError("retained_product_span must start at retained_scar_span.start")
        if self.retained_product_span.end != self.retained_scar_span.end:
            raise ValueError("retained_product_span must terminate at retained_scar_span.end")
        if self.state_kind == "pre_terminal_nick" and self.nick_state != "intact":
            raise ValueError("pre_terminal_nick view requires intact nick state")
        if self.state_kind == "post_terminal_nick" and self.nick_state != "nicked":
            raise ValueError("post_terminal_nick view requires nicked state")
        for label, span in (
            ("retained_product_span", self.retained_product_span),
            ("release_site_span", self.release_site_span),
            ("type_iis_offset_span", self.type_iis_offset_span),
            ("retained_scar_span", self.retained_scar_span),
            ("nickase_site_span", self.nickase_site_span),
        ):
            if span is not None and span.end > sequence_length:
                raise ValueError(f"{label} must stay inside primary sequence")
        if self.nickase_site_source_span is None:
            raise ValueError("scar-nick terminal view requires nickase_site_source_span")
        if self.nickase_site_span_clipped:
            raise ValueError("scar-nick terminal view requires the full nickase site span to be visible")
        return self


class ScarNickViewsManifestEntryV1(StrictScarNickModel):
    name: str
    path: str
    contract_kind: str


class ScarNickRecommendedJobEntryV1(StrictScarNickModel):
    name: str
    path: str


class ScarNickViewsManifestV1(StrictScarNickModel):
    version: Literal[1] = 1
    kind: Literal["scar_nick_views_manifest_v1"] = "scar_nick_views_manifest_v1"
    solution_id: str
    views: list[ScarNickViewsManifestEntryV1]
    recommended_jobs: list[ScarNickRecommendedJobEntryV1] = Field(default_factory=list)
    meta: dict[str, Any] = Field(default_factory=dict)


__all__ = [
    "ScarNickCoordinateSpan",
    "ScarNickRawCoordinateSpan",
    "ScarNickRecommendedJobEntryV1",
    "ScarNickTerminalNickViewV1",
    "ScarNickViewsManifestEntryV1",
    "ScarNickViewsManifestV1",
]
