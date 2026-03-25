"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/contracts/visual/linear_duplex_v1.py

Shared linear duplex visual contract for cassette QA rendering.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, model_validator

from .common import CoordinateSpan, RenderLabel, VisualContractModel


class RowLabels(VisualContractModel):
    primary: str
    complement: str


class SegmentAnnotation(VisualContractModel):
    id: str
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    semantic: str
    label: str | None = None

    @model_validator(mode="after")
    def _validate_bounds(self) -> "SegmentAnnotation":
        if self.end <= self.start:
            raise ValueError("segment end must be > start")
        return self


class SiteInstanceAnnotation(VisualContractModel):
    id: str
    variant_id: str
    specificity_id: str
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    orientation: Literal["forward", "reverse"]
    intent: Literal["intended_left", "intended_right", "extra"]
    label: str | None = None
    site_target_strand: Literal["primary", "complement"]

    @model_validator(mode="after")
    def _validate_bounds(self) -> "SiteInstanceAnnotation":
        if self.end <= self.start:
            raise ValueError("site instance end must be > start")
        return self


class NickEventAnnotation(VisualContractModel):
    id: str
    boundary: int = Field(ge=0)
    target_strand: Literal["primary", "complement"]
    source_site_id: str
    intent: Literal["intended_left", "intended_right", "extra"]
    label: str | None = None


class BoundedSegmentAnnotation(VisualContractModel):
    start_boundary: int = Field(ge=0)
    end_boundary: int = Field(ge=0)
    target_strand: Literal["primary", "complement"]
    label: str | None = None

    @model_validator(mode="after")
    def _validate_bounds(self) -> "BoundedSegmentAnnotation":
        if self.end_boundary <= self.start_boundary:
            raise ValueError("bounded segment end_boundary must be > start_boundary")
        return self


class LinearDuplexViewV1(VisualContractModel):
    version: Literal[1] = 1
    kind: Literal["linear_duplex_v1"] = "linear_duplex_v1"
    view_id: str
    solution_id: str
    title: str
    coordinate_semantics: Literal["legacy_v1", "boundary_inclusive_v2"]
    primary_sequence_5to3: str
    sequence_span: CoordinateSpan
    cassette_span: CoordinateSpan
    row_labels: RowLabels
    target_strand: Literal["primary", "complement"]
    segments: list[SegmentAnnotation] = Field(default_factory=list)
    site_instances: list[SiteInstanceAnnotation] = Field(default_factory=list)
    nick_events: list[NickEventAnnotation] = Field(default_factory=list)
    bounded_segment: BoundedSegmentAnnotation | None = None
    labels: list[RenderLabel] = Field(default_factory=list)
    meta: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_sequence(self) -> "LinearDuplexViewV1":
        if not self.primary_sequence_5to3:
            raise ValueError("primary_sequence_5to3 must be non-empty")
        sequence_length = len(self.primary_sequence_5to3)
        if self.sequence_span.end > sequence_length:
            raise ValueError("sequence_span exceeds primary sequence length")
        if self.cassette_span.start < self.sequence_span.start or self.cassette_span.end > self.sequence_span.end:
            raise ValueError("cassette_span must stay inside sequence_span")
        prior_segment_end: int | None = None
        if self.cassette_span.end > sequence_length:
            raise ValueError("cassette_span exceeds primary sequence length")
        for segment in self.segments:
            if segment.end > sequence_length:
                raise ValueError("segment exceeds primary sequence length")
            if prior_segment_end is not None and segment.start < prior_segment_end:
                raise ValueError("segments must be ordered and non-overlapping")
            prior_segment_end = segment.end
        for site in self.site_instances:
            if site.end > sequence_length:
                raise ValueError("site instance exceeds primary sequence length")
        for nick in self.nick_events:
            if nick.boundary > sequence_length:
                raise ValueError("nick boundary exceeds primary sequence length")
        if self.bounded_segment is not None:
            if self.bounded_segment.start_boundary > sequence_length:
                raise ValueError("bounded segment boundary exceeds primary sequence length")
            if self.bounded_segment.end_boundary > sequence_length:
                raise ValueError("bounded segment boundary exceeds primary sequence length")
        return self
