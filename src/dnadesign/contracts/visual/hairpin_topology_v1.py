"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/contracts/visual/hairpin_topology_v1.py

Shared ssDNA hairpin visual contract for cassette QA rendering.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, model_validator

from .common import PositiveLengthSpan, VisualContractModel


class HairpinTopologySpans(VisualContractModel):
    stem5p_span: PositiveLengthSpan
    loop_span: PositiveLengthSpan
    stem3p_span: PositiveLengthSpan

    @model_validator(mode="after")
    def _validate_topology_order(self) -> "HairpinTopologySpans":
        if self.stem5p_span.end > self.loop_span.start:
            raise ValueError("stem5p_span must end at or before loop_span.start")
        if self.loop_span.end > self.stem3p_span.start:
            raise ValueError("loop_span must end at or before stem3p_span.start")
        return self


class PairMapEntry(VisualContractModel):
    left_index: int = Field(ge=0)
    right_index: int = Field(ge=0)

    @model_validator(mode="after")
    def _validate_pair(self) -> "PairMapEntry":
        if self.right_index <= self.left_index:
            raise ValueError("pair_map right_index must be > left_index")
        return self


class FeatureSpanAnnotation(VisualContractModel):
    id: str
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    semantic: str
    label: str | None = None

    @model_validator(mode="after")
    def _validate_bounds(self) -> "FeatureSpanAnnotation":
        if self.end <= self.start:
            raise ValueError("feature span end must be > start")
        return self


class DuplexDerivedAnnotation(VisualContractModel):
    kind: str
    text: str


class HairpinTopologyViewV1(VisualContractModel):
    version: Literal[1] = 1
    kind: Literal["ssdna_hairpin_v1"] = "ssdna_hairpin_v1"
    view_id: str
    solution_id: str
    title: str
    primary_sequence_5to3: str
    topology: HairpinTopologySpans
    pair_map: list[PairMapEntry] = Field(default_factory=list)
    feature_spans: list[FeatureSpanAnnotation] = Field(default_factory=list)
    duplex_derived_annotations: list[DuplexDerivedAnnotation] = Field(default_factory=list)
    meta: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_sequence(self) -> "HairpinTopologyViewV1":
        if not self.primary_sequence_5to3:
            raise ValueError("primary_sequence_5to3 must be non-empty")
        sequence_length = len(self.primary_sequence_5to3)
        if not self.pair_map:
            raise ValueError("pair_map must be non-empty")
        for span in (
            self.topology.stem5p_span,
            self.topology.loop_span,
            self.topology.stem3p_span,
        ):
            if span.end > sequence_length:
                raise ValueError("topology span exceeds primary sequence length")
        for pair in self.pair_map:
            if pair.left_index < self.topology.stem5p_span.start:
                raise ValueError("pair_map left_index must remain inside stem5p_span")
            if pair.left_index >= self.topology.stem5p_span.end:
                raise ValueError("pair_map left_index must remain inside stem5p_span")
            if pair.right_index >= self.topology.stem3p_span.end:
                raise ValueError("pair_map right_index must remain inside stem3p_span")
            if pair.right_index < self.topology.stem3p_span.start:
                raise ValueError("pair_map right_index must remain inside stem3p_span")
            if pair.right_index >= sequence_length:
                raise ValueError("pair_map index exceeds primary sequence length")
        for feature in self.feature_spans:
            if feature.end > sequence_length:
                raise ValueError("feature span exceeds primary sequence length")
        return self
