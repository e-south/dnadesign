"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/contracts/visual/hairpin_topology_v1.py

Shared ssDNA hairpin visual contract for cassette QA rendering.

Module Author(s): OpenAI Codex
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
        for span in (
            self.topology.stem5p_span,
            self.topology.loop_span,
            self.topology.stem3p_span,
        ):
            if span.end > sequence_length:
                raise ValueError("topology span exceeds primary sequence length")
        for pair in self.pair_map:
            if pair.right_index >= sequence_length:
                raise ValueError("pair_map index exceeds primary sequence length")
        return self
