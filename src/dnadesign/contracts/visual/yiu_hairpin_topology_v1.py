"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/contracts/visual/yiu_hairpin_topology_v1.py

Shared YIU hairpin-topology visual contract.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from .common import JsonMap, PositiveLengthSpan, VisualContractModel


class IndexPair(VisualContractModel):
    left_index: int = Field(ge=0)
    right_index: int = Field(ge=0)


class YiuDisplay(VisualContractModel):
    title: str | None = None


class YiuHairpinTopologyV1(VisualContractModel):
    contract_kind: Literal["yiu_hairpin_topology_v1"] = "yiu_hairpin_topology_v1"
    state_id: str
    topology_kind: Literal["ssdna_hairpin"] = "ssdna_hairpin"
    sequence: str
    stem_left_span: PositiveLengthSpan
    stem_right_span: PositiveLengthSpan
    loop_span: PositiveLengthSpan
    pair_map: list[IndexPair] = Field(default_factory=list)
    adapter_branches: list[JsonMap] = Field(default_factory=list)
    annotations: list[JsonMap] = Field(default_factory=list)
    display: YiuDisplay = Field(default_factory=YiuDisplay)
    meta: JsonMap = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_sequence_and_pairs(self) -> "YiuHairpinTopologyV1":
        if not self.sequence:
            raise ValueError("sequence must be non-empty")
        if not self.pair_map:
            raise ValueError("pair_map must be non-empty")
        return self
