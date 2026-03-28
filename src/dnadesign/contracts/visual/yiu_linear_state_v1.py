"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/contracts/visual/yiu_linear_state_v1.py

Shared YIU linear-state visual contract.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from .common import JsonMap, VisualContractModel


class YiuDisplay(VisualContractModel):
    title: str | None = None


class YiuLinearStateV1(VisualContractModel):
    contract_kind: Literal["yiu_linear_state_v1"] = "yiu_linear_state_v1"
    state_id: str
    topology_kind: str
    alphabet: Literal["dna", "iupac_dna"] = "dna"
    primary_sequence: str
    complement_sequence: str | None = None
    segments: list[JsonMap] = Field(default_factory=list)
    annotations: list[JsonMap] = Field(default_factory=list)
    cuts: list[JsonMap] = Field(default_factory=list)
    junctions: list[JsonMap] = Field(default_factory=list)
    fragments: list[JsonMap] = Field(default_factory=list)
    display: YiuDisplay = Field(default_factory=YiuDisplay)
    meta: JsonMap = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_sequences(self) -> "YiuLinearStateV1":
        if not self.primary_sequence:
            raise ValueError("primary_sequence must be non-empty")
        return self
