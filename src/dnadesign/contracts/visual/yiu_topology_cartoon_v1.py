"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/contracts/visual/yiu_topology_cartoon_v1.py

Shared YIU topology-cartoon visual contract.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from .common import JsonMap, VisualContractModel


class YiuDisplay(VisualContractModel):
    title: str | None = None


class YiuTopologyCartoonV1(VisualContractModel):
    contract_kind: Literal["yiu_topology_cartoon_v1"] = "yiu_topology_cartoon_v1"
    state_id: str
    topology_kind: str
    sequence: str | None = None
    segments: list[JsonMap] = Field(default_factory=list)
    annotations: list[JsonMap] = Field(default_factory=list)
    cuts: list[JsonMap] = Field(default_factory=list)
    junctions: list[JsonMap] = Field(default_factory=list)
    fragments: list[JsonMap] = Field(default_factory=list)
    display: YiuDisplay = Field(default_factory=YiuDisplay)
    meta: JsonMap = Field(default_factory=dict)
