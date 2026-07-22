"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/retron_hairpin_design/catalog/specs/primitive_sources.py

Primitive source selector specs for Retron MSD compiler inputs.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class RetronMsdPrimitiveSourceSpecModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RankedPrimitiveSelectorSpec(RetronMsdPrimitiveSourceSpecModel):
    mode: Literal["rank"] = "rank"
    rank: int = Field(ge=1)

    @model_validator(mode="after")
    def _validate_mode_fields(self) -> "RankedPrimitiveSelectorSpec":
        if self.mode != "rank":
            raise ValueError("selector supports only mode=rank.")
        return self

    def requested_ranks(self) -> list[int]:
        return [self.rank]


class SnapbackCapSourceSpec(RetronMsdPrimitiveSourceSpecModel):
    kind: Literal["snapback_released_solve_cap"]
    run_dir: Path
    selector: RankedPrimitiveSelectorSpec


class ScarNickStemBaseSourceSpec(RetronMsdPrimitiveSourceSpecModel):
    kind: Literal["scar_nick_stem_bases"]
    run_dir: Path
    selector: RankedPrimitiveSelectorSpec


__all__ = [
    "RankedPrimitiveSelectorSpec",
    "ScarNickStemBaseSourceSpec",
    "SnapbackCapSourceSpec",
]
