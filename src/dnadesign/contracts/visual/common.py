"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/contracts/visual/common.py

Shared model helpers for cassette visual contracts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class VisualContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CoordinateSpan(VisualContractModel):
    start: int = Field(ge=0)
    end: int = Field(ge=0)

    @model_validator(mode="after")
    def _validate_bounds(self) -> "CoordinateSpan":
        if self.end < self.start:
            raise ValueError("span.end must be >= span.start")
        return self


class PositiveLengthSpan(CoordinateSpan):
    @model_validator(mode="after")
    def _validate_positive_length(self) -> "PositiveLengthSpan":
        super()._validate_bounds()
        if self.end <= self.start:
            raise ValueError("span.end must be > span.start")
        return self


class RenderLabel(VisualContractModel):
    text: str
    placement: str = "header"


JsonMap = dict[str, Any]
