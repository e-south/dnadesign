"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/src/config/plots.py

DenseGen plotting configuration schema.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator
from typing_extensions import Literal


class PlotConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    out_dir: str = "outputs/plots"
    format: Literal["png", "pdf", "svg"] = "pdf"
    source: Optional[Literal["usr", "parquet"]] = None
    default: List[str] = Field(default_factory=list)
    options: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    style: Dict[str, Any] = Field(default_factory=dict)
    sample_rows: Optional[int] = None

    @field_validator("sample_rows")
    @classmethod
    def _sample_rows_ok(cls, v: Optional[int]):
        if v is None:
            return v
        if int(v) <= 0:
            raise ValueError("plots.sample_rows must be > 0")
        return int(v)
