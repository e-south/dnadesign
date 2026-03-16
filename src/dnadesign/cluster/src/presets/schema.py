"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/cluster/src/presets/schema.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any, Dict, Literal

from pydantic import BaseModel, Field, field_validator


class Preset(BaseModel):
    name: str
    kind: Literal["method", "umap", "plot", "analysis"]
    params: Dict[str, Any] = Field(default_factory=dict)
    plot: Dict[str, Any] = Field(default_factory=dict)
    hue: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def nonempty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("name cannot be empty")
        return v
