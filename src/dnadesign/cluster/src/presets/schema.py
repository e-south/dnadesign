"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/cluster/src/presets/schema.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any, Dict, Literal

from pydantic import BaseModel, Field, validator


class Preset(BaseModel):
    name: str
    kind: Literal["fit", "umap", "plot"]
    params: Dict[str, Any] = Field(default_factory=dict)
    plot: Dict[str, Any] = Field(default_factory=dict)
    hue: Dict[str, Any] = Field(default_factory=dict)

    @validator("name")
    def nonempty(cls, v):
        if not v.strip():
            raise ValueError("name cannot be empty")
        return v
