"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/src/config/raster.py

Shared strict settings for final native rasterization in APIs and jobs.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Mapping

from ..core import RenderingError


def positive_number(value, name):
    try:
        valid = not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value) and value > 0
    except OverflowError:
        valid = False
    if not valid:
        raise RenderingError(f"{name} must be a positive finite number")


@dataclass(frozen=True)
class RasterOptions:
    dpi: float = 180
    inkscape: str | Path = "inkscape"
    timeout_s: float = 30
    max_pixels: int = 64_000_000

    def __post_init__(self):
        positive_number(self.dpi, "dpi")
        positive_number(self.timeout_s, "timeout_s")
        if not isinstance(self.max_pixels, int) or isinstance(self.max_pixels, bool) or self.max_pixels <= 0:
            raise RenderingError("max_pixels must be a positive integer pixel budget")
        if not isinstance(self.inkscape, (str, Path)) or not str(self.inkscape).strip():
            raise RenderingError("inkscape must name an executable or an explicit executable path")
        object.__setattr__(self, "dpi", float(self.dpi))
        object.__setattr__(self, "timeout_s", float(self.timeout_s))
        object.__setattr__(self, "inkscape", str(self.inkscape))

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]):
        if not isinstance(value, Mapping):
            raise RenderingError("rasterization must be a mapping")
        unknown = set(value) - {item.name for item in fields(cls)}
        if unknown:
            raise RenderingError(f"rasterization received unknown fields: {sorted(unknown, key=str)}")
        return cls(**value)

    def to_mapping(self):
        return {
            item.name: str(value) if isinstance(value, Path) else value
            for item in fields(self)
            if (value := getattr(self, item.name)) is not None
        }
