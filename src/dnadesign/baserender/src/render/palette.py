"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/baserender/src/render/palette.py

Tag-based palette with stable hash colors and optional explicit overrides.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
from typing import Mapping


def _hsl_to_rgb(h: float, s: float, lightness: float) -> tuple[float, float, float]:
    import colorsys

    return colorsys.hls_to_rgb(h, lightness, s)


class Palette:
    def __init__(self, overrides: Mapping[str, str] | None = None):
        self.overrides = dict(overrides or {})

    def color_for(self, tag: str) -> tuple[float, float, float]:
        hex_color = self.overrides.get(tag)
        if hex_color:
            h = hex_color.lstrip("#")
            if len(h) != 6:
                raise ValueError(f"Palette override for tag '{tag}' must be #RRGGBB")
            r = int(h[0:2], 16) / 255.0
            g = int(h[2:4], 16) / 255.0
            b = int(h[4:6], 16) / 255.0
            return (r, g, b)

        digest = hashlib.sha1(tag.encode("utf8")).hexdigest()
        hue = int(digest[:2], 16) / 255.0
        return _hsl_to_rgb(hue, 0.38, 0.70)
