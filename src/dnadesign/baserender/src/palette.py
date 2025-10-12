"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/baserender/palette.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
from typing import Mapping, Tuple


# Simple, stable hash->HSL->RGB mapping for tags with no explicit color.
def _hsl_to_rgb(h: float, s: float, l: float) -> Tuple[float, float, float]:  # noqa
    # h in [0,1], s,l in [0,1]
    import colorsys

    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return (r, g, b)


class Palette:
    def __init__(self, overrides: Mapping[str, str] | None = None):
        self.overrides = dict(overrides or {})  # hex string overrides

    def color_for(self, tag: str) -> Tuple[float, float, float]:
        # prefer override
        hex_color = self.overrides.get(tag)
        if hex_color:
            h = hex_color.lstrip("#")
            r = int(h[0:2], 16) / 255.0
            g = int(h[2:4], 16) / 255.0
            b = int(h[4:6], 16) / 255.0
            return (r, g, b)

        # derive stable pastel from hash
        digest = hashlib.sha1(tag.encode("utf8")).hexdigest()
        hue = int(digest[:2], 16) / 255.0  # 0..1
        # Softer (pastel): slightly lower saturation, higher lightness
        sat = 0.38
        light = 0.70
        return _hsl_to_rgb(hue, sat, light)
