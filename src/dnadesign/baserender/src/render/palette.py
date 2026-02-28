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
import re
from typing import Mapping

_PROMOTER_TAG_RE = re.compile(r"^promoter:(?P<name>[^:]+):(?P<component>[^:]+)$")
_PRIMARY_PASTEL_HEX = (
    "#8AB6F9",
    "#F6A89E",
    "#F2C879",
    "#8ECFA2",
    "#B8B3F5",
    "#84CDD6",
    "#E9A7D5",
    "#AFCF86",
    "#9CC3F0",
    "#F2B79B",
    "#A2D9E8",
    "#D3B4F0",
)


def _hsl_to_rgb(h: float, s: float, lightness: float) -> tuple[float, float, float]:
    import colorsys

    return colorsys.hls_to_rgb(h, lightness, s)


def _hex_to_rgb(hex_color: str) -> tuple[float, float, float]:
    color = str(hex_color).strip().lstrip("#")
    if len(color) != 6:
        raise ValueError(f"Palette color must be #RRGGBB, got {hex_color!r}")
    r = int(color[0:2], 16) / 255.0
    g = int(color[2:4], 16) / 255.0
    b = int(color[4:6], 16) / 255.0
    return (r, g, b)


def _stable_index(key: str, n: int) -> int:
    if n <= 0:
        raise ValueError("Palette index space must be positive")
    digest = hashlib.sha1(key.encode("utf8")).hexdigest()
    return int(digest[:8], 16) % n


def _promoter_component_color(name: str, component: str) -> tuple[float, float, float]:
    digest = hashlib.sha1(name.encode("utf8")).hexdigest()
    hue = int(digest[:2], 16) / 255.0
    component_norm = str(component).strip().lower()
    if component_norm in {"upstream", "-35"}:
        return _hsl_to_rgb(hue, 0.56, 0.70)
    if component_norm in {"downstream", "-10"}:
        return _hsl_to_rgb(hue, 0.50, 0.77)
    return _hsl_to_rgb(hue, 0.53, 0.74)


class Palette:
    def __init__(self, overrides: Mapping[str, str] | None = None):
        self.overrides = dict(overrides or {})

    def color_for(self, tag: str) -> tuple[float, float, float]:
        hex_color = self.overrides.get(tag)
        if hex_color:
            return _hex_to_rgb(hex_color)

        promoter_match = _PROMOTER_TAG_RE.match(str(tag))
        if promoter_match is not None:
            return _promoter_component_color(
                name=promoter_match.group("name"),
                component=promoter_match.group("component"),
            )

        idx = _stable_index(str(tag), len(_PRIMARY_PASTEL_HEX))
        return _hex_to_rgb(_PRIMARY_PASTEL_HEX[idx])
