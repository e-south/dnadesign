"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/src/render/fonts.py

Resolve declared font identities and measure literal SVG text without fallback.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from matplotlib.font_manager import FontProperties, findfont, get_font
from matplotlib.path import Path as GlyphPath
from matplotlib.textpath import TextToPath

from ..core import RenderingError
from ..core.vector import ResolvedFont


def resolve_font(family: str) -> ResolvedFont:
    if not isinstance(family, str) or not family.strip():
        raise RenderingError("font family must be a non-empty string")
    try:
        path = findfont(FontProperties(family=family), fallback_to_default=False)
        face = get_font(path)
        digest = sha256(Path(path).read_bytes()).hexdigest()
    except (ValueError, OSError, RuntimeError) as exc:
        raise RenderingError(f"font {family!r} is unavailable; select an installed font explicitly") from exc
    return ResolvedFont(family, face.family_name, face.style_name, path, digest)


@dataclass(frozen=True)
class LiteralTextMetrics:
    advance_pt: float
    left_pt: float
    bottom_pt: float
    right_pt: float
    top_pt: float


def measure_literal_text(text: str, font: ResolvedFont, size_pt: float) -> LiteralTextMetrics:
    face = get_font(font.path)
    missing = sorted({ord(char) for char in text if not face.get_char_index(ord(char))})
    if missing:
        codes = ", ".join(f"U+{code:04X}" for code in missing)
        raise RenderingError(f"font {font.requested_family!r} does not cover text characters {codes}")
    prop = FontProperties(fname=font.path, size=size_pt)
    converter = TextToPath()
    advance, _height, _descent = converter.get_text_width_height_descent(text, prop, ismath=False)
    vertices, codes = converter.get_text_path(prop, text, ismath=False)
    bounds = GlyphPath(vertices, codes).get_extents()
    scale = size_pt / converter.FONT_SCALE
    return LiteralTextMetrics(advance, bounds.x0 * scale, bounds.y0 * scale, bounds.x1 * scale, bounds.y1 * scale)
