"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/src/core/vector.py

Explicit source, caption and measured-result contracts for native SVG output.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .errors import RenderingError


def finite_tuple(value, length, name, *, positive=False):
    if not isinstance(value, (tuple, list)) or len(value) != length:
        raise RenderingError(f"{name} must contain {length} numbers")
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item):
            raise RenderingError(f"{name} must contain finite numbers")
        if positive and item <= 0:
            raise RenderingError(f"{name} must contain positive numbers")


@dataclass(frozen=True)
class SvgArtwork:
    path: Path
    sha256: str
    element_ids: tuple[str, ...]
    view_box: tuple[float, float, float, float]
    anchors: Mapping[str, tuple[float, float]]

    def validate(self):
        if not isinstance(self.path, Path):
            raise RenderingError("svg artwork.path must be a pathlib.Path")
        if not isinstance(self.sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", self.sha256):
            raise RenderingError("svg artwork.sha256 must be a lowercase SHA-256 digest")
        if (
            not isinstance(self.element_ids, tuple)
            or not self.element_ids
            or any(not isinstance(key, str) or not key.strip() for key in self.element_ids)
            or len(set(self.element_ids)) != len(self.element_ids)
        ):
            raise RenderingError("svg artwork.element_ids must be distinct, non-empty IDs")
        finite_tuple(self.view_box, 4, "svg artwork.view_box")
        if self.view_box[2] <= 0 or self.view_box[3] <= 0:
            raise RenderingError("svg artwork.view_box width and height must be positive")
        if not isinstance(self.anchors, Mapping) or not self.anchors:
            raise RenderingError("svg artwork.anchors must be a non-empty mapping")
        for key, point in self.anchors.items():
            if not isinstance(key, str) or not key.strip():
                raise RenderingError("svg artwork anchor names must be non-empty strings")
            finite_tuple(point, 2, f"svg artwork.anchors.{key}")
            if any(not 0 <= n <= 1 for n in point):
                raise RenderingError("svg artwork anchors use fractions within [0, 1], measured from the top left")
        return self


@dataclass(frozen=True)
class IllustrationCaption:
    text: str
    family: str = "DejaVu Sans"
    size_pt: float = 12.0
    anchor_fraction: tuple[float, float] = (0.5, 1.0)
    offset_px: tuple[float, float] = (0.0, 0.0)
    visible: bool = True

    def validate(self):
        if not isinstance(self.text, str) or not self.text.strip():
            raise RenderingError("caption.text must be a non-empty string")
        if not isinstance(self.family, str) or not self.family.strip():
            raise RenderingError("caption.family must name a font")
        finite_tuple((self.size_pt,), 1, "caption.size_pt", positive=True)
        finite_tuple(self.anchor_fraction, 2, "caption.anchor_fraction")
        if any(not 0 <= n <= 1 for n in self.anchor_fraction):
            raise RenderingError("caption.anchor_fraction must lie within [0, 1]")
        finite_tuple(self.offset_px, 2, "caption.offset_px")
        if not isinstance(self.visible, bool):
            raise RenderingError("caption.visible must be boolean")
        return self


@dataclass(frozen=True)
class SvgIllustration:
    artwork: SvgArtwork
    caption: IllustrationCaption | None = None


@dataclass(frozen=True)
class SvgIllustrationPlacement:
    asset_id: str
    source_sha256: str
    bounds_pt: tuple[float, float, float, float]
    anchors_pt: Mapping[str, tuple[float, float]]
    caption_font_path: str | None


@dataclass(frozen=True)
class SvgRenderDiagnostics:
    record_id: str
    base_pitch_pt: float
    strand_anchors_pt: tuple[float, float]
    sequence_origin_pt: tuple[float, float]
    feature_bounds_pt: Mapping[str, tuple[float, float, float, float]]
    nucleotide_box_bounds_pt: Mapping[str, tuple[float, float, float, float]]
    illustrations: tuple[SvgIllustrationPlacement, ...]


@dataclass(frozen=True)
class SvgRenderResult:
    svg: str
    width_pt: float
    height_pt: float
    diagnostics: SvgRenderDiagnostics
