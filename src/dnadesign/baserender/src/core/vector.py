"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/src/core/vector.py

Native vector sources, caption geometry and derived raster-result contracts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Mapping

from .errors import RenderingError


def _plain(value):
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return {item.name: _plain(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def finite_tuple(value, length, name, *, positive=False):
    if not isinstance(value, (tuple, list)) or len(value) != length:
        raise RenderingError(f"{name} must contain {length} numbers")
    for item in value:
        valid = not isinstance(item, bool) and isinstance(item, (int, float))
        try:
            valid = valid and math.isfinite(item)
        except OverflowError:
            valid = False
        if not valid:
            raise RenderingError(f"{name} must contain finite numbers")
        if positive and item <= 0:
            raise RenderingError(f"{name} must contain positive numbers")


def validate_literal_text(text, *, field_name="text", allow_blank=False):
    if not isinstance(text, str) or (not allow_blank and not text.strip()):
        raise RenderingError(f"{field_name} must be {'a string' if allow_blank else 'a non-empty string'}")
    if any(unicodedata.category(char) in {"Cc", "Zl", "Zp"} for char in text):
        raise RenderingError(f"{field_name} must be single-line literal text without control characters")


@dataclass(frozen=True)
class SvgArtwork:
    path: Path
    sha256: str
    element_ids: tuple[str, ...]
    view_box: tuple[float, float, float, float]
    anchors: Mapping[str, tuple[float, float]]

    def to_mapping(self, *, portable: bool = False):
        payload = _plain(self)
        if portable:
            payload.pop("path")
        return payload

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
        ratio = self.view_box[3] / self.view_box[2]
        if not math.isfinite(ratio) or ratio <= 0:
            raise RenderingError("svg artwork.view_box aspect ratio must remain finite and positive")
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
class AnnotationTypography:
    """One native annotation size, fixed in points or relative to sequence type.

    An explicitly selected group overrides site/span/caption sizes together.
    Omit the group to retain independently specified sizes. The selected group's
    default is 1.25 times the sequence font; fonts, text and anchors are unchanged.
    """

    size_pt: float | None = None
    sequence_scale: float | None = None

    def __post_init__(self):
        if self.size_pt is not None and self.sequence_scale is not None:
            raise RenderingError("annotation_typography selects size_pt or sequence_scale, not both")
        if self.size_pt is None and self.sequence_scale is None:
            object.__setattr__(self, "sequence_scale", 1.25)
        for name in ("size_pt", "sequence_scale"):
            value = getattr(self, name)
            if value is not None:
                finite_tuple((value,), 1, f"annotation_typography.{name}", positive=True)
        if self.size_pt is not None and self.size_pt < 6:
            raise RenderingError("annotation_typography.size_pt must be >= 6 pt")

    def resolve_size(self, sequence_size_pt):
        value = self.size_pt if self.size_pt is not None else sequence_size_pt * self.sequence_scale
        finite_tuple((value,), 1, "annotation_typography resolved size", positive=True)
        if value < 6:
            raise RenderingError("annotation_typography resolved size must be >= 6 pt")
        return float(value)

    def to_mapping(self):
        return {name: value for name in ("size_pt", "sequence_scale") if (value := getattr(self, name)) is not None}


@dataclass(frozen=True)
class IllustrationCaption:
    text: str
    family: str = "DejaVu Sans"
    size_pt: float = 12.0
    anchor_fraction: tuple[float, float] = (0.5, 1.0)
    offset_px: tuple[float, float] = (0.0, 0.0)
    visible: bool = True

    def validate(self):
        validate_literal_text(self.text, field_name="caption.text")
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

    def to_mapping(self, *, portable: bool = False):
        return {"artwork": self.artwork.to_mapping(portable=portable), "caption": _plain(self.caption)}


@dataclass(frozen=True)
class SvgIllustrationPlacement:
    asset_id: str
    source_sha256: str
    bounds_pt: tuple[float, float, float, float]
    anchors_pt: Mapping[str, tuple[float, float]]
    caption_font_path: str | None
    caption_origin_pt: tuple[float, float] | None
    caption_bounds_pt: tuple[float, float, float, float] | None


@dataclass(frozen=True)
class ResolvedFont:
    requested_family: str
    resolved_family: str
    style: str
    path: str
    sha256: str


@dataclass(frozen=True)
class TextMeasurement:
    """Literal ink bounds relative to the text origin/baseline, in points (y up)."""

    text: str
    font: ResolvedFont
    size_pt: float
    advance_pt: float
    ink_bounds_pt: tuple[float, float, float, float]

    @property
    def ink_width_pt(self) -> float:
        return self.ink_bounds_pt[2] - self.ink_bounds_pt[0]

    @property
    def ink_height_pt(self) -> float:
        return self.ink_bounds_pt[3] - self.ink_bounds_pt[1]

    def to_mapping(self, *, portable=False):
        payload = _plain(self)
        if portable:
            payload["font"].pop("path")
        return payload


@dataclass(frozen=True)
class SvgSpanLinkPlacement:
    effect_index: int
    label: str
    font_size_pt: float | None
    endpoints_pt: tuple[tuple[float, float], tuple[float, float]]
    label_bounds_pt: tuple[float, float, float, float] | None


@dataclass(frozen=True)
class SvgLegendEntryPlacement:
    tag: str
    label: str
    font_size_pt: float
    patch_bounds_pt: tuple[float, float, float, float]
    text_origin_pt: tuple[float, float]
    bounds_pt: tuple[float, float, float, float]


@dataclass(frozen=True)
class SvgRenderDiagnostics:
    record_id: str
    base_pitch_pt: float
    strand_anchors_pt: tuple[float, float]
    sequence_origin_pt: tuple[float, float]
    feature_bounds_pt: Mapping[str, tuple[float, float, float, float]]
    nucleotide_box_bounds_pt: Mapping[str, tuple[float, float, float, float]]
    illustrations: tuple[SvgIllustrationPlacement, ...]
    text_font_policy: str
    fonts: Mapping[str, ResolvedFont]
    span_links: tuple[SvgSpanLinkPlacement, ...]
    bottom_legend: tuple[SvgLegendEntryPlacement, ...]
    annotation_font_size_pt: float | None = None

    def to_mapping(self, *, portable: bool = False):
        """Return detached JSON-shaped diagnostics, optionally without local font paths."""
        payload = _plain(self)
        if self.annotation_font_size_pt is None:
            payload.pop("annotation_font_size_pt")
        if portable:
            for font in payload["fonts"].values():
                font.pop("path")
            for illustration in payload["illustrations"]:
                illustration.pop("caption_font_path")
        return payload


@dataclass(frozen=True)
class SvgRenderResult:
    svg: str
    width_pt: float
    height_pt: float
    diagnostics: SvgRenderDiagnostics


@dataclass(frozen=True)
class RasterizerInfo:
    executable: str
    executable_sha256: str
    version: str

    def to_mapping(self, *, portable: bool = False):
        payload = _plain(self)
        if portable:
            payload["executable"] = Path(self.executable).name
        return payload


@dataclass(frozen=True)
class PngRenderResult:
    png: bytes = field(repr=False)
    width_px: int
    height_px: int
    dpi: float
    source_svg_sha256: str
    diagnostics: SvgRenderDiagnostics
    rasterizer: RasterizerInfo
