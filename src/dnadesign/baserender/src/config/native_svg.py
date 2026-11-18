"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/src/config/native_svg.py

Native composition settings shared with the SVG API's source/caption contracts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import Callable, Mapping

from ..core import RenderingError, SchemaError, reject_unknown_keys, require_mapping
from ..core.vector import AnnotationTypography, IllustrationCaption, SvgArtwork, SvgIllustration
from .raster import RasterOptions
from .sequence_layout import sequence_layout_from_options


def _caption(raw, context):
    data = require_mapping(raw, context)
    reject_unknown_keys(data, {"text", "family", "size_pt", "anchor_fraction", "offset_px", "visible"}, context)
    if "text" not in data:
        raise SchemaError(f"{context}.text is required")
    return IllustrationCaption(**data).validate()


def _freeze_caption(value):
    value.validate()
    return replace(value, anchor_fraction=tuple(value.anchor_fraction), offset_px=tuple(value.offset_px))


@dataclass(frozen=True)
class NativeSvgCfg:
    illustrations: Mapping[str, SvgIllustration] = field(default_factory=dict)
    captions: Mapping[str, IllustrationCaption] = field(default_factory=dict)
    rasterization: RasterOptions = field(default_factory=RasterOptions)
    annotation_typography: AnnotationTypography | None = None

    def __post_init__(self):
        if self.annotation_typography is not None and not isinstance(self.annotation_typography, AnnotationTypography):
            raise RenderingError("native_svg.annotation_typography must be AnnotationTypography")
        if not isinstance(self.rasterization, RasterOptions):
            raise RenderingError("native_svg.rasterization must be RasterOptions")
        for name, values in (("illustrations", self.illustrations), ("captions", self.captions)):
            if not isinstance(values, Mapping) or any(not isinstance(key, str) or not key.strip() for key in values):
                raise RenderingError(f"native_svg.{name} must be a mapping keyed by nonempty asset IDs")
        illustrations = {}
        for key, value in self.illustrations.items():
            if not isinstance(value, SvgIllustration) or not isinstance(value.artwork, SvgArtwork):
                raise RenderingError("native_svg.illustrations must contain SvgIllustration values")
            art = value.artwork.validate()
            art = replace(
                art,
                view_box=tuple(art.view_box),
                anchors=MappingProxyType({key: tuple(point) for key, point in art.anchors.items()}),
            )
            illustrations[key] = SvgIllustration(art, None if value.caption is None else _freeze_caption(value.caption))
        if any(not isinstance(value, IllustrationCaption) for value in self.captions.values()):
            raise RenderingError("native_svg.captions must contain IllustrationCaption values")
        object.__setattr__(self, "illustrations", MappingProxyType(illustrations))
        object.__setattr__(
            self, "captions", MappingProxyType({key: _freeze_caption(value) for key, value in self.captions.items()})
        )

    @classmethod
    def from_mapping(cls, raw, *, resolve_path: Callable[[str, str], Path]):
        data = require_mapping(raw, "render.native_svg")
        reject_unknown_keys(
            data, {"illustrations", "captions", "rasterization", "annotation_typography"}, "render.native_svg"
        )
        typography = None
        if data.get("annotation_typography") is not None:
            settings = require_mapping(data["annotation_typography"], "native_svg.annotation_typography")
            reject_unknown_keys(settings, {"size_pt", "sequence_scale"}, "native_svg.annotation_typography")
            typography = AnnotationTypography(**settings)
        illustrations = {}
        for key, raw_value in require_mapping(data.get("illustrations", {}), "native_svg.illustrations").items():
            context = f"native_svg.illustrations.{key}"
            spec = require_mapping(raw_value, context)
            reject_unknown_keys(spec, {"artwork", "caption"}, context)
            art = require_mapping(spec.get("artwork"), context + ".artwork")
            reject_unknown_keys(art, {"path", "sha256", "element_ids", "view_box", "anchors"}, context + ".artwork")
            path, ids = art.get("path"), art.get("element_ids")
            if not isinstance(path, (str, Path)) or not str(path).strip() or not isinstance(ids, (list, tuple)):
                raise SchemaError(f"{context}.artwork needs a path string and element_ids list")
            artwork = SvgArtwork(
                resolve_path(str(path), context + ".artwork.path"),
                art.get("sha256"),
                tuple(ids),
                art.get("view_box"),
                art.get("anchors"),
            ).validate()
            illustrations[key] = SvgIllustration(
                artwork, None if spec.get("caption") is None else _caption(spec["caption"], context + ".caption")
            )
        captions = {
            key: _caption(value, f"native_svg.captions.{key}")
            for key, value in require_mapping(data.get("captions", {}), "native_svg.captions").items()
        }
        return cls(illustrations, captions, RasterOptions.from_mapping(data.get("rasterization", {})), typography)

    def to_mapping(self, *, portable=False):
        illustrations = {key: value.to_mapping(portable=portable) for key, value in self.illustrations.items()}
        raster = self.rasterization.to_mapping()
        if portable:
            raster["inkscape"] = Path(raster["inkscape"]).name
        result = {
            "illustrations": illustrations,
            "captions": {key: asdict(value) for key, value in self.captions.items()},
            "rasterization": raster,
        }
        if self.annotation_typography is not None:
            result["annotation_typography"] = self.annotation_typography.to_mapping()
        return result


def validate_native_outputs(render, outputs):
    if render.native_svg is None:
        if any(
            getattr(output, "background", None) is not None or getattr(output, "encode_timeout_s", None) is not None
            for output in outputs
        ):
            raise SchemaError("video background and encode_timeout_s currently require render.native_svg")
        return
    if not isinstance(render.native_svg, NativeSvgCfg):
        raise SchemaError("render.native_svg must be NativeSvgCfg in a typed job")
    if render.native_svg.annotation_typography is not None:
        from .style_v1 import resolve_style

        try:
            style = resolve_style(preset=render.style_preset, overrides=render.style_overrides)
            render.native_svg.annotation_typography.resolve_size(style.font_size_seq)
        except RenderingError as exc:
            raise SchemaError(str(exc)) from exc
    if render.renderer != "sequence_rows":
        raise SchemaError("render.native_svg currently requires renderer sequence_rows")
    if any(
        output.kind not in {"images", "video"} or (output.kind == "images" and output.fmt not in {"svg", "png"})
        for output in outputs
    ):
        raise SchemaError("render.native_svg supports SVG/PNG images and fixed-layout MP4 video")
    for output in outputs:
        if output.kind != "video":
            continue
        if not any(item.kind == "images" for item in outputs):
            raise SchemaError("render.native_svg video requires SVG/PNG images as a static fallback")
        layout = sequence_layout_from_options(render.options)
        for name in ("canvas_size_pt", "sequence_origin_pt"):
            if layout is None or getattr(layout, name) is None:
                raise SchemaError(f"native video requires explicit layout.{name}")
        if not isinstance(output.background, str) or re.fullmatch(r"#[0-9a-fA-F]{6}", output.background) is None:
            raise SchemaError("native video background must be an explicit opaque #RRGGBB color")
        if output.content_fit != "native":
            raise SchemaError(
                "native video content_fit must be 'native'; change the point canvas or raster DPI instead"
            )
        if output.title_text is not None or output.title_font_size is not None or output.title_align != "center":
            raise SchemaError(
                "native video output title settings are unsupported; use the record's normal display title"
            )
        if output.encode_timeout_s is not None:
            from .raster import positive_number

            try:
                positive_number(output.encode_timeout_s, "encode_timeout_s")
            except RenderingError as exc:
                raise SchemaError(str(exc)) from exc
