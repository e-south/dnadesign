"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/src/public/vector_output.py

Native SVG sequence output with pinned illustrations and measured placement.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from ..config.raster import RasterOptions
from ..config.sequence_layout import SequenceLayout
from ..core import Record
from ..core.vector import IllustrationCaption, PngRenderResult, SvgIllustration, SvgRenderResult


def render_record_svg(
    record: Record,
    *,
    style_preset: str | Path | None = None,
    style_overrides: Mapping[str, object] | None = None,
    illustrations: Mapping[str, SvgIllustration] | None = None,
    captions: Mapping[str, IllustrationCaption] | None = None,
    layout: SequenceLayout | None = None,
) -> SvgRenderResult:
    """Return native SVG plus geometry in points, with no source writes or fitting.

    Illustration and caption maps are keyed by anchored-effect asset_id. Caller
    artwork must declare source SHA-256, exact IDs, a viewBox and normalized anchors.
    All asset and font admission happens before allocating a figure.
    """
    from ..config import resolve_style
    from ..outputs.native_svg import render_svg_sequence
    from ..runtime import initialize_runtime

    initialize_runtime()
    style = resolve_style(preset=style_preset, overrides=dict(style_overrides or {}))
    return render_svg_sequence(record, style=style, illustrations=illustrations, captions=captions, layout=layout)


def rasterize_svg(
    source: SvgRenderResult,
    *,
    dpi: float = RasterOptions.dpi,
    inkscape: str | Path = RasterOptions.inkscape,
    timeout_s: float = RasterOptions.timeout_s,
    max_pixels: int = RasterOptions.max_pixels,
) -> PngRenderResult:
    """Rasterize an existing native SVG at the requested output resolution.

    Inkscape must already be available. No layout, fitting or artwork decoding
    is repeated; diagnostics retain the source composition's point geometry.
    """
    from ..outputs.raster import rasterize_svg as convert

    return convert(
        source, options=RasterOptions(dpi=dpi, inkscape=inkscape, timeout_s=timeout_s, max_pixels=max_pixels)
    )
