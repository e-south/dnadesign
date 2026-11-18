"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/src/outputs/native_svg.py

Compose sequence rows with native artwork, captions and measured diagnostics.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import TYPE_CHECKING, Mapping

from ..config import SequenceLayout, Style
from ..core import Record, RenderingError
from ..core.vector import (
    IllustrationCaption,
    SvgArtwork,
    SvgIllustration,
    SvgIllustrationPlacement,
    SvgLegendEntryPlacement,
    SvgRenderDiagnostics,
    SvgRenderResult,
    SvgSpanLinkPlacement,
)
from ..render.fonts import resolve_font
from ..render.svg_assets import load_packaged_svg_illustration, load_svg_selection
from ..render.svg_caption import measure_caption

if TYPE_CHECKING:
    from ..core.vector import ResolvedFont
    from ..render.sequence_rows import PreparedSequenceRows
    from .svg import SvgPlacement


@dataclass(frozen=True)
class PreparedNativeSvg:
    rows: PreparedSequenceRows
    placements: tuple[SvgPlacement, ...]
    illustrations: tuple[SvgIllustrationPlacement, ...]
    fonts: Mapping[str, ResolvedFont]
    slots: Mapping[int, str]
    px_to_pt: float


def _box_pt(box, layout, scale):
    x0, y0, x1, y1 = box
    return (x0 * scale, (layout.height - y1) * scale, x1 * scale, (layout.height - y0) * scale)


def prepare_native_svg(
    record: Record,
    *,
    style: Style,
    illustrations: Mapping[str, SvgIllustration] | None = None,
    captions: Mapping[str, IllustrationCaption] | None = None,
    layout: SequenceLayout | None = None,
) -> PreparedNativeSvg:
    """Bind assets, fonts and geometry before allocating a figure."""
    if not isinstance(record, Record):
        raise RenderingError("record must be a Record")
    if style.figure_scale != 1:
        raise RenderingError("native SVG requires figure_scale=1 to preserve physical typography")
    if record.display.trajectory_panel is not None:
        raise RenderingError("native SVG currently supports sequence rows without a trajectory panel")
    for name, mapping in (("illustrations", illustrations), ("captions", captions)):
        if mapping is not None and not isinstance(mapping, Mapping):
            raise RenderingError(f"{name} must be a mapping keyed by asset_id")
    illustrations = dict(illustrations or {})
    captions = dict(captions or {})
    effects = [effect for effect in record.effects if effect.kind == "anchored_illustration"]
    if any(
        not isinstance(effect.params.get("asset_id"), str) or not effect.params["asset_id"].strip()
        for effect in effects
    ):
        raise RenderingError("anchored illustration asset_id must be a non-empty string")
    if any(not isinstance(key, str) for key in illustrations.keys() | captions.keys()):
        raise RenderingError("illustration and caption selection keys must be strings")
    asset_ids = {effect.params["asset_id"] for effect in effects}
    unknown = (illustrations.keys() | captions.keys()) - asset_ids
    if unknown:
        raise RenderingError(f"unused illustration or caption selections: {sorted(unknown)}")
    fonts = {"sequence": resolve_font(style.font_mono), "label": resolve_font(style.font_label)}
    selections = {}
    for asset_id in asset_ids:
        spec = illustrations.get(asset_id)
        if spec is None:
            spec = load_packaged_svg_illustration(asset_id)
        if not isinstance(spec, SvgIllustration) or not isinstance(spec.artwork, SvgArtwork):
            raise RenderingError("illustrations must contain SvgIllustration with SvgArtwork")
        source = spec.artwork.validate()
        source = replace(source, anchors=MappingProxyType({key: tuple(point) for key, point in source.anchors.items()}))
        caption = captions.get(asset_id, spec.caption)
        if caption is not None:
            if not isinstance(caption, IllustrationCaption):
                raise RenderingError("captions must contain IllustrationCaption")
            caption.validate()
        font = resolve_font(caption.family) if caption is not None else None
        if font is not None:
            fonts[f"caption:{asset_id}"] = font
            caption = replace(caption, family=font.resolved_family)
        selections[asset_id] = (source, load_svg_selection(source), caption, font)
    bound_effects = []
    for effect in record.effects:
        if effect.kind == "anchored_illustration":
            if "image_tint" in effect.params:
                raise RenderingError("image_tint is a raster operation; select vector artwork colors explicitly")
            source = selections[effect.params["asset_id"]][0]
            effect = replace(effect, params={**effect.params, "svg_asset": source})
        bound_effects.append(effect)
    bound = replace(record, effects=tuple(bound_effects))
    from ..render.palette import Palette
    from ..render.renderer import validate_records_for_rendering
    from ..render.sequence_rows import SequenceRowsRenderer
    from .svg import SvgPlacement

    palette = Palette(style.palette)
    # Neutral admission and measured preparation precede figure allocation.
    # Native composition can qualify its own text/artwork bounds on this plan.
    validate_records_for_rendering(
        (bound,), renderer_name="sequence_rows", style=style, palette=palette, preflight_renderer=False
    )
    feature_ids = {feature.id for feature in bound.features if feature.id is not None}
    for effect in bound.effects:
        if effect.kind == "anchored_illustration":
            for binding in effect.target["bindings"]:
                if binding["feature_id"] not in feature_ids:
                    raise RenderingError(f"missing illustration feature binding: {binding['feature_id']!r}")
    renderer = SequenceRowsRenderer()
    prepared = renderer.prepare(bound, style, palette, layout_request=layout, preserve_type=True)
    layout = prepared.layout
    scale = 72.0 / style.dpi

    pending = []
    diagnostics = []
    slot_by_effect = {}
    for effect_index, geometry in prepared.illustration_geometries.items():
        effect = prepared.record.effects[effect_index]
        asset_id = effect.params["asset_id"]
        source, fragment, caption, font = selections[asset_id]
        slot = f"br-vector-{len(pending)}"
        bounds = _box_pt(geometry.box, layout, scale)
        caption_geometry = measure_caption(
            caption,
            font,
            bounds,
            asset_id=asset_id,
            px_to_pt=scale,
            canvas_size_pt=(layout.width * scale, layout.height * scale),
        )
        pending.append(SvgPlacement(slot, source, fragment, bounds, caption, caption_geometry))
        slot_by_effect[effect_index] = slot
        diagnostics.append(
            SvgIllustrationPlacement(
                asset_id=asset_id,
                source_sha256=source.sha256,
                bounds_pt=bounds,
                anchors_pt=MappingProxyType(
                    {key: (x * scale, (layout.height - y) * scale) for key, (x, y) in geometry.anchors.items()}
                ),
                caption_font_path=None if font is None else font.path,
                caption_origin_pt=None if caption_geometry is None else caption_geometry.origin_pt,
                caption_bounds_pt=None if caption_geometry is None else caption_geometry.bounds_pt,
            )
        )

    return PreparedNativeSvg(
        prepared, tuple(pending), tuple(diagnostics), MappingProxyType(fonts), MappingProxyType(slot_by_effect), scale
    )


def draw_native_svg(source: PreparedNativeSvg) -> SvgRenderResult:
    from ..render.nucleotide_geometry import displayed_nucleotides, nucleotide_box
    from ..render.sequence_rows import SequenceRowsRenderer
    from .svg import SvgSlot, compose_svg

    prepared = source.rows
    record = bound = prepared.record
    layout, style, scale = prepared.layout, prepared.style, source.px_to_pt
    renderer = SequenceRowsRenderer()
    pending = source.placements

    def draw_body(axis, effect, geometry, *, effect_index):
        axis.add_artist(SvgSlot(source.slots[effect_index]))

    import matplotlib.pyplot as plt

    figure = None
    try:
        figure = renderer.draw_prepared(prepared, illustration_drawer=draw_body)

        svg = compose_svg(figure, pending)
        nucleotide_boxes = {}
        for placement in layout.placements:
            feature = bound.features[placement.feature_index]
            if feature.kind != "interval_annotation":
                shape = nucleotide_box(
                    displayed_nucleotides(feature),
                    style,
                    x=placement.x,
                    y=placement.y,
                    width=placement.w,
                    height=placement.h,
                    cell_width=layout.cw,
                    cell_height=layout.ch,
                )
                nucleotide_boxes[placement.feature_id] = _box_pt(shape.paint_bounds, layout, scale)
        result_diagnostics = SvgRenderDiagnostics(
            record_id=record.id,
            base_pitch_pt=layout.cw * scale,
            strand_anchors_pt=((layout.height - layout.y_forward) * scale, (layout.height - layout.y_reverse) * scale),
            sequence_origin_pt=(layout.x_left * scale, (layout.height - layout.y_forward) * scale),
            feature_bounds_pt=MappingProxyType(
                {key: _box_pt(box, layout, scale) for key, box in layout.feature_boxes.items()}
            ),
            nucleotide_box_bounds_pt=MappingProxyType(nucleotide_boxes),
            illustrations=source.illustrations,
            text_font_policy="reference",
            fonts=source.fonts,
            bottom_legend=tuple(
                SvgLegendEntryPlacement(
                    tag=entry.tag,
                    label=entry.label,
                    font_size_pt=entry.font_size_pt,
                    patch_bounds_pt=_box_pt(entry.patch_bounds, layout, scale),
                    text_origin_pt=(entry.text_origin[0] * scale, (layout.height - entry.text_origin[1]) * scale),
                    bounds_pt=_box_pt(entry.bounds, layout, scale),
                )
                for entry in prepared.bottom_legend
            ),
            span_links=tuple(
                SvgSpanLinkPlacement(
                    effect_index=index,
                    label=geometry.label,
                    font_size_pt=geometry.font_size_pt,
                    endpoints_pt=(
                        (geometry.x1 * scale, (layout.height - geometry.y) * scale),
                        (geometry.x2 * scale, (layout.height - geometry.y) * scale),
                    ),
                    label_bounds_pt=None
                    if geometry.label_bounds is None
                    else _box_pt(geometry.label_bounds, layout, scale),
                )
                for index, geometry in prepared.span_link_geometries.items()
            ),
        )
        return SvgRenderResult(svg, layout.width * scale, layout.height * scale, result_diagnostics)
    finally:
        if figure is not None:
            plt.close(figure)


def render_svg_sequence(
    record: Record,
    *,
    style: Style,
    illustrations: Mapping[str, SvgIllustration] | None = None,
    captions: Mapping[str, IllustrationCaption] | None = None,
    layout: SequenceLayout | None = None,
) -> SvgRenderResult:
    return draw_native_svg(
        prepare_native_svg(record, style=style, illustrations=illustrations, captions=captions, layout=layout)
    )
