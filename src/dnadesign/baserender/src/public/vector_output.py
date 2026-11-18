"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/src/public/vector_output.py

Native SVG sequence output with pinned illustrations and measured placement.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from ..config.sequence_layout import SequenceLayout
from ..core import Record, RenderingError
from ..core.vector import (
    IllustrationCaption,
    SvgArtwork,
    SvgIllustration,
    SvgIllustrationPlacement,
    SvgRenderDiagnostics,
    SvgRenderResult,
    SvgSpanLinkPlacement,
)


def _default_illustration(asset_id):
    root = Path(__file__).resolve().parents[2] / "assets" / "overlays"
    if not isinstance(asset_id, str) or Path(asset_id).name != asset_id:
        raise RenderingError("illustration asset IDs must not contain paths")
    try:
        payload = json.loads((root / f"{asset_id}.json").read_text())
        if payload.get("schema") != "baserender.anchored_illustration_asset.v1" or payload.get("asset_id") != asset_id:
            raise RenderingError("packaged illustration manifest schema or asset_id mismatch")
        vector = payload["vector"]
        filename = payload["source_svg"]
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise RenderingError("packaged SVG source must be a file name")
        return SvgIllustration(
            SvgArtwork(
                root / filename,
                vector["sha256"],
                tuple(vector["element_ids"]),
                tuple(vector["view_box"]),
                {key: (point["x_fraction"], point["y_fraction_from_top"]) for key, point in payload["anchors"].items()},
            ),
            IllustrationCaption(**payload["caption"]),
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise RenderingError(
            f"asset {asset_id!r} has no valid packaged vector selection; supply SvgIllustration"
        ) from exc


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
    from ..render.fonts import resolve_font
    from ..render.svg_assets import load_svg_selection
    from ..render.svg_caption import measure_caption
    from ..runtime import initialize_runtime

    initialize_runtime()
    if not isinstance(record, Record):
        raise RenderingError("record must be a Record")
    style = resolve_style(preset=style_preset, overrides=dict(style_overrides or {}))
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
            spec = _default_illustration(asset_id)
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
    from ..outputs.svg import SvgPlacement, SvgSlot, compose_svg
    from ..render.nucleotide_geometry import displayed_nucleotides, nucleotide_box
    from ..render.palette import Palette
    from ..render.renderer import validate_records_for_rendering
    from ..render.sequence_rows import SequenceRowsRenderer

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

    def box_pt(box):
        x0, y0, x1, y1 = box
        return (x0 * scale, (layout.height - y1) * scale, x1 * scale, (layout.height - y0) * scale)

    pending = []
    diagnostics = []
    slot_by_effect = {}
    for effect_index, geometry in prepared.illustration_geometries.items():
        effect = prepared.record.effects[effect_index]
        asset_id = effect.params["asset_id"]
        source, fragment, caption, font = selections[asset_id]
        slot = f"br-vector-{len(pending)}"
        bounds = box_pt(geometry.box)
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

    def draw_body(axis, effect, geometry, *, effect_index):
        axis.add_artist(SvgSlot(slot_by_effect[effect_index]))

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
                nucleotide_boxes[placement.feature_id] = box_pt(shape.paint_bounds)
        result_diagnostics = SvgRenderDiagnostics(
            record_id=record.id,
            base_pitch_pt=layout.cw * scale,
            strand_anchors_pt=((layout.height - layout.y_forward) * scale, (layout.height - layout.y_reverse) * scale),
            sequence_origin_pt=(layout.x_left * scale, (layout.height - layout.y_forward) * scale),
            feature_bounds_pt=MappingProxyType({key: box_pt(box) for key, box in layout.feature_boxes.items()}),
            nucleotide_box_bounds_pt=MappingProxyType(nucleotide_boxes),
            illustrations=tuple(diagnostics),
            text_font_policy="reference",
            fonts=MappingProxyType(fonts),
            span_links=tuple(
                SvgSpanLinkPlacement(
                    effect_index=index,
                    label=geometry.label,
                    font_size_pt=geometry.font_size_pt,
                    endpoints_pt=(
                        (geometry.x1 * scale, (layout.height - geometry.y) * scale),
                        (geometry.x2 * scale, (layout.height - geometry.y) * scale),
                    ),
                    label_bounds_pt=None if geometry.label_bounds is None else box_pt(geometry.label_bounds),
                )
                for index, geometry in prepared.span_link_geometries.items()
            ),
        )
        return SvgRenderResult(svg, layout.width * scale, layout.height * scale, result_diagnostics)
    finally:
        if figure is not None:
            plt.close(figure)
