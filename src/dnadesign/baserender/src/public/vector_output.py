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


def _font_path(family):
    from matplotlib.font_manager import FontProperties, findfont

    try:
        return findfont(FontProperties(family=family), fallback_to_default=False)
    except ValueError as exc:
        raise RenderingError(f"font {family!r} is unavailable; select an installed font explicitly") from exc


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
    from ..render.svg_assets import load_svg_selection
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
    _font_path(style.font_mono)
    _font_path(style.font_label)
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
        font_path = _font_path(caption.family) if caption is not None else None
        selections[asset_id] = (source, load_svg_selection(source), caption, font_path)
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
    prepared = renderer.prepare(bound, style, palette, layout_request=layout)
    layout = prepared.layout
    placements = []
    diagnostics = []
    scale = 72.0 / style.dpi

    def draw_body(axis, effect, geometry):
        source, fragment, caption, font_path = selections[effect.params["asset_id"]]
        slot = f"br-vector-{len(placements)}"
        # The final canvas height is supplied below before serialization.
        placements.append((slot, source, fragment, caption, geometry))
        diagnostics.append((effect.params["asset_id"], source, geometry, font_path))
        axis.add_artist(SvgSlot(slot))

    import matplotlib.pyplot as plt

    figure = None
    try:
        figure = renderer.draw_prepared(prepared, illustration_drawer=draw_body)

        def box_pt(box):
            x0, y0, x1, y1 = box
            return (x0 * scale, (layout.height - y1) * scale, x1 * scale, (layout.height - y0) * scale)

        pending = [
            SvgPlacement(slot, source, fragment, box_pt(geometry.box), caption, scale)
            for slot, source, fragment, caption, geometry in placements
        ]
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
            record.id,
            layout.cw * scale,
            ((layout.height - layout.y_forward) * scale, (layout.height - layout.y_reverse) * scale),
            (layout.x_left * scale, (layout.height - layout.y_forward) * scale),
            MappingProxyType({key: box_pt(box) for key, box in layout.feature_boxes.items()}),
            MappingProxyType(nucleotide_boxes),
            tuple(
                SvgIllustrationPlacement(
                    asset_id,
                    source.sha256,
                    box_pt(geometry.box),
                    MappingProxyType(
                        {key: (x * scale, (layout.height - y) * scale) for key, (x, y) in geometry.anchors.items()}
                    ),
                    font_path,
                )
                for asset_id, source, geometry, font_path in diagnostics
            ),
        )
        return SvgRenderResult(svg, layout.width * scale, layout.height * scale, result_diagnostics)
    finally:
        if figure is not None:
            plt.close(figure)
