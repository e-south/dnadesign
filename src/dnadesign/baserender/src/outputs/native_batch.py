"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/src/outputs/native_batch.py

Admit native batches once and share measured compositions across output formats.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Sequence

from ..config import SequenceLayout, Style
from ..config.native_svg import NativeSvgCfg
from ..core import Record, RenderingError
from ..core.vector import SvgIllustration
from ..integrations import validate_records_output_policy
from ..render.palette import Palette
from ..render.renderer import validate_records_for_rendering
from ..render.svg_assets import load_svg_selection
from .native_svg import PreparedNativeSvg, draw_native_svg, prepare_native_svg
from .raster import rasterize_svg


@dataclass(frozen=True)
class PreparedNativeBatch:
    plans: tuple[PreparedNativeSvg, ...]
    config: NativeSvgCfg

    @classmethod
    def prepare(
        cls,
        records: Sequence[Record],
        *,
        config: NativeSvgCfg,
        style: Style,
        layout: SequenceLayout | None,
        output_kind="images",
        image_output_mode="directory",
    ):
        records = validate_records_for_rendering(
            records,
            renderer_name="sequence_rows",
            style=style,
            palette=Palette(style.palette),
            preflight_renderer=False,
        )
        validate_records_output_policy(records, output_kind=output_kind, image_output_mode=image_output_mode)
        asset_ids = {
            effect.params.get("asset_id")
            for record in records
            for effect in record.effects
            if effect.kind == "anchored_illustration"
        }
        unused = (config.illustrations.keys() | config.captions.keys()) - asset_ids
        if unused:
            raise RenderingError(f"unused native SVG illustration/caption selections: {sorted(unused)}")
        plans = []
        for record in records:
            selected = {
                effect.params.get("asset_id") for effect in record.effects if effect.kind == "anchored_illustration"
            }
            plans.append(
                prepare_native_svg(
                    record,
                    style=style,
                    layout=layout,
                    illustrations={key: value for key, value in config.illustrations.items() if key in selected},
                    captions={key: value for key, value in config.captions.items() if key in selected},
                )
            )
        return cls(tuple(plans), config)

    def draw(self, *, rasterizer=None):
        for plan in self.plans:
            svg = draw_native_svg(plan)
            item = {
                "record_id": plan.rows.record.id,
                "source_svg_sha256": sha256(svg.svg.encode()).hexdigest(),
                "width_pt": svg.width_pt,
                "height_pt": svg.height_pt,
                "diagnostics": svg.diagnostics.to_mapping(portable=True),
                "resolved_illustrations": {
                    placement.slot_id: SvgIllustration(placement.artwork, placement.caption).to_mapping(portable=True)
                    for placement in plan.placements
                },
            }
            png = None
            if rasterizer is not None:
                png = rasterize_svg(svg, options=self.config.rasterization, rasterizer=rasterizer)
                item.update(
                    {
                        "width_px": png.width_px,
                        "height_px": png.height_px,
                        "dpi": png.dpi,
                        "rasterizer": png.rasterizer.to_mapping(portable=True),
                    }
                )
            yield svg, png, item

    def verify_sources(self):
        # Recheck caller-pinned inputs before the existing publication boundary.
        checked = set()
        for plan in self.plans:
            for placement in plan.placements:
                artwork = placement.artwork
                identity = (artwork.path, artwork.sha256)
                if identity not in checked:
                    load_svg_selection(artwork)
                    checked.add(identity)
