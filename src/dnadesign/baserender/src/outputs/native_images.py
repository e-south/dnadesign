"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/src/outputs/native_images.py

Prepare complete native image batches and write into the owning publication stage.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Sequence

from ..config import ImagesOutputCfg, SequenceLayout, Style
from ..config.native_svg import NativeSvgCfg
from ..core import Record, RenderingError, SchemaError
from ..core.vector import RasterizerInfo, SvgIllustration
from ..integrations import validate_records_output_policy
from ..render.palette import Palette
from ..render.renderer import validate_records_for_rendering
from ..render.svg_assets import load_svg_selection
from .names import _safe_stem, _unique_stem
from .native_svg import draw_native_svg, prepare_native_svg
from .raster import rasterize_svg, resolve_rasterizer


@dataclass(frozen=True)
class NativeImagesWriter:
    config: NativeSvgCfg
    fmt: str
    rasterizer: RasterizerInfo | None

    @classmethod
    def resolve(cls, config: NativeSvgCfg, *, fmt: str):
        backend = resolve_rasterizer(config.rasterization) if fmt == "png" else None
        return cls(config, fmt, backend)

    def write(self, records: Sequence[Record], *, output: ImagesOutputCfg, style: Style, layout: SequenceLayout | None):
        if output.fmt != self.fmt or self.fmt not in {"svg", "png"}:
            raise SchemaError("native images support the admitted SVG/PNG format only")
        if not records:
            raise SchemaError("No records to render")
        if output.path is not None and len(records) != 1:
            raise SchemaError(
                "native single-file images require exactly one record; use an image directory for a batch"
            )
        records = validate_records_for_rendering(
            records,
            renderer_name="sequence_rows",
            style=style,
            palette=Palette(style.palette),
            preflight_renderer=False,
        )
        validate_records_output_policy(
            records, output_kind="images", image_output_mode="single_file" if output.path else "directory"
        )
        asset_ids = {
            effect.params.get("asset_id")
            for record in records
            for effect in record.effects
            if effect.kind == "anchored_illustration"
        }
        unused = (self.config.illustrations.keys() | self.config.captions.keys()) - asset_ids
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
                    illustrations={key: value for key, value in self.config.illustrations.items() if key in selected},
                    captions={key: value for key, value in self.config.captions.items() if key in selected},
                )
            )
        root = output.path.parent if output.path is not None else output.dir
        if root is None:
            raise SchemaError("native images require an output path or directory")
        root.mkdir(parents=True, exist_ok=True)
        used, metrics = set(), []
        for record, plan in zip(records, plans, strict=True):
            name = (
                output.path.name
                if output.path is not None
                else f"{_unique_stem(_safe_stem(record.id), used)}.{self.fmt}"
            )
            svg = draw_native_svg(plan)
            item = {
                "record_id": record.id,
                "file": name,
                "source_svg_sha256": sha256(svg.svg.encode()).hexdigest(),
                "width_pt": svg.width_pt,
                "height_pt": svg.height_pt,
                "diagnostics": svg.diagnostics.to_mapping(portable=True),
                "resolved_illustrations": {
                    placement.slot_id: SvgIllustration(placement.artwork, placement.caption).to_mapping(portable=True)
                    for placement in plan.placements
                },
            }
            if self.fmt == "svg":
                payload = svg.svg.encode("utf-8")
            else:
                png = rasterize_svg(svg, options=self.config.rasterization, rasterizer=self.rasterizer)
                payload = png.png
                item.update(
                    {
                        "width_px": png.width_px,
                        "height_px": png.height_px,
                        "dpi": png.dpi,
                        "rasterizer": png.rasterizer.to_mapping(portable=True),
                    }
                )
            with (root / name).open("xb") as handle:
                handle.write(payload)
            metrics.append(item)
        # Recheck caller-pinned inputs before the existing publication boundary.
        checked = set()
        for plan in plans:
            for placement in plan.placements:
                artwork = placement.artwork
                identity = (artwork.path, artwork.sha256)
                if identity not in checked:
                    load_svg_selection(artwork)
                    checked.add(identity)
        return {"representation": "native_svg", "record_count": len(records), "records": metrics}
