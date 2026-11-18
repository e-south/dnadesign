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
from typing import Sequence

from ..config import ImagesOutputCfg, SequenceLayout, Style
from ..config.native_svg import NativeSvgCfg
from ..core import Record, SchemaError
from ..core.vector import RasterizerInfo
from .names import _safe_stem, _unique_stem
from .native_batch import PreparedNativeBatch
from .raster import resolve_rasterizer


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
        batch = PreparedNativeBatch.prepare(
            records,
            config=self.config,
            style=style,
            layout=layout,
            image_output_mode="single_file" if output.path else "directory",
        )
        root = output.path.parent if output.path is not None else output.dir
        if root is None:
            raise SchemaError("native images require an output path or directory")
        root.mkdir(parents=True, exist_ok=True)
        used, metrics = set(), []
        for svg, png, item in batch.draw(rasterizer=self.rasterizer):
            name = (
                output.path.name if output.path else f"{_unique_stem(_safe_stem(item['record_id']), used)}.{self.fmt}"
            )
            item["file"] = name
            with (root / name).open("xb") as handle:
                handle.write(svg.svg.encode("utf-8") if png is None else png.png)
            metrics.append(item)
        batch.verify_sources()
        return {"representation": "native_svg", "record_count": len(records), "records": metrics}
