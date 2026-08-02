"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/src/outputs/images.py

Writes BaseRender records to deterministic image artifacts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from ..config import ImagesOutputCfg, Style
from ..config.adapter_contracts import adapter_grid_record_limit, validate_records_output_policy
from ..core import Record, SchemaError
from ..render import Palette, render_record
from ..render.renderer import get_renderer_descriptor
from .names import _safe_stem, _unique_stem


def _grid_max_rows_for_records(records: list[Record]) -> int | None:
    """Return the strictest positive max-row hint on the supplied records."""

    max_rows: int | None = None
    for record in records:
        meta = record.meta
        if not isinstance(meta, Mapping):
            continue
        raw_value = meta.get("grid_max_rows")
        if raw_value is None:
            continue
        try:
            parsed = int(raw_value)
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            max_rows = parsed if max_rows is None else min(max_rows, parsed)
    return max_rows


def _grid_ncols_for_records(records: list[Record], *, default_ncols: int) -> int:
    """Return a grid column count from record-level layout hints."""

    max_rows = _grid_max_rows_for_records(records)
    if max_rows is None or len(records) <= max_rows:
        return max(1, min(default_ncols, len(records)))
    return max(1, math.ceil(len(records) / max_rows))


def _figure_rgba(fig):
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    return data.reshape((height, width, 4)).copy()


def _render_record_grid_figure_local(
    records: list[Record],
    *,
    renderer_name: str,
    style: Style,
    palette: Palette,
    ncols: int,
    max_rows: int | None = None,
):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    limits = tuple(
        limit
        for limit in (
            get_renderer_descriptor(renderer_name).max_grid_records,
            adapter_grid_record_limit(records),
        )
        if limit is not None
    )
    grid_limit = min(limits) if limits else None
    if grid_limit is not None and len(records) > grid_limit:
        raise SchemaError(
            f"renderer {renderer_name!r} supports at most {grid_limit} record per grid; render records individually"
        )
    panel_images: list[object] = []
    for record in records:
        panel = render_record(record, renderer_name=renderer_name, style=style, palette=palette)
        panel_images.append(_figure_rgba(panel))
        plt.close(panel)

    max_h = max(image.shape[0] for image in panel_images)
    max_w = max(image.shape[1] for image in panel_images)
    cols = min(ncols, len(panel_images))
    if max_rows is None:
        rows = int((len(panel_images) + cols - 1) / cols)
    else:
        rows = min(max_rows, len(panel_images))
    dpi = 120
    fig_w = (cols * max_w) / dpi
    fig_h = (rows * max_h) / dpi
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h), dpi=dpi, squeeze=False)
    flat_axes = list(axes.flat)
    for ax in flat_axes:
        ax.set_axis_off()

    for idx, image in enumerate(panel_images):
        if max_rows is None:
            ax = flat_axes[idx]
        else:
            row = idx % rows
            col = idx // rows
            ax = axes[row, col]
        ax.imshow(image)
        ax.set_axis_off()

    for ax in flat_axes[len(panel_images) :]:
        ax.set_axis_off()

    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99, wspace=0.055, hspace=0.025)
    if cols > 1:
        for col in range(1, cols):
            left_edge = axes[0, col - 1].get_position().x1
            right_edge = axes[0, col].get_position().x0
            separator_x = (left_edge + right_edge) / 2.0
            fig.add_artist(
                Line2D(
                    [separator_x, separator_x],
                    [0.015, 0.985],
                    transform=fig.transFigure,
                    color="#CBD5E1",
                    linewidth=1.2,
                    alpha=0.9,
                )
            )
    return fig


def write_images(
    records: Iterable[Record],
    *,
    output: ImagesOutputCfg,
    renderer_name: str,
    style: Style,
    palette: Palette,
) -> Path:
    import matplotlib.pyplot as plt

    materialized = list(records)
    if not materialized:
        raise SchemaError("No records to render after adapter, transforms, and selection")
    validate_records_output_policy(
        materialized,
        output_kind="images",
        image_output_mode="single_file" if output.path is not None else "directory",
    )

    if output.path is not None:
        out_path = output.path.resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if len(materialized) == 1:
            fig = render_record(materialized[0], renderer_name=renderer_name, style=style, palette=palette)
        else:
            fig = _render_record_grid_figure_local(
                materialized,
                renderer_name=renderer_name,
                style=style,
                palette=palette,
                ncols=_grid_ncols_for_records(materialized, default_ncols=1),
                max_rows=_grid_max_rows_for_records(materialized),
            )
        fig.patch.set_facecolor("white")
        fig.patch.set_alpha(1.0)
        fig.savefig(
            out_path,
            format=output.fmt,
            bbox_inches=None,
            pad_inches=0.0,
            facecolor="white",
        )
        plt.close(fig)
        return out_path

    assert output.dir is not None
    out_dir = output.dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    used: set[str] = set()
    for index, record in enumerate(materialized):
        stem = _safe_stem(record.id if record.id else f"record_{index}")
        name = _unique_stem(stem, used)
        out_path = out_dir / f"{name}.{output.fmt}"

        fig = render_record(record, renderer_name=renderer_name, style=style, palette=palette)
        fig.patch.set_facecolor("white")
        fig.patch.set_alpha(1.0)
        fig.savefig(
            out_path,
            format=output.fmt,
            bbox_inches=None,
            pad_inches=0.0,
            facecolor="white",
        )
        plt.close(fig)
    return out_dir


__all__ = [
    "_figure_rgba",
    "_grid_max_rows_for_records",
    "_grid_ncols_for_records",
    "_render_record_grid_figure_local",
    "render_record",
    "write_images",
]
