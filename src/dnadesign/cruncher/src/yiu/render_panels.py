"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cruncher/src/yiu/render_panels.py

Shared YIU helpers for loading view contracts and assembling composite renders.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from dnadesign.cruncher.yiu.bundle_models import PayloadViewEntry
from dnadesign.cruncher.yiu.view_io import load_contract_rows
from dnadesign.cruncher.yiu.view_registry import validate_payload_view_entry

YIU_NUCLEOTIDE_LEGEND_CANONICAL_COLOR = "#4B5563"
YIU_NUCLEOTIDE_LEGEND_MISMATCH_COLOR = "#B91C1C"
YIU_NUCLEOTIDE_LEGEND_TEXT_COLOR = "#111827"
YIU_NUCLEOTIDE_LEGEND_HEIGHT_PX = 52
YIU_NUCLEOTIDE_LEGEND_COLUMN_SPACING = 0.75
YIU_NUCLEOTIDE_LEGEND_CENTER_Y = 0.30
YIU_NUCLEOTIDE_LEGEND_FONT_SIZE = 9.0


def _adapter_kind_for_view(view: PayloadViewEntry) -> str:
    return validate_payload_view_entry(view).adapter_kind


def load_view_records(contract_path: Path, *, view: PayloadViewEntry, baserender_module):
    rows = load_contract_rows(contract_path, input_kind=view.input_kind)
    return baserender_module.adapt_records(
        rows,
        adapter_kind=_adapter_kind_for_view(view),
        alphabet="IUPAC_DNA",
    )


def render_view_panel(
    *,
    baserender_module,
    records,
    renderer_kind: str,
    style_preset: str | None,
    style_overrides: dict[str, object],
):
    record_or_records = records[0] if len(records) == 1 else records
    grid = {"ncols": 2} if len(records) == 2 else None
    return baserender_module.render(
        record_or_records,
        renderer=renderer_kind,
        style={"preset": style_preset, "overrides": style_overrides},
        grid=grid,
    )


def figure_to_rgba_array(fig) -> np.ndarray:
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    return np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape((height, width, 4))


def trim_white_border_rgba(arr: np.ndarray, *, threshold: int = 248, pad_px: int = 8) -> np.ndarray:
    if arr.ndim != 3 or arr.shape[2] < 3:
        return arr
    rgb = arr[:, :, :3]
    non_white = np.any(rgb < int(threshold), axis=2)
    ys, xs = np.where(non_white)
    if ys.size == 0 or xs.size == 0:
        return arr
    height, width = arr.shape[:2]
    pad = max(0, int(pad_px))
    top = max(0, int(ys.min()) - pad)
    bottom = min(height - 1, int(ys.max()) + pad)
    left = max(0, int(xs.min()) - pad)
    right = min(width - 1, int(xs.max()) + pad)
    return arr[top : bottom + 1, left : right + 1, :]


def _draw_composite_nucleotide_legend(axis) -> None:
    from matplotlib.lines import Line2D

    axis.set_axis_off()
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.patch.set_alpha(0.0)

    items = [
        ("Canonical", YIU_NUCLEOTIDE_LEGEND_CANONICAL_COLOR),
        ("Mismatch", YIU_NUCLEOTIDE_LEGEND_MISMATCH_COLOR),
    ]
    legend = axis.legend(
        handles=[
            Line2D(
                [],
                [],
                marker="s",
                linestyle="None",
                markersize=8.0,
                color=color,
                markerfacecolor=color,
                markeredgecolor=color,
            )
            for _, color in items
        ],
        labels=[label for label, _ in items],
        loc="center",
        bbox_to_anchor=(0.5, YIU_NUCLEOTIDE_LEGEND_CENTER_Y),
        ncol=2,
        frameon=False,
        borderaxespad=0.0,
        borderpad=0.0,
        handlelength=0.8,
        handletextpad=0.35,
        columnspacing=YIU_NUCLEOTIDE_LEGEND_COLUMN_SPACING,
        fontsize=YIU_NUCLEOTIDE_LEGEND_FONT_SIZE,
    )
    legend._legend_box.align = "center"
    for text in legend.get_texts():
        text.set_color(YIU_NUCLEOTIDE_LEGEND_TEXT_COLOR)
        text.set_fontfamily("DejaVu Sans")


def save_composite_render(*, panel_images: list[np.ndarray], render_path: Path) -> None:
    import matplotlib.pyplot as plt

    if not panel_images:
        raise ValueError("YIU composite render requires at least one panel image")
    trimmed_images = [trim_white_border_rgba(image) for image in panel_images]
    height_ratios = [max(1, image.shape[0]) for image in trimmed_images]
    legend_height = YIU_NUCLEOTIDE_LEGEND_HEIGHT_PX
    max_width = max(image.shape[1] for image in trimmed_images)
    total_height = sum(height_ratios) + legend_height
    composite = plt.figure(figsize=(max_width / 180.0, total_height / 180.0), dpi=180)
    try:
        axes = composite.subplots(
            nrows=len(panel_images) + 1,
            ncols=1,
            gridspec_kw={"height_ratios": [*height_ratios, legend_height], "hspace": 0.03},
        )
        if not isinstance(axes, np.ndarray):
            axes = np.asarray([axes], dtype=object)
        axes_list = axes.tolist()
        panel_axes = axes_list[:-1]
        legend_axis = axes_list[-1]
        for axis, image in zip(panel_axes, trimmed_images, strict=True):
            axis.imshow(image)
            axis.set_axis_off()
        _draw_composite_nucleotide_legend(legend_axis)
        composite.patch.set_facecolor("white")
        composite.patch.set_alpha(1.0)
        render_path.parent.mkdir(parents=True, exist_ok=True)
        composite.savefig(
            render_path,
            format=render_path.suffix.lstrip(".") or "pdf",
            bbox_inches="tight",
            pad_inches=0.01,
            facecolor="white",
        )
    finally:
        plt.close(composite)


__all__ = [
    "figure_to_rgba_array",
    "load_contract_rows",
    "load_view_records",
    "render_view_panel",
    "save_composite_render",
    "trim_white_border_rgba",
]
