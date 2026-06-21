"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/retron_hairpin_design/review_outputs/pwm/sequence_rows.py

BaseRender sequence-row panel rendering for bidirectional TetR PWM trim review outputs.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import dnadesign.baserender as baserender

from ..contracts.plan import PwmTrimPanel
from .baserender_record import (
    PwmLogoColumn,
    PwmLogoLayer,
    build_pwm_baserender_record,
    observed_sequence_for_panel,
)
from .typography import SCALE_BAR_FONT_SIZE, SEQUENCE_FONT_SIZE
from .visual_layers import EXCLUDED_COLOR, PANEL_COLOR

PANEL_WIDTH = 690


def render_pwm_sequence_row_panel(
    columns: Sequence[PwmLogoColumn],
    *,
    parent_sequence: str,
    panel: PwmTrimPanel,
    logo_layers: Sequence[PwmLogoLayer],
) -> Image.Image:
    record = build_pwm_baserender_record(
        columns,
        parent_sequence=parent_sequence,
        panel=panel,
        logo_layers=logo_layers,
    )
    fig = baserender.render_record_figure(
        record,
        style_preset="presentation_default",
        style_overrides=_sequence_row_style(),
    )
    try:
        image = _figure_to_image(fig)
    finally:
        plt.close(fig)
    return _resize_to_width(_crop_white(image), PANEL_WIDTH)


def _sequence_row_style() -> dict[str, object]:
    style = dict(baserender.cruncher_showcase_style_overrides())
    style.update(
        {
            "show_reverse_complement": True,
            "legend": False,
            "figure_scale": 1.2,
            "padding_y": 12.0,
            "font_size_seq": SEQUENCE_FONT_SIZE,
            "font_size_label": SEQUENCE_FONT_SIZE,
            "font_size_feature_label": SEQUENCE_FONT_SIZE,
            "show_coordinate_ticks": True,
            "uniform_display_font_size": True,
            "palette": {**dict(style.get("palette") or {}), "tf:tetR_trim": PANEL_COLOR},
        }
    )
    style["layout"] = {**dict(style.get("layout") or {}), "outer_pad_cells": 0.5}
    style["motif_logo"] = {
        **dict(style.get("motif_logo") or {}),
        "bits_to_cells": 1.05,
        "letter_coloring": {
            "mode": "match_window_seq",
            "other_color": EXCLUDED_COLOR,
            "observed_color_source": "feature_fill",
        },
        "scale_bar": {"enabled": True, "location": "left_of_logo", "font_size": SCALE_BAR_FONT_SIZE},
    }
    return style


def _figure_to_image(fig) -> Image.Image:
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape((height, width, 4))
    return Image.fromarray(data.copy(), mode="RGBA")


def _crop_white(image: Image.Image) -> Image.Image:
    rgba = np.asarray(image.convert("RGBA"))
    alpha = rgba[:, :, 3]
    rgb = rgba[:, :, :3]
    mask = ((rgb < 245).any(axis=2)) & (alpha > 0)
    if not mask.any():
        return image
    ys, xs = np.where(mask)
    pad = 12
    return image.crop(
        (
            max(0, int(xs.min()) - pad),
            max(0, int(ys.min()) - pad),
            min(image.width, int(xs.max()) + pad + 1),
            min(image.height, int(ys.max()) + pad + 1),
        )
    )


def _resize_to_width(image: Image.Image, width: int) -> Image.Image:
    if image.width == width:
        return image.convert("RGB")
    height = max(1, int(round(image.height * (width / max(1, image.width)))))
    return image.resize((width, height), Image.Resampling.LANCZOS).convert("RGB")


__all__ = [
    "PANEL_COLOR",
    "PANEL_WIDTH",
    "PwmLogoColumn",
    "PwmLogoLayer",
    "observed_sequence_for_panel",
    "render_pwm_sequence_row_panel",
]
