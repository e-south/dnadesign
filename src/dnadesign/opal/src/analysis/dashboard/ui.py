"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/analysis/dashboard/ui.py

Builds reusable UI widgets for dashboard notebooks. Keeps UI wiring thin by
centralizing control construction.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import polars as pl

from .hues import HueRegistry


@dataclass(frozen=True)
class UmapControls:
    umap_x_input: Any
    umap_y_input: Any
    umap_color_dropdown: Any
    hue_registry: HueRegistry


def build_umap_controls(
    *,
    mo: Any,
    df_active: pl.DataFrame,
    hue_registry: HueRegistry,
    default_hue_key: str | None = None,
) -> UmapControls:
    default_x = None
    default_y = None
    for name in df_active.columns:
        if name.endswith("__umap_x"):
            default_x = name
            break
    for name in df_active.columns:
        if name.endswith("__umap_y"):
            default_y = name
            break
    if "cluster__ldn_v1__umap_x" in df_active.columns:
        default_x = "cluster__ldn_v1__umap_x"
    if "cluster__ldn_v1__umap_y" in df_active.columns:
        default_y = "cluster__ldn_v1__umap_y"

    umap_x_input = mo.ui.text(
        value=default_x or "",
        label="UMAP X column",
        full_width=True,
    )
    umap_y_input = mo.ui.text(
        value=default_y or "",
        label="UMAP Y column",
        full_width=True,
    )
    options = ["(none)"] + [option.key for option in hue_registry.options]
    default_label = default_hue_key if default_hue_key in options else None
    umap_color_dropdown = mo.ui.dropdown(
        options=options,
        value=default_label or "(none)",
        label="Color by",
        full_width=True,
    )
    return UmapControls(
        umap_x_input=umap_x_input,
        umap_y_input=umap_y_input,
        umap_color_dropdown=umap_color_dropdown,
        hue_registry=hue_registry,
    )
