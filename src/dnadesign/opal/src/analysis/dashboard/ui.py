"""UI helpers for dashboard notebooks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import polars as pl

from .util import build_color_dropdown_options, build_friendly_column_labels


@dataclass(frozen=True)
class UmapControls:
    umap_x_input: Any
    umap_y_input: Any
    umap_color_dropdown: Any
    umap_color_label_map: dict[str, str]
    umap_size_slider: Any
    umap_opacity_slider: Any


_DEFAULT_EXTRA_COLORS = [
    "opal__score__scalar",
    "opal__score__rank",
    "opal__score__top_k",
    "opal__ledger__score",
    "opal__ledger__top_k",
    "opal__cache__score",
    "opal__cache__top_k",
    "opal__transient__score",
    "opal__transient__logic_fidelity",
    "opal__transient__effect_scaled",
    "opal__transient__observed_event",
    "opal__transient__top_k",
]


def build_umap_controls(
    *,
    mo: Any,
    df_active: pl.DataFrame,
    rf_model_source_value: str | None,
    score_source_value: str | None,
    campaign_slug: str | None,
    extra_color_options: Sequence[str] | None = None,
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
    extra_color_options = list(extra_color_options or _DEFAULT_EXTRA_COLORS)
    umap_color_cols = build_color_dropdown_options(
        df_active,
        extra=extra_color_options,
        include_none=False,
    )
    umap_color_cols = [_name for _name in umap_color_cols if _name != "id_right"]
    umap_color_default = "cluster__ldn_v1" if "cluster__ldn_v1" in umap_color_cols else "(none)"

    rf_prefix = "OPAL artifact" if rf_model_source_value == "OPAL artifact (model.joblib)" else "Transient"
    score_source_label = score_source_value or "Selected"
    friendly_labels = build_friendly_column_labels(
        score_source_label=score_source_label,
        rf_prefix=rf_prefix,
        campaign_slug=campaign_slug,
    )
    umap_color_label_map: dict[str, str] = {}
    umap_color_options = ["(none)"]
    for col in umap_color_cols:
        label = friendly_labels.get(col, col)
        if label in umap_color_label_map:
            label = col
        umap_color_label_map[label] = col
        umap_color_options.append(label)
    umap_color_default_label = umap_color_default
    for label, col in umap_color_label_map.items():
        if col == umap_color_default:
            umap_color_default_label = label
            break
    umap_color_dropdown = mo.ui.dropdown(
        options=umap_color_options,
        value=umap_color_default_label,
        label="Color by",
        full_width=True,
    )
    umap_size_slider = mo.ui.slider(5, 200, value=30, label="Point size")
    umap_opacity_slider = mo.ui.slider(0.1, 1.0, value=0.7, label="Opacity", step=0.05)
    return UmapControls(
        umap_x_input=umap_x_input,
        umap_y_input=umap_y_input,
        umap_color_dropdown=umap_color_dropdown,
        umap_color_label_map=umap_color_label_map,
        umap_size_slider=umap_size_slider,
        umap_opacity_slider=umap_opacity_slider,
    )
