"""
Projection rendering helpers for generated latentdna marimo notebooks.
"""

from __future__ import annotations

import math
from pathlib import Path

import marimo as mo
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .browser_runtime_support import (
    CONTROL_PLANE_PALETTE,
    display_hue_label,
    draw_reference_labels,
    fig_to_image,
    load_table,
    scatter_style,
    shared_join_key,
)


def enrich_projection_frame(
    frame: pd.DataFrame, joinable_tables: list[dict[str, object]], *, output_root: Path
) -> pd.DataFrame:
    enriched = frame.copy()
    for item in joinable_tables:
        relative_path = item.get("relative_path")
        artifact_id = str(item.get("artifact_id") or "artifact")
        if not isinstance(relative_path, str):
            continue
        table = load_table(output_root / relative_path)
        if table.empty:
            continue
        join_key = shared_join_key(enriched, table)
        if join_key is None:
            continue
        table = table.drop(columns=[column for column in ["x", "y"] if column in table.columns])
        rename_map = {}
        for column in table.columns:
            if column == join_key:
                continue
            if column == "cluster_label":
                rename_map[column] = f"cluster_label__{artifact_id}"
            elif column in enriched.columns:
                rename_map[column] = f"{column}__{artifact_id}"
        if rename_map:
            table = table.rename(columns=rename_map)
        keep_columns = [join_key] + [column for column in table.columns if column != join_key]
        if len(keep_columns) <= 1:
            continue
        enriched = enriched.merge(table[keep_columns], on=join_key, how="left")
    return enriched


def load_projection_frame(
    view_id: str, projection_id: str, joinable_tables: list[dict[str, object]], *, output_root: Path
) -> pd.DataFrame:
    frame = load_table(output_root / "projections" / projection_id / "coords.parquet")
    if frame.empty:
        return frame
    return enrich_projection_frame(frame, joinable_tables, output_root=output_root)


def render_projection_grid(
    panel_specs: list[dict[str, object]],
    *,
    hue_column: str | None,
    joinable_tables: list[dict[str, object]],
    reference_labels: list[str],
    output_root: Path,
    workspace_dir: Path,
):
    if not panel_specs:
        return mo.callout("No persisted projection coordinates are available for this geometry layout.", kind="warn")
    frames = []
    for spec in panel_specs:
        projection_id = str(spec.get("projection_id") or "")
        view_id = str(spec.get("view_id") or "")
        if not projection_id or not view_id:
            frames.append(pd.DataFrame())
            continue
        frames.append(load_projection_frame(view_id, projection_id, joinable_tables, output_root=output_root))
    if not any(not frame.empty for frame in frames):
        return mo.callout(
            "The selected geometry layout is declared, but none of its projections are materialized yet.",
            kind="warn",
        )

    n_panels = len(panel_specs)
    ncols = 1 if n_panels == 1 else 2
    nrows = int(math.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(7.2 * ncols, 4.9 * nrows))
    axes_array = np.atleast_1d(axes).reshape(nrows, ncols).ravel()
    numeric_frames = [
        frame[hue_column].dropna().astype(float)
        for frame in frames
        if hue_column is not None and hue_column in frame.columns and pd.api.types.is_numeric_dtype(frame[hue_column])
    ]
    numeric_vmin = None
    numeric_vmax = None
    if numeric_frames:
        numeric_vmin = float(min(series.min() for series in numeric_frames))
        numeric_vmax = float(max(series.max() for series in numeric_frames))
    category_values = sorted(
        {
            str(value)
            for frame in frames
            if hue_column is not None
            and hue_column in frame.columns
            and not pd.api.types.is_numeric_dtype(frame[hue_column])
            for value in frame[hue_column].fillna("NA").astype(str).unique()
        }
    )
    category_map = {
        category: CONTROL_PLANE_PALETTE[index % len(CONTROL_PLANE_PALETTE)]
        for index, category in enumerate(category_values)
    }
    scatter_artist = None
    for axis_index, (ax, spec, frame) in enumerate(zip(axes_array, panel_specs, frames, strict=True)):
        if frame.empty or "x" not in frame.columns or "y" not in frame.columns:
            ax.text(0.5, 0.5, "Projection missing", ha="center", va="center", fontsize=11, color="#5C6874")
            ax.set_axis_off()
            continue
        point_size, alpha = scatter_style(len(frame))
        if hue_column is not None and hue_column in frame.columns:
            hue_series = frame[hue_column]
            if pd.api.types.is_numeric_dtype(hue_series):
                valid = hue_series.notna()
                scatter_artist = ax.scatter(
                    frame.loc[valid, "x"].to_numpy(dtype=float),
                    frame.loc[valid, "y"].to_numpy(dtype=float),
                    c=hue_series.loc[valid].to_numpy(dtype=float),
                    cmap="viridis",
                    vmin=numeric_vmin,
                    vmax=numeric_vmax,
                    s=point_size,
                    alpha=alpha,
                    linewidths=0.0,
                )
                if (~valid).any():
                    ax.scatter(
                        frame.loc[~valid, "x"].to_numpy(dtype=float),
                        frame.loc[~valid, "y"].to_numpy(dtype=float),
                        c="#B9C3CD",
                        s=point_size,
                        alpha=0.45,
                        linewidths=0.0,
                    )
            else:
                for category in category_values:
                    mask = hue_series.fillna("NA").astype(str) == category
                    if not mask.any():
                        continue
                    ax.scatter(
                        frame.loc[mask, "x"].to_numpy(dtype=float),
                        frame.loc[mask, "y"].to_numpy(dtype=float),
                        c=category_map[category],
                        s=point_size,
                        alpha=alpha,
                        linewidths=0.0,
                        label=category,
                    )
        else:
            ax.scatter(
                frame["x"].to_numpy(dtype=float),
                frame["y"].to_numpy(dtype=float),
                c=CONTROL_PLANE_PALETTE[0],
                s=point_size,
                alpha=alpha,
                linewidths=0.0,
            )
        draw_reference_labels(ax, frame, reference_labels=reference_labels)
        ax.set_title(
            str(spec.get("title") or spec.get("view_id") or f"Panel {axis_index + 1}"),
            fontsize=11,
            fontweight="semibold",
        )
        ax.set_xlabel("Projection 1")
        ax.set_ylabel("Projection 2")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, color="#D5DCE4", linewidth=0.7, alpha=0.55)
        ax.set_axisbelow(True)
    for ax in axes_array[n_panels:]:
        ax.set_axis_off()
    if scatter_artist is not None and hue_column is not None and numeric_frames:
        fig.colorbar(
            scatter_artist,
            ax=axes_array[:n_panels].tolist(),
            shrink=0.86,
            label=display_hue_label(hue_column),
        )
    elif category_values and hue_column is not None:
        handles = [
            plt.Line2D([], [], linestyle="", marker="o", markersize=7, color=category_map[category], label=category)
            for category in category_values
        ]
        fig.legend(
            handles=handles,
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
            frameon=False,
            title=display_hue_label(hue_column),
        )
    return fig_to_image(fig)
