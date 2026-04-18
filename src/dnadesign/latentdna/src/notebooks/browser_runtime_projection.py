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

from ..visual_style import (
    PUBLICATION_PALETTE,
    TEXT_COLOR,
    humanize_display_text,
    wrap_plot_title,
)
from ..visual_style import (
    scatter_style as shared_scatter_style,
)
from .browser_runtime_support import (
    available_hues_for_frames,
    category_color_map,
    classify_hue_series,
    display_hue_label,
    draw_reference_labels,
    load_table,
    render_matplotlib_figure,
    shared_join_key,
    style_notebook_axes,
    style_notebook_legend,
)


def _assert_unique_join_key(table: pd.DataFrame, join_key: str, *, artifact_id: str) -> None:
    duplicates = table[join_key][table[join_key].duplicated()].astype(str).tolist()
    if duplicates:
        preview = ", ".join(duplicates[:5])
        raise ValueError(f"duplicate metadata join keys for `{join_key}` in `{artifact_id}`: {preview}")


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
        _assert_unique_join_key(table, join_key, artifact_id=artifact_id)
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


def _layout_frames(
    panel_specs: list[dict[str, object]],
    *,
    frames: list[pd.DataFrame] | None,
    joinable_tables: list[dict[str, object]],
    output_root: Path,
) -> list[pd.DataFrame]:
    if frames is not None:
        return list(frames)
    loaded: list[pd.DataFrame] = []
    for spec in panel_specs:
        projection_id = str(spec.get("projection_id") or "")
        view_id = str(spec.get("view_id") or "")
        if not projection_id or not view_id:
            loaded.append(pd.DataFrame())
            continue
        loaded.append(load_projection_frame(view_id, projection_id, joinable_tables, output_root=output_root))
    return loaded


def _panel_grid_dimensions(panel_count: int) -> tuple[int, int]:
    if panel_count <= 1:
        return 1, 1
    if panel_count == 4:
        return 2, 2
    if panel_count == 8:
        return 2, 4
    columns = min(3, panel_count)
    rows = int(math.ceil(panel_count / columns))
    return rows, columns


def render_projection_grid(
    panel_specs: list[dict[str, object]],
    *,
    frames: list[pd.DataFrame] | None = None,
    hue_column: str | None,
    hue_kinds: dict[str, str] | None,
    joinable_tables: list[dict[str, object]],
    reference_labels: list[str],
    output_root: Path,
    workspace_dir: Path,
):
    del workspace_dir
    if not panel_specs:
        return mo.callout("No persisted projection coordinates are available for this geometry layout.", kind="warn")

    resolved_frames = _layout_frames(
        panel_specs,
        frames=frames,
        joinable_tables=joinable_tables,
        output_root=output_root,
    )
    if not any(not frame.empty for frame in resolved_frames):
        return mo.callout(
            "The selected geometry layout is declared, but none of its projections are materialized yet.",
            kind="warn",
        )

    effective_hue = hue_column
    if effective_hue:
        allowed = available_hues_for_frames(
            resolved_frames,
            preferred_hues=[effective_hue],
            hue_kinds=hue_kinds or {},
        )
        if effective_hue not in allowed:
            effective_hue = None

    n_panels = len(panel_specs)
    nrows, ncols = _panel_grid_dimensions(n_panels)
    panel_size = 6.1 if n_panels == 1 else 3.65
    panel_height = panel_size + (0.1 if n_panels == 1 else 0.16)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=((panel_size * ncols) + 0.45, panel_height * nrows),
    )
    axes_array = np.atleast_1d(axes).reshape(nrows, ncols).ravel()

    hue_kind = None
    if effective_hue is not None:
        configured_hue_kind = (hue_kinds or {}).get(effective_hue)
        hue_kind = classify_hue_series(
            pd.concat(
                [frame[effective_hue] for frame in resolved_frames if effective_hue in frame.columns],
                ignore_index=True,
            ),
            configured_kind=configured_hue_kind,
        )
    treat_as_categorical = hue_kind in {"categorical", "binary"}

    numeric_frames = [
        pd.to_numeric(frame[effective_hue], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        for frame in resolved_frames
        if effective_hue is not None and hue_kind == "continuous" and effective_hue in frame.columns
    ]
    numeric_vmin = None
    numeric_vmax = None
    if numeric_frames:
        combined_numeric = pd.concat(numeric_frames, ignore_index=True)
        if combined_numeric.nunique() >= 2:
            numeric_vmin = float(combined_numeric.min())
            numeric_vmax = float(combined_numeric.max())
        else:
            effective_hue = None
            hue_kind = None

    category_values = sorted(
        {
            str(value)
            for frame in resolved_frames
            if effective_hue is not None and treat_as_categorical and effective_hue in frame.columns
            for value in frame[effective_hue].fillna("NA").astype(str).unique()
        }
    )
    category_map = category_color_map(category_values)

    scatter_artist = None
    max_title_lines = 1
    for axis_index, (ax, spec, frame) in enumerate(zip(axes_array, panel_specs, resolved_frames, strict=True)):
        if frame.empty or "x" not in frame.columns or "y" not in frame.columns:
            ax.text(0.5, 0.5, "Projection missing", ha="center", va="center", fontsize=11, color="#5C6874")
            ax.set_axis_off()
            continue

        point_style = shared_scatter_style(len(frame))
        if effective_hue is None or effective_hue not in frame.columns:
            ax.scatter(
                frame["x"].to_numpy(dtype=float),
                frame["y"].to_numpy(dtype=float),
                c=PUBLICATION_PALETTE[0],
                s=point_style.point_size,
                alpha=point_style.alpha,
                linewidths=point_style.linewidths,
                edgecolors=point_style.edgecolors,
                rasterized=point_style.rasterized,
            )
        elif hue_kind == "continuous":
            hue_series = pd.to_numeric(frame[effective_hue], errors="coerce")
            valid = hue_series.notna()
            scatter_artist = ax.scatter(
                frame.loc[valid, "x"].to_numpy(dtype=float),
                frame.loc[valid, "y"].to_numpy(dtype=float),
                c=hue_series.loc[valid].to_numpy(dtype=float),
                cmap="cividis",
                vmin=numeric_vmin,
                vmax=numeric_vmax,
                s=point_style.point_size,
                alpha=point_style.alpha,
                linewidths=point_style.linewidths,
                edgecolors=point_style.edgecolors,
                rasterized=point_style.rasterized,
            )
        else:
            hue_series = frame[effective_hue].fillna("NA").astype(str)
            for category in category_values:
                mask = hue_series == category
                if not mask.any():
                    continue
                ax.scatter(
                    frame.loc[mask, "x"].to_numpy(dtype=float),
                    frame.loc[mask, "y"].to_numpy(dtype=float),
                    c=category_map[category],
                    s=point_style.point_size,
                    alpha=point_style.alpha,
                    linewidths=point_style.linewidths,
                    edgecolors=point_style.edgecolors,
                    rasterized=point_style.rasterized,
                    label=category,
                )

        wrapped_title = wrap_plot_title(
            humanize_display_text(str(spec.get("title") or spec.get("view_id") or f"Panel {axis_index + 1}")),
            width=34 if n_panels == 1 else 26,
        )
        max_title_lines = max(max_title_lines, wrapped_title.count("\n") + 1)
        ax.set_title(wrapped_title, fontweight="semibold", pad=10 if "\n" in wrapped_title else 8)
        ax.set_xlabel("Projection 1")
        ax.set_ylabel("Projection 2")
        style_notebook_axes(ax, grid=True, square=True)

    for ax in axes_array[n_panels:]:
        ax.set_axis_off()

    bottom_margin = 0.085
    if category_values and effective_hue is not None:
        legend = fig.legend(
            handles=[
                plt.Line2D(
                    [],
                    [],
                    linestyle="",
                    marker="o",
                    markersize=7,
                    color=category_map[category],
                    label=humanize_display_text(category),
                )
                for category in category_values
            ],
            loc="lower center",
            bbox_to_anchor=(0.5, 0.02),
            frameon=False,
            ncol=max(1, len(category_values)),
            borderaxespad=0.0,
            columnspacing=0.95,
            handletextpad=0.45,
        )
        style_notebook_legend(legend)
        bottom_margin = 0.11

    right_margin = 0.97
    label_right_padding_px = 12.0
    if scatter_artist is not None and effective_hue is not None and hue_kind == "continuous":
        right_margin = 0.82
        label_right_padding_px = 80.0
    top_margin = max(0.8, 0.96 - (0.042 * max(max_title_lines - 1, 0)))
    fig.subplots_adjust(
        left=0.08,
        right=right_margin,
        top=top_margin,
        bottom=bottom_margin,
        wspace=0.26 if n_panels > 1 else 0.2,
        hspace=(0.62 + (0.04 * max(max_title_lines - 1, 0))) if n_panels > 1 else 0.3,
    )
    fig.canvas.draw()

    for ax, frame in zip(axes_array, resolved_frames, strict=True):
        if ax.axison and not frame.empty:
            draw_reference_labels(
                ax,
                frame,
                reference_labels=reference_labels,
                right_padding_px=label_right_padding_px,
                left_padding_px=12.0,
            )

    if scatter_artist is not None and effective_hue is not None and hue_kind == "continuous":
        colorbar = fig.colorbar(
            scatter_artist,
            ax=axes_array[:n_panels].tolist(),
            shrink=0.84,
            pad=0.02,
            fraction=0.05,
            label=display_hue_label(effective_hue),
        )
        colorbar.ax.tick_params(labelsize=10.5, colors=TEXT_COLOR)
        colorbar.set_label(display_hue_label(effective_hue), fontsize=11.5, color=TEXT_COLOR)

    return render_matplotlib_figure(fig, alt="Latent geometry projection grid")
