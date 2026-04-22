"""Live plot-review rendering helpers for hue-switchable notebook surfaces."""

from __future__ import annotations

import json
import math
from pathlib import Path

import marimo as mo
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..contracts.plot import ResolvedPlotSpec, metric_panel_uses_square_axes
from ..plots.render import (
    _category_color_map as static_category_color_map,
)
from ..plots.render import (
    _derived_panel_label,
    _render_curve_panel,
    _render_distribution_panel,
    _render_metric_panel,
    _render_placeholder_panel,
)
from ..plots.render import (
    _grid_figure_size as static_grid_figure_size,
)
from ..plots.render import (
    _panel_grid_dimensions as static_panel_grid_dimensions,
)
from ..visual_style import (
    TEXT_COLOR,
    compact_candidate_title,
    display_category_text,
    humanize_display_text,
    legend_layout,
    ordered_categories,
    wrap_plot_title,
)
from ..visual_style import scatter_style as shared_scatter_style
from .browser_runtime_projection import load_projection_frame, render_projection_grid
from .browser_runtime_support import (
    category_color_map as notebook_category_color_map,
)
from .browser_runtime_support import (
    classify_hue_series,
    continuous_hue_render_params,
    display_hue_label,
    draw_reference_labels,
    load_artifact_manifest,
    load_table,
    normalize_categorical_hue_series,
    render_matplotlib_figure,
    style_notebook_axes,
    style_notebook_legend,
)

_SINGLE_ROW_PANEL_PLOT_IDS = frozenset(
    {
        "design_centroid_margin_gallery",
        "representation_scree_diagnostic",
        "appendix_umap_gallery",
    }
)


def _load_error_frame(message: str) -> pd.DataFrame:
    frame = pd.DataFrame()
    frame.attrs["load_error"] = message
    return frame


def _frame_load_error(frame: pd.DataFrame) -> str:
    if not isinstance(getattr(frame, "attrs", None), dict):
        return ""
    return str(frame.attrs.get("load_error") or "").strip()


def _callout_from_frame_errors(frames: list[pd.DataFrame], *, fallback_message: str):
    unique_errors = list(dict.fromkeys(_frame_load_error(frame) for frame in frames if _frame_load_error(frame)))
    if unique_errors:
        return mo.callout("Live plot data are unavailable: " + "; ".join(unique_errors), kind="warn")
    return mo.callout(fallback_message, kind="warn")


def _projection_view_id(output_root: Path, projection_id: str) -> str | None:
    try:
        manifest = load_artifact_manifest(
            output_root / "projections" / projection_id,
            artifact_kind="projection",
            artifact_id=projection_id,
            allow_missing_status=True,
            allowed_statuses={"ok", "attention"},
        )
    except ValueError:
        return None
    return next(
        (
            str(item.get("id"))
            for item in manifest.get("inputs", [])
            if isinstance(item, dict) and item.get("kind") == "view_matrix" and str(item.get("id") or "").strip()
        ),
        None,
    )


def load_plot_review_frames(
    plot_spec: dict[str, object],
    *,
    joinable_tables: list[dict[str, object]],
    output_root: Path,
) -> list[pd.DataFrame]:
    kind = str(plot_spec.get("kind") or "")
    if kind == "projection_grid":
        requested_columns = [
            str(option.get("column"))
            for option in plot_spec.get("hue_options", [])
            if isinstance(option, dict) and option.get("column")
        ]
        frames: list[pd.DataFrame] = []
        for projection_id in plot_spec.get("projection_ids", []):
            view_id = _projection_view_id(output_root, str(projection_id))
            frame = load_projection_frame(
                view_id,
                str(projection_id),
                joinable_tables,
                output_root=output_root,
                required_columns=requested_columns,
                strict_required_columns=False,
            )
            if view_id is not None and not frame.empty:
                frame.attrs["view_id"] = view_id
            frames.append(frame)
        return frames
    if kind in {"xy_scatter_grid", "paired_xy_scatter_grid", "distribution_grid"}:
        frames: list[pd.DataFrame] = []
        for scalar_id in plot_spec.get("scalar_ids", []):
            try:
                frame = load_table(
                    output_root / "scalars" / str(scalar_id) / "table.parquet",
                    require_fresh_manifest=True,
                )
            except ValueError as exc:
                frame = _load_error_frame(str(exc))
            frames.append(frame)
        return frames
    if kind in {"metric_panel_grid", "categorical_count"}:
        scalar_id = str(plot_spec.get("scalar_id") or "")
        if not scalar_id:
            return []
        try:
            frame = load_table(output_root / "scalars" / scalar_id / "table.parquet", require_fresh_manifest=True)
        except ValueError as exc:
            frame = _load_error_frame(str(exc))
        return [frame]
    return []


def _prefer_single_row_panel_layout(plot_id: str | None, panel_count: int) -> bool:
    return bool(plot_id in _SINGLE_ROW_PANEL_PLOT_IDS and 1 < panel_count <= 4)


def _panel_grid_dimensions(panel_count: int, *, prefer_single_row: bool = False) -> tuple[int, int]:
    if panel_count <= 1:
        return 1, 1
    if prefer_single_row and panel_count <= 4:
        return 1, panel_count
    if panel_count == 5:
        return 2, 3
    if panel_count == 6:
        return 2, 3
    if panel_count in {7, 8}:
        return 2, 4
    if panel_count == 4:
        return 2, 2
    columns = min(4, panel_count)
    rows = int(math.ceil(panel_count / columns))
    return rows, columns


def _panel_figure_size(panel_count: int, *, prefer_single_row: bool = False) -> tuple[float, float]:
    rows, columns = _panel_grid_dimensions(panel_count, prefer_single_row=prefer_single_row)
    panel_width = 3.55 if prefer_single_row and columns >= 4 else 4.15
    panel_height = 4.2 if prefer_single_row and panel_count > 1 else 4.35
    return ((panel_width * columns) + 0.35, (panel_height * rows) + 0.2)


def _configured_hue_kinds(plot_spec: dict[str, object]) -> dict[str, str]:
    options = plot_spec.get("hue_options", [])
    return {
        str(option.get("column")): str(option.get("type"))
        for option in options
        if isinstance(option, dict) and option.get("column") and option.get("type")
    }


def _scatter_axis_label(frame: pd.DataFrame, *, value_column: str, label_column: str) -> str:
    if label_column in frame.columns:
        labels = {
            str(value).strip() for value in frame[label_column].dropna().astype(str).tolist() if str(value).strip()
        }
        if len(labels) == 1:
            return humanize_display_text(next(iter(labels)))
    return humanize_display_text(value_column)


def _shared_numeric_bounds(frames: list[pd.DataFrame], hue_column: str) -> tuple[float | None, float | None]:
    values = [
        pd.to_numeric(frame[hue_column], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        for frame in frames
        if hue_column in frame.columns
    ]
    if not values:
        return None, None
    combined = pd.concat(values, ignore_index=True)
    if combined.empty or combined.nunique() < 2:
        return None, None
    return float(combined.min()), float(combined.max())


def _continuous_hue_params(frames: list[pd.DataFrame], hue_column: str) -> dict[str, object]:
    values = [
        pd.to_numeric(frame[hue_column], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        for frame in frames
        if hue_column in frame.columns
    ]
    combined = pd.concat(values, ignore_index=True) if values else pd.Series(dtype=float)
    if combined.empty or combined.nunique() < 2:
        return {"cmap": "viridis", "norm": None, "vmin": None, "vmax": None}
    return continuous_hue_render_params(hue_column, combined)


def _categorical_hue_values(frames: list[pd.DataFrame], hue_column: str) -> list[str]:
    return ordered_categories(
        {
            str(value)
            for frame in frames
            if hue_column in frame.columns
            for value in normalize_categorical_hue_series(hue_column, frame[hue_column]).unique()
        },
        column=hue_column,
    )


def _plot_panel_title(plot_spec: dict[str, object], index: int, fallback: str) -> str:
    titles = plot_spec.get("panel_titles", [])
    if isinstance(titles, list) and index < len(titles):
        return str(titles[index])
    return fallback


def render_plot_review_surface(
    plot_spec: dict[str, object],
    *,
    frames: list[pd.DataFrame],
    hue_column: str | None,
    reference_labels: list[str],
    joinable_tables: list[dict[str, object]],
    output_root: Path,
    workspace_dir: Path,
):
    plot_alt_text = str(plot_spec.get("alt_text") or plot_spec.get("plot_id") or "latentdna live plot")
    resolved_plot_spec = {key: value for key, value in plot_spec.items() if key != "alt_text"}
    kind = str(resolved_plot_spec.get("kind") or "")
    if kind == "projection_grid":
        prefer_single_row = _prefer_single_row_panel_layout(
            str(resolved_plot_spec.get("plot_id") or ""),
            len(list(resolved_plot_spec.get("projection_ids", []))),
        )
        panel_specs = [
            {
                "view_id": str(
                    (frames[index].attrs.get("view_id") if index < len(frames) else None)
                    or _projection_view_id(output_root, str(projection_id))
                    or projection_id
                ),
                "projection_id": str(projection_id),
                "title": _plot_panel_title(resolved_plot_spec, index, str(projection_id)),
            }
            for index, projection_id in enumerate(resolved_plot_spec.get("projection_ids", []))
        ]
        return render_projection_grid(
            panel_specs,
            frames=frames,
            plot_id=str(resolved_plot_spec.get("plot_id") or ""),
            hue_column=hue_column,
            hue_kinds=_configured_hue_kinds(resolved_plot_spec),
            joinable_tables=joinable_tables,
            reference_labels=reference_labels,
            output_root=output_root,
            workspace_dir=workspace_dir,
            alt_text=plot_alt_text,
            prefer_single_row=prefer_single_row,
        )
    if kind in {"xy_scatter_grid", "paired_xy_scatter_grid"}:
        return _render_scatter_grid(
            resolved_plot_spec,
            frames=frames,
            hue_column=hue_column,
            reference_labels=reference_labels,
            alt_text=plot_alt_text,
        )
    if kind == "categorical_count":
        return _render_categorical_count_grid(resolved_plot_spec, frames=frames, alt_text=plot_alt_text)
    if kind == "metric_panel_grid":
        return _render_metric_grid(resolved_plot_spec, frames=frames, alt_text=plot_alt_text)
    if kind == "distribution_grid":
        return _render_distribution_grid(resolved_plot_spec, frames=frames, alt_text=plot_alt_text)
    if kind == "curve_grid":
        return _render_curve_grid(resolved_plot_spec, output_root=output_root, alt_text=plot_alt_text)
    return mo.callout("The selected plot does not support live notebook rendering.", kind="warn")


def _render_scatter_grid(
    plot_spec: dict[str, object],
    *,
    frames: list[pd.DataFrame],
    hue_column: str | None,
    reference_labels: list[str],
    alt_text: str,
):
    if not frames or not any(not frame.empty for frame in frames):
        return _callout_from_frame_errors(
            frames,
            fallback_message="The selected plot has no persisted scalar data to render.",
        )

    x_column = str(plot_spec.get("x_column") or "x")
    y_column = str(plot_spec.get("y_column") or "y")
    resolved_frames = [frame for frame in frames]
    hue_kinds = _configured_hue_kinds(plot_spec)

    effective_hue = hue_column if hue_column and any(hue_column in frame.columns for frame in resolved_frames) else None
    hue_kind = None
    if effective_hue is not None:
        hue_series = pd.concat(
            [frame[effective_hue] for frame in resolved_frames if effective_hue in frame.columns],
            ignore_index=True,
        )
        hue_kind = classify_hue_series(hue_series, configured_kind=hue_kinds.get(effective_hue))

    category_values = (
        _categorical_hue_values(resolved_frames, effective_hue) if hue_kind != "continuous" and effective_hue else []
    )
    category_map = notebook_category_color_map(category_values, column=effective_hue)
    numeric_vmin, numeric_vmax = (
        _shared_numeric_bounds(resolved_frames, effective_hue)
        if hue_kind == "continuous" and effective_hue
        else (None, None)
    )
    continuous_params = (
        _continuous_hue_params(resolved_frames, effective_hue)
        if hue_kind == "continuous" and effective_hue
        else {"cmap": "viridis", "norm": None, "vmin": None, "vmax": None}
    )
    if hue_kind == "continuous" and (numeric_vmin is None or numeric_vmax is None):
        effective_hue = None
        hue_kind = None

    panel_count = len(resolved_frames)
    plot_id = str(plot_spec.get("plot_id") or "")
    prefer_single_row = _prefer_single_row_panel_layout(str(plot_spec.get("plot_id") or ""), panel_count)
    rows, columns = _panel_grid_dimensions(panel_count, prefer_single_row=prefer_single_row)
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=_panel_figure_size(panel_count, prefer_single_row=prefer_single_row),
        squeeze=False,
    )
    axes_flat = axes.ravel()
    scatter_artist = None
    max_title_lines = 1
    annotation_frames: list[pd.DataFrame | None] = []

    for axis in axes_flat[panel_count:]:
        axis.set_axis_off()

    for index, (ax, frame) in enumerate(zip(axes_flat, resolved_frames, strict=False)):
        panel_title = compact_candidate_title(_plot_panel_title(plot_spec, index, f"Panel {index + 1}"))
        load_error = _frame_load_error(frame)
        if frame.empty or x_column not in frame.columns or y_column not in frame.columns:
            _render_placeholder_panel(
                ax,
                panel_title=panel_title,
                message="Panel unavailable" if load_error else "Panel data missing",
                detail=load_error or "The required coordinates are not available in this snapshot",
                square=True,
            )
            annotation_frames.append(None)
            continue

        finite_mask = (
            pd.to_numeric(frame[x_column], errors="coerce").replace([np.inf, -np.inf], np.nan).notna()
            & pd.to_numeric(frame[y_column], errors="coerce").replace([np.inf, -np.inf], np.nan).notna()
        )
        finite_frame = frame.loc[finite_mask].copy()
        if finite_frame.empty:
            _render_placeholder_panel(
                ax,
                panel_title=panel_title,
                message="Margins unavailable",
                detail="No finite values in this snapshot",
                square=True,
            )
            ax.set_xlabel(
                wrap_plot_title(
                    _scatter_axis_label(frame, value_column=x_column, label_column="x_display_name"),
                    width=28,
                    max_lines=2,
                )
            )
            ax.set_ylabel(
                wrap_plot_title(
                    _scatter_axis_label(frame, value_column=y_column, label_column="y_display_name"),
                    width=28,
                    max_lines=2,
                )
            )
            annotation_frames.append(None)
            continue

        point_style = shared_scatter_style(len(finite_frame))
        x_values = finite_frame[x_column].to_numpy(dtype=float)
        y_values = finite_frame[y_column].to_numpy(dtype=float)
        x_span = float(np.ptp(np.asarray(x_values, dtype=np.float64))) if x_values.size else 0.0
        y_span = float(np.ptp(np.asarray(y_values, dtype=np.float64))) if y_values.size else 0.0
        collapsed_panel = x_span <= 1e-12 and y_span <= 1e-12
        if effective_hue is None or effective_hue not in frame.columns:
            if collapsed_panel:
                centroid_x = float(x_values[0])
                centroid_y = float(y_values[0])
                ax.scatter(
                    [centroid_x],
                    [centroid_y],
                    c="#111111",
                    s=max(point_style.point_size * 18.0, 90.0),
                    alpha=0.92,
                    linewidths=0.7,
                    edgecolors="white",
                    rasterized=point_style.rasterized,
                )
                ax.set_xlim(centroid_x - 0.055, centroid_x + 0.055)
                ax.set_ylim(centroid_y - 0.055, centroid_y + 0.055)
                ax.text(
                    0.5,
                    0.93,
                    "Collapsed to one point",
                    transform=ax.transAxes,
                    ha="center",
                    va="top",
                    fontsize=9.0,
                    color="#5C6874",
                )
            else:
                ax.scatter(
                    x_values,
                    y_values,
                    c="#0072B2",
                    s=point_style.point_size,
                    alpha=point_style.alpha,
                    linewidths=point_style.linewidths,
                    edgecolors=point_style.edgecolors,
                    rasterized=point_style.rasterized,
                )
        elif hue_kind == "continuous":
            hue_values = pd.to_numeric(finite_frame[effective_hue], errors="coerce")
            valid = hue_values.notna()
            if collapsed_panel:
                centroid_x = float(x_values[0])
                centroid_y = float(y_values[0])
                collapsed_color = float(hue_values.loc[valid].iloc[0]) if valid.any() else float(numeric_vmin or 0.0)
                scatter_artist = ax.scatter(
                    [centroid_x],
                    [centroid_y],
                    c=[collapsed_color],
                    cmap=str(continuous_params["cmap"]),
                    norm=continuous_params["norm"],
                    vmin=None if continuous_params["norm"] is not None else continuous_params["vmin"],
                    vmax=None if continuous_params["norm"] is not None else continuous_params["vmax"],
                    s=max(point_style.point_size * 18.0, 90.0),
                    alpha=0.92,
                    linewidths=0.7,
                    edgecolors="white",
                    rasterized=point_style.rasterized,
                )
                ax.set_xlim(centroid_x - 0.055, centroid_x + 0.055)
                ax.set_ylim(centroid_y - 0.055, centroid_y + 0.055)
                ax.text(
                    0.5,
                    0.93,
                    "Collapsed to one point",
                    transform=ax.transAxes,
                    ha="center",
                    va="top",
                    fontsize=9.0,
                    color="#5C6874",
                )
            elif valid.any():
                scatter_artist = ax.scatter(
                    finite_frame.loc[valid, x_column].to_numpy(dtype=float),
                    finite_frame.loc[valid, y_column].to_numpy(dtype=float),
                    c=hue_values.loc[valid].to_numpy(dtype=float),
                    cmap=str(continuous_params["cmap"]),
                    norm=continuous_params["norm"],
                    vmin=None if continuous_params["norm"] is not None else continuous_params["vmin"],
                    vmax=None if continuous_params["norm"] is not None else continuous_params["vmax"],
                    s=point_style.point_size,
                    alpha=point_style.alpha,
                    linewidths=point_style.linewidths,
                    edgecolors=point_style.edgecolors,
                    rasterized=point_style.rasterized,
                )
        else:
            hue_values = normalize_categorical_hue_series(effective_hue, finite_frame[effective_hue])
            if collapsed_panel:
                centroid_x = float(x_values[0])
                centroid_y = float(y_values[0])
                collapsed_category = str(hue_values.iloc[0]) if not hue_values.empty else category_values[0]
                ax.scatter(
                    [centroid_x],
                    [centroid_y],
                    c=category_map.get(collapsed_category, "#111111"),
                    s=max(point_style.point_size * 18.0, 90.0),
                    alpha=0.92,
                    linewidths=0.7,
                    edgecolors="white",
                    rasterized=point_style.rasterized,
                )
                ax.set_xlim(centroid_x - 0.055, centroid_x + 0.055)
                ax.set_ylim(centroid_y - 0.055, centroid_y + 0.055)
                ax.text(
                    0.5,
                    0.93,
                    "Collapsed to one point",
                    transform=ax.transAxes,
                    ha="center",
                    va="top",
                    fontsize=9.0,
                    color="#5C6874",
                )
            else:
                for category in category_values:
                    mask = hue_values == category
                    if not mask.any():
                        continue
                    ax.scatter(
                        finite_frame.loc[mask, x_column].to_numpy(dtype=float),
                        finite_frame.loc[mask, y_column].to_numpy(dtype=float),
                        c=category_map[category],
                        s=point_style.point_size,
                        alpha=point_style.alpha,
                        linewidths=point_style.linewidths,
                        edgecolors=point_style.edgecolors,
                        rasterized=point_style.rasterized,
                    )

        if x_values.size and float(x_values.min()) < 0.0 < float(x_values.max()):
            ax.axvline(0.0, color="#94A3B8", linewidth=0.9, linestyle="--", alpha=0.9, zorder=0)
        if y_values.size and float(y_values.min()) < 0.0 < float(y_values.max()):
            ax.axhline(0.0, color="#94A3B8", linewidth=0.9, linestyle="--", alpha=0.9, zorder=0)

        wrapped_title = wrap_plot_title(
            panel_title,
            width=24,
            max_lines=3,
        )
        max_title_lines = max(max_title_lines, wrapped_title.count("\n") + 1)
        ax.set_title(wrapped_title, pad=10 if "\n" in wrapped_title else 8)
        ax.set_xlabel(
            wrap_plot_title(
                _scatter_axis_label(frame, value_column=x_column, label_column="x_display_name"),
                width=28,
                max_lines=2,
            )
        )
        ax.set_ylabel(
            wrap_plot_title(
                _scatter_axis_label(frame, value_column=y_column, label_column="y_display_name"),
                width=28,
                max_lines=2,
            )
        )
        style_notebook_axes(ax, grid=True, square=True)
        annotation_frames.append(finite_frame)

    legend_bottom = 0.0
    label_right_padding_px = 12.0
    colorbar_bottom = 0.0
    if category_values and effective_hue is not None:
        legend_labels = [display_category_text(category, column=effective_hue) for category in category_values]
        layout = legend_layout(
            legend_labels,
            plot_id=plot_id,
            default_anchor_y=0.012 if plot_id == "design_centroid_margin_gallery" else 0.02,
            default_base_margin=0.11,
            row_step=0.043,
            single_row=True,
        )
        legend = fig.legend(
            handles=[
                plt.Line2D(
                    [],
                    [],
                    linestyle="",
                    marker="o",
                    markersize=7,
                    color=category_map[category],
                    label=label,
                )
                for category, label in zip(category_values, legend_labels, strict=True)
            ],
            loc="lower center",
            bbox_to_anchor=(0.5, layout.anchor_y),
            ncol=layout.columns,
            frameon=False,
            borderaxespad=0.0,
            columnspacing=1.05,
            handletextpad=0.5,
        )
        style_notebook_legend(legend)
        legend_bottom = layout.bottom_margin
        if plot_id == "design_centroid_margin_gallery":
            legend_bottom = max(legend_bottom + 0.015, 0.125)
    if scatter_artist is not None and effective_hue is not None and hue_kind == "continuous":
        label_right_padding_px = 28.0
        colorbar_bottom = 0.12
        bottom_margin = max(legend_bottom, 0.2 if panel_count > 1 else 0.16)
    else:
        bottom_margin = max(legend_bottom, 0.08)

    top_margin = max(0.8, 0.96 - (0.042 * max(max_title_lines - 1, 0)))
    fig.subplots_adjust(
        left=0.09,
        right=0.98,
        top=top_margin,
        bottom=bottom_margin,
        wspace=(
            0.31
            if plot_id == "design_centroid_margin_gallery" and panel_count > 1
            else 0.24
            if panel_count > 1
            else 0.18
        ),
        hspace=(0.62 + (0.04 * max(max_title_lines - 1, 0))) if panel_count > 1 else 0.3,
    )

    if scatter_artist is not None and effective_hue is not None and hue_kind == "continuous":
        colorbar_width = 0.66 if panel_count > 1 else 0.56
        colorbar_left = (1.0 - colorbar_width) / 2.0
        colorbar = fig.colorbar(
            scatter_artist,
            cax=fig.add_axes([colorbar_left, colorbar_bottom, colorbar_width, 0.028]),
            orientation="horizontal",
        )
        colorbar.set_label(display_hue_label(effective_hue), fontsize=11.5, color=TEXT_COLOR)
        colorbar.ax.tick_params(labelsize=11.5, colors=TEXT_COLOR)
        colorbar.ax.xaxis.set_label_position("bottom")
        colorbar.ax.xaxis.set_ticks_position("bottom")
    fig.canvas.draw()

    for ax, frame in zip(axes_flat, annotation_frames, strict=False):
        if ax.axison and frame is not None and not frame.empty:
            draw_reference_labels(
                ax,
                frame,
                reference_labels=reference_labels,
                x_column=x_column,
                y_column=y_column,
                right_padding_px=label_right_padding_px,
                left_padding_px=28.0,
            )
    return render_matplotlib_figure(fig, alt=alt_text)


def _render_metric_grid(plot_spec: dict[str, object], *, frames: list[pd.DataFrame], alt_text: str):
    if not frames or frames[0].empty:
        return _callout_from_frame_errors(
            frames,
            fallback_message="The selected plot has no persisted scalar data to render.",
        )

    frame = frames[0]
    spec = ResolvedPlotSpec.model_validate(plot_spec)
    value_column = spec.value_column if spec.value_column and spec.value_column in frame.columns else None
    if value_column is None:
        value_column = "metric_value" if "metric_value" in frame.columns else "row_count"
    spec = spec.model_copy(update={"value_column": value_column})

    panel_values = list(dict.fromkeys(frame[spec.row_column].astype(str).tolist())) if spec.row_column else ["panel"]
    rows_count, columns = static_panel_grid_dimensions(len(panel_values))
    square_metric_panels = metric_panel_uses_square_axes(spec.plot_id)
    metric_figsize = static_grid_figure_size(len(panel_values), square_panels=square_metric_panels)
    if spec.plot_id == "representation_health_summary":
        metric_figsize = (metric_figsize[0] + (1.45 * columns), metric_figsize[1])
    fig, axes = plt.subplots(
        rows_count,
        columns,
        figsize=metric_figsize,
        squeeze=False,
    )
    records = frame.to_dict(orient="records")
    color_map, categories = static_category_color_map([records], spec.color_column)
    for axis in axes.ravel()[len(panel_values) :]:
        axis.set_axis_off()
    for axis, panel_value in zip(axes.ravel(), panel_values, strict=False):
        panel_rows = [row for row in records if str(row.get(spec.row_column)) == panel_value]
        panel_title = str(panel_rows[0].get(spec.panel_column) or panel_value) if panel_rows else panel_value
        _render_metric_panel(
            axis,
            rows=panel_rows,
            spec=spec,
            panel_title=panel_title,
            color_map=color_map,
            square=square_metric_panels,
        )
    legend_bottom = 0.0
    if categories and spec.color_column is not None:
        legend_labels = [display_category_text(category, column=spec.color_column) for category in categories]
        layout = legend_layout(
            legend_labels,
            plot_id=spec.plot_id,
            default_anchor_y=0.02,
            default_base_margin=0.11,
            row_step=0.043,
            single_row=True,
        )
        legend = fig.legend(
            handles=[
                plt.Line2D(
                    [],
                    [],
                    linestyle="",
                    marker="o",
                    markersize=7,
                    color=color_map[category],
                    label=label,
                )
                for category, label in zip(categories, legend_labels, strict=True)
            ],
            loc="lower center",
            bbox_to_anchor=(0.5, layout.anchor_y),
            ncol=layout.columns,
            frameon=False,
            borderaxespad=0.0,
            columnspacing=1.05,
            handletextpad=0.5,
        )
        style_notebook_legend(legend)
        legend_bottom = layout.bottom_margin
    fig.tight_layout(
        rect=(0.0, legend_bottom, 1.0, 1.0),
        pad=0.95,
        h_pad=1.4,
        w_pad=1.85 if spec.plot_id == "representation_health_summary" else 0.95,
    )
    return render_matplotlib_figure(fig, alt=alt_text)


def _render_distribution_grid(plot_spec: dict[str, object], *, frames: list[pd.DataFrame], alt_text: str):
    if not frames or not any(not frame.empty for frame in frames):
        return _callout_from_frame_errors(
            frames,
            fallback_message="The selected plot has no persisted scalar data to render.",
        )

    spec = ResolvedPlotSpec.model_validate(plot_spec)
    metric_columns = list(spec.metric_columns or [])
    panel_entries: list[tuple[str, pd.DataFrame | None, str | None, str | None]] = []
    panel_titles = list(spec.panel_titles or [])
    panel_title_index = 0

    def next_panel_title(*, scalar_id: str, metric_column: str | None) -> str:
        nonlocal panel_title_index
        if panel_title_index < len(panel_titles):
            title = str(panel_titles[panel_title_index])
        else:
            title = (
                f"{_derived_panel_label(scalar_id)} · {humanize_display_text(metric_column)}"
                if metric_column and _derived_panel_label(scalar_id)
                else humanize_display_text(metric_column or scalar_id)
            )
        panel_title_index += 1
        return title

    for scalar_id, frame in zip(spec.scalar_ids, frames, strict=False):
        load_error = _frame_load_error(frame)
        numeric_columns = [
            column
            for column in frame.columns
            if pd.api.types.is_numeric_dtype(frame[column]) and column not in {"left_indices", "right_indices"}
        ]
        if metric_columns:
            for metric_column in metric_columns:
                panel_title = next_panel_title(scalar_id=str(scalar_id), metric_column=metric_column)
                if frame.empty:
                    panel_entries.append((panel_title, None, None, load_error or "Panel data missing"))
                    continue
                if metric_column not in numeric_columns:
                    panel_entries.append(
                        (
                            panel_title,
                            None,
                            None,
                            f"`{metric_column}` is missing or non-numeric in `{scalar_id}`",
                        )
                    )
                    continue
                panel_entries.append((panel_title, frame, metric_column, None))
        else:
            metric_column = (
                spec.value_column
                if spec.value_column in numeric_columns
                else (numeric_columns[0] if numeric_columns else None)
            )
            panel_title = next_panel_title(scalar_id=str(scalar_id), metric_column=metric_column)
            if frame.empty:
                panel_entries.append((panel_title, None, None, load_error or "Panel data missing"))
                continue
            if metric_column is None:
                panel_entries.append((panel_title, None, None, f"no numeric columns are available in `{scalar_id}`"))
                continue
            panel_entries.append((panel_title, frame, metric_column, None))
    if not any(frame is not None and metric_column is not None for _, frame, metric_column, _ in panel_entries):
        return _callout_from_frame_errors(
            frames,
            fallback_message="The selected plot has no numeric distributions to render.",
        )

    rows_count, columns = static_panel_grid_dimensions(len(panel_entries))
    square_distribution_panels = spec.plot_id == "context_delta_distributions"
    fig, axes = plt.subplots(
        rows_count,
        columns,
        figsize=static_grid_figure_size(len(panel_entries), square_panels=square_distribution_panels),
        squeeze=False,
    )
    for axis in axes.ravel()[len(panel_entries) :]:
        axis.set_axis_off()
    for axis, (panel_title, frame, metric_column, error_detail) in zip(axes.ravel(), panel_entries, strict=False):
        if frame is None or metric_column is None:
            _render_placeholder_panel(
                axis,
                panel_title=panel_title,
                message="Panel unavailable",
                detail=wrap_plot_title(str(error_detail or "Panel data missing"), width=34, max_lines=4),
                square=square_distribution_panels,
            )
            continue
        _render_distribution_panel(
            axis,
            rows=frame.to_dict(orient="records"),
            metric_column=metric_column,
            color_column=spec.color_column,
            render_mode=spec.render_mode or "histogram",
            panel_title=panel_title,
            square=square_distribution_panels,
        )
    fig.tight_layout(pad=0.95, h_pad=1.4, w_pad=0.95)
    return render_matplotlib_figure(fig, alt=alt_text)


def _render_curve_grid(plot_spec: dict[str, object], *, output_root: Path, alt_text: str):
    spec = ResolvedPlotSpec.model_validate(plot_spec)
    reducer_summaries: list[tuple[str, dict[str, object]]] = []
    for reducer_id in spec.reducer_ids:
        summary_path = output_root / "reducers" / str(reducer_id) / "summary.json"
        if not summary_path.is_file():
            continue
        reducer_summaries.append((str(reducer_id), json.loads(summary_path.read_text(encoding="utf-8"))))
    if not reducer_summaries:
        return mo.callout("The selected plot has no persisted reducer summaries to render.", kind="warn")

    prefer_single_row = _prefer_single_row_panel_layout(spec.plot_id, len(reducer_summaries))
    rows_count, columns = static_panel_grid_dimensions(
        len(reducer_summaries),
        prefer_single_row=prefer_single_row,
    )
    square_curve_panels = spec.plot_id == "representation_scree_diagnostic"
    fig, axes = plt.subplots(
        rows_count,
        columns,
        figsize=static_grid_figure_size(
            len(reducer_summaries),
            square_panels=square_curve_panels,
            prefer_single_row=prefer_single_row,
        ),
        squeeze=False,
    )
    titles = spec.panel_titles or [humanize_display_text(reducer_id) for reducer_id, _ in reducer_summaries]
    for axis in axes.ravel()[len(reducer_summaries) :]:
        axis.set_axis_off()
    for axis, (reducer_id, summary), panel_title in zip(axes.ravel(), reducer_summaries, titles, strict=False):
        _render_curve_panel(
            axis,
            reducer_id=reducer_id,
            summary=summary,
            panel_title=panel_title,
            square=square_curve_panels,
            show_legend=False,
        )
    legend_labels = ["Explained variance ratio", "Cumulative variance ratio"]
    layout = legend_layout(
        legend_labels,
        plot_id=spec.plot_id,
        default_anchor_y=0.02,
        default_base_margin=0.11,
        row_step=0.043,
        single_row=True,
    )
    legend = fig.legend(
        handles=[
            plt.Line2D([], [], marker="o", linewidth=1.8, color="#0072B2", label=legend_labels[0]),
            plt.Line2D([], [], marker="s", linewidth=1.8, color="#009E73", label=legend_labels[1]),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, layout.anchor_y),
        ncol=layout.columns,
        frameon=False,
        borderaxespad=0.0,
        columnspacing=1.1,
        handletextpad=0.5,
    )
    style_notebook_legend(legend)
    fig.tight_layout(rect=(0.0, layout.bottom_margin, 1.0, 1.0), pad=0.95, h_pad=1.4, w_pad=0.95)
    return render_matplotlib_figure(fig, alt=alt_text)


def _render_categorical_count_grid(plot_spec: dict[str, object], *, frames: list[pd.DataFrame], alt_text: str):
    if not frames or frames[0].empty:
        return _callout_from_frame_errors(
            frames,
            fallback_message="The selected plot has no persisted scalar data to render.",
        )

    frame = frames[0].copy()
    spec = ResolvedPlotSpec.model_validate(plot_spec)
    required_columns = [spec.row_column, spec.column_column, spec.value_column]
    missing_columns = [column for column in required_columns if column is not None and column not in frame.columns]
    if missing_columns:
        return mo.callout(
            f"The selected plot is missing required categorical-count columns: {', '.join(missing_columns)}.",
            kind="warn",
        )

    if spec.panel_column and spec.panel_column in frame.columns:
        panel_values = list(dict.fromkeys(frame[spec.panel_column].astype(str).tolist()))
    else:
        panel_values = [str(plot_spec.get("plot_id") or "panel")]

    square_count_panels = spec.plot_id == "dataset_overview"
    if square_count_panels and len(panel_values) <= 3:
        rows_count, columns = 1, len(panel_values)
    elif len(panel_values) <= 3 and not square_count_panels:
        rows_count, columns = len(panel_values), 1
    else:
        rows_count, columns = static_panel_grid_dimensions(len(panel_values))
    fig, axes = plt.subplots(
        rows_count,
        columns,
        figsize=(
            ((4.0 * columns) + 0.35, 4.55)
            if square_count_panels and rows_count == 1
            else static_grid_figure_size(len(panel_values), square_panels=True)
            if square_count_panels
            else (7.2 * columns, 3.85 * rows_count)
        ),
        squeeze=False,
    )

    for axis in axes.ravel()[len(panel_values) :]:
        axis.set_axis_off()

    for axis, panel_value in zip(axes.ravel(), panel_values, strict=False):
        if spec.panel_column and spec.panel_column in frame.columns:
            panel_frame = frame.loc[frame[spec.panel_column].astype(str) == panel_value].copy()
        else:
            panel_frame = frame.copy()
        if panel_frame.empty:
            axis.set_axis_off()
            continue
        if "order" in panel_frame.columns:
            panel_frame = panel_frame.sort_values("order", kind="stable")

        labels = [humanize_display_text(value) for value in panel_frame[spec.column_column].tolist()]
        values = pd.to_numeric(panel_frame[spec.value_column], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        y_positions = np.arange(len(panel_frame), dtype=float)
        color_map = notebook_category_color_map(labels)
        bar_colors = [color_map[label] for label in labels]

        axis.barh(
            y_positions,
            values,
            color=bar_colors,
            edgecolor="white",
            linewidth=0.6,
            alpha=0.92,
        )
        axis.set_yticks(y_positions, labels)
        axis.invert_yaxis()
        axis.set_ylabel("")
        axis.set_xlabel("Percent of N")
        show_as_percent = all(0.0 <= value <= 1.0 for value in values)
        if show_as_percent:
            from matplotlib.ticker import PercentFormatter

            axis.xaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        max_value = max(values, default=0.0)
        axis.set_xlim(0.0, max_value * 1.14 if max_value > 0 else 1.0)

        denominator_values = {
            int(float(value))
            for value in panel_frame.get("denominator", pd.Series(dtype=float)).tolist()
            if value is not None and not pd.isna(value)
        }
        panel_title = humanize_display_text(panel_value)
        if len(denominator_values) == 1:
            panel_title = f"{panel_title}\nN = {next(iter(denominator_values)):,}"
        axis.set_title(wrap_plot_title(panel_title, width=24), pad=10)
        style_notebook_axes(axis, grid=True, square=square_count_panels)

        counts = pd.to_numeric(panel_frame.get("count", pd.Series([np.nan] * len(panel_frame))), errors="coerce")
        percents = pd.to_numeric(panel_frame.get("percent", pd.Series([np.nan] * len(panel_frame))), errors="coerce")
        for index, (y_position, value) in enumerate(zip(y_positions, values, strict=False)):
            label_parts: list[str] = []
            if index < len(counts) and not pd.isna(counts.iloc[index]):
                label_parts.append(f"{int(counts.iloc[index]):,}")
            if index < len(percents) and not pd.isna(percents.iloc[index]):
                label_parts.append(f"{float(percents.iloc[index]):.1f}%")
            axis.text(
                value + (max_value * 0.02 if max_value > 0 else 0.03),
                y_position,
                " | ".join(label_parts) if label_parts else f"{value:.1%}",
                va="center",
                ha="left",
                fontsize=11.0,
                color=TEXT_COLOR,
            )

    fig.tight_layout(
        pad=0.95,
        h_pad=1.5,
        w_pad=1.2 if square_count_panels and rows_count == 1 else 0.95,
    )
    return render_matplotlib_figure(fig, alt=alt_text)
