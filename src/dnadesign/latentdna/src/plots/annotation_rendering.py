"""Matplotlib drawing helpers for static annotation overlays."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..annotation_layout import choose_annotation_placement
from ..contracts.errors import ContractViolationError
from ..contracts.plot import ResolvedPlotSpec
from ..visual_style import ANNOTATION_LABEL_BOX_ALPHA, SPINE_COLOR, TEXT_COLOR, humanize_display_text
from ..workspaces.loader import WorkspaceContext
from .annotations import annotation_label_text
from .axes import explicit_axis_label
from .renderers.scatter import coerce_finite_float, color_series
from .tables import require_row_columns


def _annotation_hue_column(spec: ResolvedPlotSpec) -> str | None:
    annotation = getattr(spec, "annotation", None)
    if annotation is None:
        return None
    column = getattr(annotation, "hue_column", None)
    return str(column) if column else None


def annotation_continuous_color_encoding(
    rows: list[dict[str, object]],
    spec: ResolvedPlotSpec,
    *,
    template: dict[str, object] | None = None,
) -> dict[str, object] | None:
    """Build strict continuous color encoding for annotation markers."""

    column = _annotation_hue_column(spec)
    if column is None:
        return None
    require_row_columns(rows, [column], context="plot annotation hue")
    numeric = np.asarray(
        [
            coerce_finite_float(row.get(column)) if coerce_finite_float(row.get(column)) is not None else np.nan
            for row in rows
        ],
        dtype=np.float64,
    )
    finite = numeric[np.isfinite(numeric)]
    if finite.size != numeric.size and getattr(spec.annotation, "missing_policy", "fail") == "fail":
        raise ContractViolationError(f"plot annotation hue {column!r} contains non-finite annotation hue value(s)")
    if finite.size == 0:
        return None
    if template is not None:
        return {
            "column": column,
            "values": numeric,
            "cmap": template["cmap"],
            "norm": template["norm"],
            "vmin": template.get("vmin"),
            "vmax": template.get("vmax"),
        }
    from matplotlib import colors as mcolors

    minimum = float(np.nanmin(finite))
    maximum = float(np.nanmax(finite))
    if minimum == maximum:
        pad = max(abs(minimum) * 0.05, 1.0e-6)
        return {
            "column": column,
            "values": numeric,
            "cmap": "cividis",
            "norm": mcolors.Normalize(vmin=minimum - pad, vmax=maximum + pad),
            "vmin": minimum - pad,
            "vmax": maximum + pad,
        }
    if minimum < 0.0 < maximum:
        max_abs = max(abs(minimum), abs(maximum), 1.0e-6)
        return {
            "column": column,
            "values": numeric,
            "cmap": "PuOr",
            "norm": mcolors.TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs),
            "vmin": None,
            "vmax": None,
        }
    return {
        "column": column,
        "values": numeric,
        "cmap": "cividis",
        "norm": mcolors.Normalize(vmin=minimum, vmax=maximum),
        "vmin": minimum,
        "vmax": maximum,
    }


def _annotation_colorbar_label(spec: ResolvedPlotSpec) -> str:
    annotation = getattr(spec, "annotation", None)
    column = _annotation_hue_column(spec)
    explicit_label = getattr(annotation, "colorbar_label", None) if annotation is not None else None
    return explicit_axis_label(explicit_label or humanize_display_text(column or "value"), width=24, max_lines=3)


def add_annotation_colorbar(
    figure: Any,
    axis: Any,
    *,
    spec: ResolvedPlotSpec,
    color_encoding: dict[str, object],
) -> None:
    """Attach a continuous colorbar for annotation-only hue encodings."""

    from matplotlib.cm import ScalarMappable

    label = _annotation_colorbar_label(spec)
    colorbar = figure.colorbar(
        ScalarMappable(norm=color_encoding["norm"], cmap=str(color_encoding["cmap"])),
        ax=axis,
        fraction=0.046,
        pad=0.04,
        label=label,
    )
    colorbar.ax.tick_params(labelsize=10, colors=TEXT_COLOR)
    colorbar.set_label(label, fontsize=11, color=TEXT_COLOR)


def _draw_continuous_annotation_markers(
    axis: Any,
    *,
    rows: list[dict[str, object]],
    resolved_x: str,
    resolved_y: str,
    color_encoding: dict[str, object],
    marker_size: float,
    marker: str | None,
    edgecolors: str,
) -> None:
    if marker is None or marker_size <= 0.0:
        return
    values = np.asarray(color_encoding["values"], dtype=np.float64)
    valid = np.isfinite(values)
    invalid = ~valid
    if np.any(valid):
        indices = np.flatnonzero(valid)
        axis.scatter(
            [float(rows[index][resolved_x]) for index in indices],
            [float(rows[index][resolved_y]) for index in indices],
            c=values[valid],
            cmap=str(color_encoding["cmap"]),
            norm=color_encoding["norm"],
            s=marker_size,
            marker=marker,
            edgecolors=edgecolors,
            linewidths=0.8,
            zorder=5,
        )
    if np.any(invalid):
        indices = np.flatnonzero(invalid)
        axis.scatter(
            [float(rows[index][resolved_x]) for index in indices],
            [float(rows[index][resolved_y]) for index in indices],
            c="#9AA5B1",
            s=marker_size,
            marker=marker,
            edgecolors=edgecolors,
            linewidths=0.8,
            zorder=5,
        )


def draw_annotation_callouts(
    axis: Any,
    *,
    rows: list[dict[str, object]],
    resolved_x: str,
    resolved_y: str,
    label_texts: list[str],
    marker_colors: list[str],
    font_size: float = 9.5,
    marker_size: float = 128.0,
    marker: str | None = "*",
    continuous_color: dict[str, object] | None = None,
) -> None:
    """Draw labeled callouts for a small reference overlay."""

    if not rows:
        return
    x_values = [float(row[resolved_x]) for row in rows]
    y_values = [float(row[resolved_y]) for row in rows]
    placed_boxes: list[tuple[float, float, float, float]] = []
    axes_box = axis.get_window_extent()
    display_x_mid = float((axes_box.x0 + axes_box.x1) / 2.0)
    display_y_mid = float((axes_box.y0 + axes_box.y1) / 2.0)
    if continuous_color is not None:
        _draw_continuous_annotation_markers(
            axis,
            rows=rows,
            resolved_x=resolved_x,
            resolved_y=resolved_y,
            color_encoding=continuous_color,
            marker_size=marker_size,
            marker=marker,
            edgecolors="white",
        )
    elif marker is not None and marker_size > 0.0:
        axis.scatter(
            x_values,
            y_values,
            c=marker_colors,
            s=marker_size,
            marker=marker,
            edgecolors="white",
            linewidths=0.8,
            zorder=5,
        )
    for row, label_text in sorted(
        zip(rows, label_texts, strict=True),
        key=lambda item: item[1].casefold(),
    ):
        point_x = float(row[resolved_x])
        point_y = float(row[resolved_y])
        display_x, display_y = axis.transData.transform((point_x, point_y))
        placement = choose_annotation_placement(
            display_x=display_x,
            display_y=display_y,
            label_text=label_text,
            axes_box=axes_box,
            placed_boxes=placed_boxes,
            x_mid=display_x_mid,
            y_mid=display_y_mid,
            font_size=font_size,
            left_padding_px=10.0,
            right_padding_px=10.0,
        )
        placed_boxes.append(placement.box)
        annotation = axis.annotate(
            label_text,
            xy=(point_x, point_y),
            xytext=(placement.offset_x, placement.offset_y),
            textcoords="offset pixels",
            fontsize=font_size,
            fontweight="semibold",
            ha=placement.ha,
            va=placement.va,
            color=TEXT_COLOR,
            bbox={
                "boxstyle": "round,pad=0.18",
                "fc": "white",
                "ec": "none",
                "alpha": ANNOTATION_LABEL_BOX_ALPHA,
            },
            arrowprops={"arrowstyle": "-", "color": SPINE_COLOR, "linewidth": 0.9},
            zorder=6,
        )
        annotation.set_clip_on(True)
        if annotation.arrow_patch is not None:
            annotation.arrow_patch.set_clip_on(True)


def draw_near_point_labels(
    axis: Any,
    *,
    rows: list[dict[str, object]],
    resolved_x: str,
    resolved_y: str,
    label_texts: list[str],
    font_size: float = 8.4,
) -> None:
    """Draw compact labels near selected points without callout arrows."""

    for row, label_text in sorted(zip(rows, label_texts, strict=True), key=lambda item: item[1].casefold()):
        axis.annotate(
            label_text,
            xy=(float(row[resolved_x]), float(row[resolved_y])),
            xytext=(7.0, 6.0),
            textcoords="offset pixels",
            fontsize=font_size,
            fontweight="semibold",
            ha="left",
            va="bottom",
            color=TEXT_COLOR,
            bbox={
                "boxstyle": "round,pad=0.16",
                "fc": "white",
                "ec": "none",
                "alpha": ANNOTATION_LABEL_BOX_ALPHA,
            },
            zorder=6,
        )


def draw_annotation_highlights(
    axis: Any,
    *,
    rows: list[dict[str, object]],
    resolved_x: str,
    resolved_y: str,
    marker_size: float = 96.0,
    marker: str | None = "*",
    continuous_color: dict[str, object] | None = None,
) -> None:
    """Draw unlabeled reference highlights for large overlays."""

    if not rows:
        return
    if continuous_color is not None:
        _draw_continuous_annotation_markers(
            axis,
            rows=rows,
            resolved_x=resolved_x,
            resolved_y=resolved_y,
            color_encoding=continuous_color,
            marker_size=marker_size,
            marker=marker,
            edgecolors="#111111",
        )
        return
    axis.scatter(
        [float(row[resolved_x]) for row in rows],
        [float(row[resolved_y]) for row in rows],
        s=marker_size,
        marker=marker,
        facecolors="#111111",
        edgecolors="#111111",
        linewidths=0.75,
        zorder=5,
    )


def draw_resolved_annotations(
    axis: Any,
    *,
    context: WorkspaceContext,
    spec: ResolvedPlotSpec,
    rows: list[dict[str, object]],
    resolved_x: str,
    resolved_y: str,
    resolved_label_column: str | None,
    color_map: dict[str, str],
    font_size: float = 9.5,
    marker_size: float = 128.0,
    marker: str | None = "*",
    annotation_color_encoding: dict[str, object] | None = None,
) -> None:
    """Draw an already-resolved reference or explicit-label annotation overlay."""

    if not rows or resolved_label_column is None:
        return
    annotation_color_encoding = annotation_color_encoding or annotation_continuous_color_encoding(rows, spec)
    annotation_marker = getattr(spec.annotation, "marker", marker) if spec.annotation is not None else marker
    label_mode = (
        "label_and_highlight"
        if spec.annotation is None
        else context.config.reference_sets[spec.annotation.reference_set].label_mode
    )
    if label_mode == "highlight_only" or len(rows) > 5:
        draw_annotation_highlights(
            axis,
            rows=rows,
            resolved_x=resolved_x,
            resolved_y=resolved_y,
            marker=annotation_marker,
            continuous_color=annotation_color_encoding,
        )
        return
    if label_mode != "label_and_highlight":
        return
    if getattr(spec, "plot_id", "") == "candidate_decision_frontier" and spec.annotation is None:
        draw_near_point_labels(
            axis,
            rows=rows,
            resolved_x=resolved_x,
            resolved_y=resolved_y,
            label_texts=[
                annotation_label_text(
                    context,
                    spec=spec,
                    row=row,
                    resolved_label_column=resolved_label_column,
                )
                for row in rows
            ],
            font_size=font_size,
        )
        return
    highlight_colors = (
        ["#111111"] * len(rows)
        if spec.annotation is not None
        else color_series(
            rows,
            spec.color_column,
            color_map=color_map if color_map else None,
        )[0]
    )
    draw_annotation_callouts(
        axis,
        rows=rows,
        resolved_x=resolved_x,
        resolved_y=resolved_y,
        label_texts=[
            annotation_label_text(
                context,
                spec=spec,
                row=row,
                resolved_label_column=resolved_label_column,
            )
            for row in rows
        ],
        marker_colors=highlight_colors,
        font_size=font_size,
        marker_size=marker_size,
        marker=annotation_marker,
        continuous_color=annotation_color_encoding,
    )
