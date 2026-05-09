"""Matplotlib drawing helpers for static annotation overlays."""

from __future__ import annotations

from typing import Any

from ..annotation_layout import choose_annotation_placement
from ..contracts.plot import ResolvedPlotSpec
from ..visual_style import ANNOTATION_LABEL_BOX_ALPHA, SPINE_COLOR, TEXT_COLOR
from ..workspaces.loader import WorkspaceContext
from .annotations import annotation_label_text
from .renderers.scatter import color_series


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
    if marker is not None and marker_size > 0.0:
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
) -> None:
    """Draw unlabeled reference highlights for large overlays."""

    if not rows:
        return
    axis.scatter(
        [float(row[resolved_x]) for row in rows],
        [float(row[resolved_y]) for row in rows],
        s=marker_size,
        marker="*",
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
) -> None:
    """Draw an already-resolved reference or explicit-label annotation overlay."""

    if not rows or resolved_label_column is None:
        return
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
        marker=marker,
    )
