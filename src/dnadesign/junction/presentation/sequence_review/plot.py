"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/junction/presentation/sequence_review/plot.py

Deterministic Junction sequence-comparison plot.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import re
from io import BytesIO
from typing import Sequence

import matplotlib
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_svg import FigureCanvasSVG
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure

from .contract import JunctionSequenceDissimilarityV1
from .metrics import DissimilaritySelection, pairwise_matrices, resolve_selection

_BACKGROUND = "#FCFCFA"
_INK = "#24313D"
_MUTED = "#68717D"
_TOEHOLD = "#E4B86B"
_TOEHOLD_DARK = "#8A5E18"
_BARCODE = "#75B9B4"
_BARCODE_DARK = "#286F6B"
_COMBINED = "#A99BC3"
_COMBINED_DARK = "#5E4B7A"
_LEFT_MARGIN_MIN = 0.065
_LEFT_MARGIN_MAX = 0.24
_LABEL_PADDING_POINTS = 4.0
_DISPLAY_LABEL_MAX = 28
_DISPLAY_TARGET_MAX = 14
_DISPLAY_LOCAL_MAX = 13


def _safe_identifier(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.:/-]+", "-", value).strip("-") or "junction"


def _compact_display_identifier(value: str, *, maximum: int) -> str:
    safe = _safe_identifier(value)
    if len(safe) <= maximum:
        return safe
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:8]
    return f"{safe[: maximum - len(digest) - 1]}~{digest}"


def _unique_display_tokens(values: Sequence[str], *, maximum: int) -> dict[str, str]:
    distinct = tuple(sorted(set(values)))
    tokens = {value: _compact_display_identifier(value, maximum=maximum) for value in distinct}
    if len(set(tokens.values())) == len(tokens):
        return tokens
    ordinal_width = max(2, len(str(len(distinct))))
    return {
        value: (f"{_safe_identifier(value)[: maximum - ordinal_width - 1]}~{ordinal:0{ordinal_width}d}")
        for ordinal, value in enumerate(distinct, start=1)
    }


def _labels(
    review: JunctionSequenceDissimilarityV1,
    selection: DissimilaritySelection,
) -> tuple[str, ...]:
    selected = tuple(review.junctions[index] for index in selection.indices)
    local = tuple(junction.junction_id.rsplit(":", 1)[-1] for junction in selected)
    if len(local) == len(set(local)):
        tokens = _unique_display_tokens(local, maximum=_DISPLAY_LABEL_MAX)
        return tuple(tokens[label] for label in local)
    target_tokens = _unique_display_tokens(
        tuple(junction.target_id for junction in selected),
        maximum=_DISPLAY_TARGET_MAX,
    )
    labels = tuple(
        f"{target_tokens[junction.target_id]}/"
        f"{_compact_display_identifier(junction.junction_id.rsplit(':', 1)[-1], maximum=_DISPLAY_LOCAL_MAX)}"
        for junction in selected
    )
    if len(labels) == len(set(labels)):
        return labels
    sources = tuple(f"{junction.target_id}/{junction.junction_id.rsplit(':', 1)[-1]}" for junction in selected)
    tokens = _unique_display_tokens(sources, maximum=_DISPLAY_LABEL_MAX)
    return tuple(tokens[source] for source in sources)


def _color_map(name: str, light: str, dark: str) -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(name, (_BACKGROUND, light, dark), N=9)


def _limits(matrix: np.ndarray) -> tuple[float, float]:
    if matrix.shape[0] < 2:
        return 0.0, 1.0
    values = matrix[np.triu_indices(matrix.shape[0], k=1)]
    minimum = float(values.min())
    maximum = float(values.max())
    if minimum == maximum:
        return 0.0, max(1.0, maximum)
    return minimum, maximum


def _matrix_value_text_color(image, value: float) -> str:
    """Choose legible text against the rendered heatmap cell."""

    red, green, blue, _alpha = image.cmap(image.norm(value))
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    return "#FFFFFF" if luminance < 0.52 else _INK


def _draw_matrix(
    axis,
    matrix: np.ndarray,
    *,
    labels: tuple[str, ...],
    title: str,
    cmap,
    gid: str,
    show_y_labels: bool,
) -> None:
    axis.set_gid(gid)
    masked = np.ma.array(matrix, mask=np.eye(matrix.shape[0], dtype=bool))
    boundaries = np.arange(matrix.shape[0] + 1, dtype=np.float64) - 0.5
    image = axis.pcolormesh(
        boundaries,
        boundaries,
        masked,
        cmap=cmap,
        vmin=_limits(matrix)[0],
        vmax=_limits(matrix)[1],
        shading="flat",
    )
    image.set_gid(f"{gid}:matrix")
    axis.set_xlim(-0.5, matrix.shape[0] - 0.5)
    axis.set_ylim(matrix.shape[0] - 0.5, -0.5)
    axis.set_aspect("equal", adjustable="box")
    axis.set_title(title, fontsize=13.0, color=_INK, pad=12, fontweight="normal")
    axis.set_xticks(range(len(labels)), labels=labels, rotation=55, ha="right", rotation_mode="anchor")
    axis.set_yticks(
        range(len(labels)),
        labels=labels if show_y_labels else ("",) * len(labels),
    )
    tick_size = 8.5 if len(labels) <= 12 else 7.0
    axis.tick_params(axis="both", which="both", length=0, labelsize=tick_size, colors=_INK)
    for spine in axis.spines.values():
        spine.set_visible(False)
    if len(labels) <= 12:
        integer_values = np.allclose(matrix, np.rint(matrix))
        value_fontsize = min(9.0, max(6.5, 88.0 / len(labels)))
        for row in range(len(labels)):
            for column in range(len(labels)):
                value = "—" if row == column else f"{matrix[row, column]:.0f}"
                if row != column and not integer_values:
                    value = f"{matrix[row, column]:.2f}"
                artist = axis.text(
                    column,
                    row,
                    value,
                    ha="center",
                    va="center",
                    fontsize=value_fontsize,
                    color=_MUTED if row == column else _matrix_value_text_color(image, matrix[row, column]),
                )
                artist.set_gid(f"{gid}:value:{row}:{column}")
    colorbar = axis.figure.colorbar(image, ax=axis, fraction=0.046, pad=0.035)
    colorbar.solids.set_rasterized(False)
    colorbar.outline.set_visible(False)
    colorbar.ax.tick_params(labelsize=8, colors=_MUTED, length=0)


def _left_margin_for_visible_labels(figure: Figure, axis) -> float:
    """Reserve bounded space for the first matrix's rendered row labels."""

    canvas = FigureCanvasAgg(figure)
    canvas.draw()
    renderer = canvas.get_renderer()
    labels = [label for label in axis.get_yticklabels() if label.get_text()]
    leftmost = min(label.get_window_extent(renderer).x0 for label in labels)
    padding = _LABEL_PADDING_POINTS * figure.dpi / 72.0
    if leftmost >= padding:
        return _LEFT_MARGIN_MIN
    required = _LEFT_MARGIN_MIN + (padding - leftmost) / figure.bbox.width
    return min(_LEFT_MARGIN_MAX, required)


def plot_sequence_dissimilarity(
    review: JunctionSequenceDissimilarityV1,
    *,
    junction_ids: Sequence[str] | None = None,
):
    """Draw the string metrics used by Junction's current search policy."""

    selection = resolve_selection(review, junction_ids)
    matrices = pairwise_matrices(review, selection)
    labels = _labels(review, selection)
    figure = Figure(figsize=(17.4, 6.5), dpi=150)
    axes = figure.subplots(1, 3)
    figure.patch.set_facecolor(_BACKGROUND)
    figure.suptitle(
        (
            f"Pairwise string metrics compare {selection.selected_count} of {selection.total_count} junctions"
            if selection.selected_count != selection.total_count
            else f"Pairwise string metrics compare {selection.selected_count} junctions"
        ),
        x=0.5,
        y=0.985,
        ha="center",
        va="top",
        fontsize=19.0,
        fontweight="semibold",
        color=_INK,
    )
    panels = (
        (
            matrices.toehold,
            "Toeholds\nWeighted edit distance · larger is more different",
            _color_map("junction-toeholds", _TOEHOLD, _TOEHOLD_DARK),
            "toeholds",
        ),
        (
            matrices.barcode,
            "Barcodes\nEdit distance · larger is more different",
            _color_map("junction-barcodes", _BARCODE, _BARCODE_DARK),
            "barcodes",
        ),
        (
            matrices.combined,
            "Toehold + barcode\nLongest shared span · smaller is better separated",
            _color_map("junction-combined", _COMBINED, _COMBINED_DARK),
            "combined",
        ),
    )
    for index, (axis, (matrix, title, cmap, suffix)) in enumerate(zip(axes, panels, strict=True)):
        axis.set_facecolor(_BACKGROUND)
        _draw_matrix(
            axis,
            matrix,
            labels=labels,
            title=title,
            cmap=cmap,
            gid=f"junction-sequence-dissimilarity:{suffix}",
            show_y_labels=index == 0,
        )
    figure.subplots_adjust(
        left=_LEFT_MARGIN_MIN,
        right=0.965,
        top=0.82,
        bottom=0.18,
        wspace=0.34,
    )
    figure.subplots_adjust(left=_left_margin_for_visible_labels(figure, axes[0]))
    return figure


def render_sequence_dissimilarity_svg(
    review: JunctionSequenceDissimilarityV1,
    *,
    junction_ids: Sequence[str] | None = None,
) -> bytes:
    """Render canonical SVG bytes for one bounded sequence comparison."""

    deterministic_rc = dict(matplotlib.rcParamsDefault)
    deterministic_rc["svg.hashsalt"] = "dnadesign.junction"
    with matplotlib.rc_context(deterministic_rc):
        figure = plot_sequence_dissimilarity(review, junction_ids=junction_ids)
        buffer = BytesIO()
        FigureCanvasSVG(figure).print_svg(buffer, metadata={"Date": None})
    lines = (line.rstrip() for line in buffer.getvalue().splitlines())
    return b"\n".join(lines) + b"\n"


__all__ = ["plot_sequence_dissimilarity", "render_sequence_dissimilarity_svg"]
