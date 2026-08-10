"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/junction/presentation/sequence_review/plot.py

Deterministic Junction sequence-comparison plot.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import re
from io import BytesIO
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

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


def _safe_identifier(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.:/-]+", "-", value).strip("-") or "junction"


def _labels(
    review: JunctionSequenceDissimilarityV1,
    selection: DissimilaritySelection,
) -> tuple[str, ...]:
    selected = tuple(review.junctions[index] for index in selection.indices)
    local = tuple(junction.junction_id.rsplit(":", 1)[-1] for junction in selected)
    if len(local) == len(set(local)):
        return tuple(_safe_identifier(label) for label in local)
    return tuple(
        _safe_identifier(f"{junction.target_id}/{junction.junction_id.rsplit(':', 1)[-1]}") for junction in selected
    )


def _color_map(name: str, light: str, dark: str) -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(name, (_BACKGROUND, light, dark))


def _limits(matrix: np.ndarray) -> tuple[float, float]:
    if matrix.shape[0] < 2:
        return 0.0, 1.0
    values = matrix[np.triu_indices(matrix.shape[0], k=1)]
    minimum = float(values.min())
    maximum = float(values.max())
    if minimum == maximum:
        return 0.0, max(1.0, maximum)
    return minimum, maximum


def _draw_matrix(axis, matrix: np.ndarray, *, labels: tuple[str, ...], title: str, cmap, gid: str) -> None:
    axis.set_gid(gid)
    masked = np.ma.array(matrix, mask=np.eye(matrix.shape[0], dtype=bool))
    image = axis.imshow(masked, cmap=cmap, vmin=_limits(matrix)[0], vmax=_limits(matrix)[1])
    axis.set_title(title, fontsize=13.0, color=_INK, pad=12, fontweight="normal")
    axis.set_xticks(range(len(labels)), labels=labels, rotation=55, ha="right", rotation_mode="anchor")
    axis.set_yticks(range(len(labels)), labels=labels)
    tick_size = 8.5 if len(labels) <= 12 else 7.0
    axis.tick_params(axis="both", which="both", length=0, labelsize=tick_size, colors=_INK)
    for spine in axis.spines.values():
        spine.set_visible(False)
    if len(labels) <= 12:
        integer_values = np.allclose(matrix, np.rint(matrix))
        for row in range(len(labels)):
            for column in range(len(labels)):
                value = "—" if row == column else f"{matrix[row, column]:.0f}"
                if row != column and not integer_values:
                    value = f"{matrix[row, column]:.2f}"
                axis.text(column, row, value, ha="center", va="center", fontsize=8.5, color=_INK)
    colorbar = axis.figure.colorbar(image, ax=axis, fraction=0.046, pad=0.035)
    colorbar.outline.set_visible(False)
    colorbar.ax.tick_params(labelsize=8, colors=_MUTED, length=0)


def plot_sequence_dissimilarity(
    review: JunctionSequenceDissimilarityV1,
    *,
    junction_ids: Sequence[str] | None = None,
):
    """Draw the string metrics used by Junction's current search policy."""

    selection = resolve_selection(review, junction_ids)
    matrices = pairwise_matrices(review, selection)
    labels = _labels(review, selection)
    figure, axes = plt.subplots(1, 3, figsize=(15.2, 5.7), dpi=150)
    figure.patch.set_facecolor(_BACKGROUND)
    figure.suptitle(
        "Pairwise string metrics show how the selected junctions differ",
        x=0.5,
        y=0.985,
        ha="center",
        va="top",
        fontsize=19.0,
        fontweight="semibold",
        color=_INK,
    )
    scope = (
        f"{selection.selected_count} of {selection.total_count} junctions"
        if selection.selected_count != selection.total_count
        else f"{selection.selected_count} junctions"
    )
    figure.text(
        0.5,
        0.925,
        f"{_safe_identifier(review.assembly_group_id)} · {scope} · thermodynamic screening not run",
        ha="center",
        va="top",
        fontsize=11.5,
        color=_MUTED,
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
    for axis, (matrix, title, cmap, suffix) in zip(axes, panels, strict=True):
        axis.set_facecolor(_BACKGROUND)
        _draw_matrix(
            axis,
            matrix,
            labels=labels,
            title=title,
            cmap=cmap,
            gid=f"junction-sequence-dissimilarity:{suffix}",
        )
    figure.subplots_adjust(left=0.065, right=0.965, top=0.81, bottom=0.19, wspace=0.42)
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
        try:
            buffer = BytesIO()
            figure.savefig(buffer, format="svg", metadata={"Date": None})
        finally:
            plt.close(figure)
    lines = (line.rstrip() for line in buffer.getvalue().splitlines())
    return b"\n".join(lines) + b"\n"


__all__ = ["plot_sequence_dissimilarity", "render_sequence_dissimilarity_svg"]
