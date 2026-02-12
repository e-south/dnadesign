"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/baserender/src/render/effects/motif_logo.py

Motif-logo effect drawer overlaying probability stacks on top of kmer feature boxes.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import math

from matplotlib.patches import Rectangle

from ...core import Effect, Record, RenderingError
from ..layout import LayoutContext
from ..palette import Palette

_BASE_COLORS = {
    "A": "#1f77b4",
    "C": "#2ca02c",
    "G": "#ff7f0e",
    "T": "#d62728",
}


def _info_content(prob: float, n_bases: int = 4) -> float:
    if prob <= 0:
        return 0.0
    return max(0.0, math.log2(n_bases) + prob * math.log2(prob))


def _normalize_row(row: list[float]) -> list[float]:
    s = sum(row)
    if s <= 0:
        raise RenderingError("motif_logo matrix rows must have positive sum")
    return [float(x) / s for x in row]


def draw_motif_logo(
    ax,
    effect: Effect,
    record: Record,
    layout: LayoutContext,
    style,
    palette: Palette,
    feature_boxes: dict[str, tuple[float, float, float, float]],
) -> None:
    feature_id = str(effect.target.get("feature_id", ""))
    if feature_id == "":
        raise RenderingError("motif_logo target.feature_id is required")

    box = feature_boxes.get(feature_id)
    if box is None:
        raise RenderingError(f"motif_logo target feature '{feature_id}' not found")
    x1, y1, x2, y2 = box

    matrix_raw = effect.params.get("matrix")
    if not isinstance(matrix_raw, list) or not matrix_raw:
        raise RenderingError("motif_logo params.matrix must be a non-empty list")

    feature = next((f for f in record.features if (f.id or "") == feature_id), None)
    if feature is None:
        raise RenderingError(f"motif_logo target feature '{feature_id}' not found in record")
    if feature.kind != "kmer":
        raise RenderingError("motif_logo target feature must be kind='kmer'")
    if feature.label is None:
        raise RenderingError("motif_logo target kmer feature must have label")

    label = feature.label.upper()
    observed = list(label)

    width = x2 - x1
    height = y2 - y1
    n = len(matrix_raw)
    if n == 0:
        return
    col_w = width / n

    for i, row_raw in enumerate(matrix_raw):
        if not isinstance(row_raw, (list, tuple)) or len(row_raw) < 4:
            raise RenderingError("motif_logo matrix rows must contain at least 4 probabilities [A,C,G,T]")
        row = _normalize_row([float(row_raw[0]), float(row_raw[1]), float(row_raw[2]), float(row_raw[3])])
        pairs = [("A", row[0]), ("C", row[1]), ("G", row[2]), ("T", row[3])]
        # Probability stack with information-content scaling.
        stacks = []
        for base, prob in pairs:
            info = _info_content(prob)
            stacks.append((base, prob * max(0.1, info)))
        stacks.sort(key=lambda item: item[1])

        x = x1 + i * col_w
        y = y1
        total = sum(v for _, v in stacks) or 1.0
        for base, raw_h in stacks:
            frac = raw_h / total
            h = frac * height
            alpha = 0.35
            if i < len(observed) and observed[i] == base:
                alpha = 0.9
            ax.add_patch(
                Rectangle(
                    (x, y),
                    col_w,
                    h,
                    facecolor=_BASE_COLORS[base],
                    edgecolor="none",
                    alpha=alpha,
                    zorder=4.5,
                    clip_on=False,
                )
            )
            y += h
