"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/src/render/junction_annealed_fragments.py

Registered renderer for nucleotide-level Junction fragment annealing.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..config import Style
from ..core import Record
from .junction_review.annealed_panel import draw_annealed_panel, fragment_selection, validate_fragment_rows
from .junction_review.foundation import review_from_record, validate_figure_size
from .palette import Palette

_RENDERER = "junction_annealed_fragments"
_FIGURE_WIDTH = 15.2


def _figure_height(fragment_count: int) -> float:
    return 1.45 + 0.88 * fragment_count


@dataclass(frozen=True)
class JunctionAnnealedFragmentsRenderer:
    """Render explicitly selected, sequence-derived fragment-annealing maps."""

    def preflight(
        self,
        record: Record,
        style: Style,
        palette: Palette,
        options: Mapping[str, object] | None = None,
    ) -> None:
        _ = palette
        review = review_from_record(record)
        indices = fragment_selection(review, options, renderer=_RENDERER)
        validate_fragment_rows(review, indices, renderer=_RENDERER)
        validate_figure_size(
            style,
            renderer=_RENDERER,
            width=_FIGURE_WIDTH,
            height=_figure_height(len(indices)),
        )

    def render(
        self,
        record: Record,
        style: Style,
        palette: Palette,
        options: Mapping[str, object] | None = None,
    ):
        _ = palette
        review = review_from_record(record)
        indices = fragment_selection(review, options, renderer=_RENDERER)
        validate_fragment_rows(review, indices, renderer=_RENDERER)
        height = _figure_height(len(indices))
        size = validate_figure_size(style, renderer=_RENDERER, width=_FIGURE_WIDTH, height=height)
        figure, axis = plt.subplots(figsize=size, dpi=style.dpi)
        draw_annealed_panel(axis, review, indices, height=height)
        figure.subplots_adjust(left=0.008, right=0.995, top=0.995, bottom=0.01)
        return figure


__all__ = ["JunctionAnnealedFragmentsRenderer"]
