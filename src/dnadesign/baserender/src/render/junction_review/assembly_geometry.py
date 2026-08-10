"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/src/render/junction_review/assembly_geometry.py

Bounded layout geometry for Junction assembly-process review plots.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import math
from dataclasses import dataclass

PRODUCT_BASES_PER_WINDOW = 100
MAX_RECOVERED_DUPLEX_BASES = 1_024
MAX_ASSEMBLY_FRAGMENTS = 64
PRODUCT_ROW_STEP = 0.66


@dataclass(frozen=True, slots=True)
class ProductWindow:
    index: int
    start: int
    end: int


def product_windows(length: int) -> tuple[ProductWindow, ...]:
    """Split an exact duplex into legible, coordinate-preserving rows."""

    if length < 1:
        raise ValueError("recovered product length must be positive")
    return tuple(
        ProductWindow(index=index, start=start, end=min(start + PRODUCT_BASES_PER_WINDOW, length))
        for index, start in enumerate(range(0, length, PRODUCT_BASES_PER_WINDOW))
    )


def assembly_figure_height(product_length: int) -> float:
    """Return a bounded height that keeps each product window readable."""

    rows = math.ceil(product_length / PRODUCT_BASES_PER_WINDOW)
    return 5.33 + max(0, rows - 1) * PRODUCT_ROW_STEP


__all__ = [
    "MAX_ASSEMBLY_FRAGMENTS",
    "MAX_RECOVERED_DUPLEX_BASES",
    "PRODUCT_BASES_PER_WINDOW",
    "PRODUCT_ROW_STEP",
    "ProductWindow",
    "assembly_figure_height",
    "product_windows",
]
