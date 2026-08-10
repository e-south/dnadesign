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
from collections.abc import Sequence
from dataclasses import dataclass

MAX_EXPECTED_PCR_DUPLEX_BASES = 1_024
MAX_ASSEMBLY_FRAGMENTS = 64
PRODUCT_ROW_STEP = 0.66
ORDER_GROUP_GAP_BASES = 14
MOLECULAR_PLOT_LEFT = 0.06
MOLECULAR_PLOT_RIGHT = 0.94
MIN_ASSEMBLY_WIDTH = 15.2
MAX_ASSEMBLY_WIDTH = 64.0
TARGET_BASE_WIDTH_INCHES = 0.085
PRODUCT_BASE_WIDTH_INCHES = 0.13
STAGE_TITLE_TO_MOLECULE = 0.72
MOLECULE_TO_TRANSITION = 0.38
TRANSITION_TO_STAGE_TITLE = 0.38


@dataclass(frozen=True, slots=True)
class ProductWindow:
    index: int
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class AssemblyLayout:
    """Canvas and coordinates for an exact, strand-resolved process figure."""

    width: float
    height: float
    order_left: float
    order_base_step: float
    order_fontsize: float
    target_left: float
    target_base_step: float
    target_fontsize: float
    barcode_base_step_y: float
    product_left: float
    product_right: float
    product_bases_per_row: int
    title_y: float
    input_title_y: float
    input_first_y: float
    fragmentation_transition_y: float
    orders_title_y: float
    orders_first_y: float
    annealing_transition_y: float
    preligation_title_y: float
    preligation_first_y: float
    recovery_transition_y: float
    product_title_y: float
    product_first_y: float


def product_windows(length: int, *, bases_per_row: int) -> tuple[ProductWindow, ...]:
    """Split an exact duplex only when its review-scale spacing requires it."""

    if min(length, bases_per_row) < 1:
        raise ValueError("expected PCR product length and row capacity must be positive")
    return tuple(
        ProductWindow(index=index, start=start, end=min(start + bases_per_row, length))
        for index, start in enumerate(range(0, length, bases_per_row))
    )


def assembly_layout(
    *,
    fragment_widths: Sequence[int],
    target_length: int,
    barcode_length: int,
    product_length: int,
) -> AssemblyLayout:
    """Lay out two continuous molecular stages and one wrapped product."""

    if not fragment_widths or min(*fragment_widths, target_length, barcode_length, product_length) < 1:
        raise ValueError("assembly layout dimensions must be positive")
    fragment_count = len(fragment_widths)
    order_span = sum(fragment_widths) + ORDER_GROUP_GAP_BASES * (fragment_count - 1)
    plot_fraction = MOLECULAR_PLOT_RIGHT - MOLECULAR_PLOT_LEFT
    molecular_width = order_span * TARGET_BASE_WIDTH_INCHES / plot_fraction
    single_product_width = product_length * PRODUCT_BASE_WIDTH_INCHES / plot_fraction
    width = min(MAX_ASSEMBLY_WIDTH, max(MIN_ASSEMBLY_WIDTH, molecular_width, single_product_width))
    order_base_step = min(PRODUCT_BASE_WIDTH_INCHES / width, plot_fraction / order_span)
    target_base_step = min(PRODUCT_BASE_WIDTH_INCHES / width, plot_fraction / target_length)
    order_fontsize = min(11.0, max(3.6, width * order_base_step * 72.0 * 1.18))
    target_fontsize = min(11.0, max(3.6, width * target_base_step * 72.0 * 1.18))
    barcode_base_step_y = width * target_base_step

    product_bases_per_row = min(
        product_length,
        max(1, math.floor(width * plot_fraction / PRODUCT_BASE_WIDTH_INCHES + 1e-9)),
    )
    product_fraction = product_bases_per_row * PRODUCT_BASE_WIDTH_INCHES / width
    product_left = 0.5 - product_fraction / 2
    product_right = 0.5 + product_fraction / 2
    product_rows = math.ceil(product_length / product_bases_per_row)
    stem_height = 0.12 + barcode_length * barcode_base_step_y

    title_y = -0.08
    input_title_y = -0.60
    input_first_y = input_title_y - STAGE_TITLE_TO_MOLECULE
    fragmentation_transition_y = input_first_y - MOLECULE_TO_TRANSITION
    orders_title_y = fragmentation_transition_y - TRANSITION_TO_STAGE_TITLE
    orders_first_y = orders_title_y - STAGE_TITLE_TO_MOLECULE
    orders_bottom = orders_first_y - 0.22
    annealing_transition_y = orders_bottom - MOLECULE_TO_TRANSITION
    preligation_title_y = annealing_transition_y - TRANSITION_TO_STAGE_TITLE
    preligation_first_y = preligation_title_y - stem_height - STAGE_TITLE_TO_MOLECULE
    preligation_bottom = preligation_first_y - 0.22
    recovery_transition_y = preligation_bottom - MOLECULE_TO_TRANSITION
    product_title_y = recovery_transition_y - TRANSITION_TO_STAGE_TITLE
    product_first_y = product_title_y - STAGE_TITLE_TO_MOLECULE
    product_bottom = product_first_y - (product_rows - 1) * PRODUCT_ROW_STEP - 0.22
    height = -product_bottom + 0.18

    def absolute(offset: float) -> float:
        return height + offset

    return AssemblyLayout(
        width=width,
        height=height,
        order_left=0.5 - order_span * order_base_step / 2,
        order_base_step=order_base_step,
        order_fontsize=order_fontsize,
        target_left=0.5 - target_length * target_base_step / 2,
        target_base_step=target_base_step,
        target_fontsize=target_fontsize,
        barcode_base_step_y=barcode_base_step_y,
        product_left=product_left,
        product_right=product_right,
        product_bases_per_row=product_bases_per_row,
        title_y=absolute(title_y),
        input_title_y=absolute(input_title_y),
        input_first_y=absolute(input_first_y),
        fragmentation_transition_y=absolute(fragmentation_transition_y),
        orders_title_y=absolute(orders_title_y),
        orders_first_y=absolute(orders_first_y),
        annealing_transition_y=absolute(annealing_transition_y),
        preligation_title_y=absolute(preligation_title_y),
        preligation_first_y=absolute(preligation_first_y),
        recovery_transition_y=absolute(recovery_transition_y),
        product_title_y=absolute(product_title_y),
        product_first_y=absolute(product_first_y),
    )
