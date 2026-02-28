"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/src/cli/notebook_cells_template.py

Marimo notebook cell block template entrypoint for DenseGen notebook scaffolding.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from .notebook_cells_template_base import NOTEBOOK_TEMPLATE_CELLS_BASE
from .notebook_cells_template_gallery import NOTEBOOK_TEMPLATE_CELLS_GALLERY


def notebook_template_cells(*, baserender_export_cell_template, records_export_cell_template) -> str:
    return (
        NOTEBOOK_TEMPLATE_CELLS_BASE
        + NOTEBOOK_TEMPLATE_CELLS_GALLERY
        + baserender_export_cell_template()
        + records_export_cell_template()
    )
