"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/analysis/notebook_set_template/details_cells.py

Notebook-set template builders for details cells OPAL analysis notebook set template.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from .detail_panel_cells import render_detail_panel_cells
from .layout_cells import render_layout_cells
from .reader_evidence_cells import render_reader_evidence_cells


def render_details_cells() -> str:
    """Render evidence, secondary detail panels, layout, and app entrypoint cells."""

    return "\n\n".join(
        (
            render_detail_panel_cells(),
            render_reader_evidence_cells(),
            render_layout_cells(),
        )
    )


__all__ = ["render_details_cells"]
