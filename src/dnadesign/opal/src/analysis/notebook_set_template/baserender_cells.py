"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/analysis/notebook_set_template/baserender_cells.py

Notebook-set template builders for BaseRender cells OPAL analysis notebook set template.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from .baserender_record_cells import render_baserender_record_cells
from .baserender_scope_cells import render_baserender_scope_cells


def render_baserender_cells() -> str:
    """Render campaign-set selected-sequence BaseRender cells."""

    return "\n\n".join((render_baserender_scope_cells(), render_baserender_record_cells()))


__all__ = ["render_baserender_cells"]
