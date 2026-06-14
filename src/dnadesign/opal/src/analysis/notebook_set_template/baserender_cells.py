from __future__ import annotations

from .baserender_record_cells import render_baserender_record_cells
from .baserender_scope_cells import render_baserender_scope_cells


def render_baserender_cells() -> str:
    """Render campaign-set selected-sequence BaseRender cells."""

    return "\n\n".join((render_baserender_scope_cells(), render_baserender_record_cells()))


__all__ = ["render_baserender_cells"]
