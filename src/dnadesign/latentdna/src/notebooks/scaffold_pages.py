"""
Page composition cell templates for generated latentdna marimo notebooks.
"""

from __future__ import annotations

from .scaffold_geometry_panels import (
    render_geometry_frames_cell,
    render_geometry_hue_selector_cell,
    render_geometry_panel_cell,
    render_geometry_resolution_cell,
)
from .scaffold_panels import (
    render_browser_surface_cell,
    render_page_display_cell,
    render_scope_note_cell,
)
from .scaffold_plot_review import render_plot_review_cell


def render_page_cells() -> tuple[str, ...]:
    return (
        render_scope_note_cell(),
        render_plot_review_cell(),
        render_geometry_resolution_cell(),
        render_geometry_frames_cell(),
        render_geometry_hue_selector_cell(),
        render_geometry_panel_cell(),
        render_browser_surface_cell(),
        render_page_display_cell(),
    )
