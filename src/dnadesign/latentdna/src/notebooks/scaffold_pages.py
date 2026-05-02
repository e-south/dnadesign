"""
Page composition cell templates for generated latentdna marimo notebooks.
"""

from __future__ import annotations

from .scaffold_panels import (
    render_geometry_frames_cell,
    render_geometry_hue_selector_cell,
    render_geometry_panel_cell,
    render_geometry_resolution_cell,
    render_page_display_cell,
    render_page_tabs_cell,
    render_plot_review_cell,
    render_scope_note_cell,
)


def render_page_cells() -> tuple[str, ...]:
    return (
        render_scope_note_cell(),
        render_plot_review_cell(),
        render_geometry_resolution_cell(),
        render_geometry_frames_cell(),
        render_geometry_hue_selector_cell(),
        render_geometry_panel_cell(),
        render_page_tabs_cell(),
        render_page_display_cell(),
    )
