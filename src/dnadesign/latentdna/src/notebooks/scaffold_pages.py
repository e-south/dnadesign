"""
Page composition cell templates for generated latentdna marimo notebooks.
"""

from __future__ import annotations

from .scaffold_panels import (
    render_compare_cell,
    render_context_audit_cell,
    render_deliverable_cell,
    render_geometry_cell,
    render_inventory_cell,
    render_overview_cell,
    render_page_display_cell,
    render_page_tabs_cell,
)


def render_page_cells() -> tuple[str, ...]:
    return (
        render_context_audit_cell(),
        render_overview_cell(),
        render_geometry_cell(),
        render_compare_cell(),
        render_deliverable_cell(),
        render_inventory_cell(),
        render_page_tabs_cell(),
        render_page_display_cell(),
    )
