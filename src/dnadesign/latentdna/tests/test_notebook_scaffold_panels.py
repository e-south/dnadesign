"""
Contract tests for generated marimo browser scaffold cells.
"""

from __future__ import annotations

from dnadesign.latentdna.src.notebooks.scaffold_panels import (
    render_geometry_frames_cell,
    render_geometry_panel_cell,
    render_plot_review_cell,
)


def test_browser_surface_panels_do_not_eagerly_render_inactive_surface() -> None:
    plot_review_cell = render_plot_review_cell()
    geometry_frames_cell = render_geometry_frames_cell()
    geometry_panel_cell = render_geometry_panel_cell()

    assert "surface_selector" in plot_review_cell
    assert 'str(surface_selector.value) == "plots"' in plot_review_cell
    assert 'str(surface_selector.value) != "plots"' in plot_review_cell
    assert "lazy=True" not in plot_review_cell

    assert "surface_selector" in geometry_frames_cell
    assert 'str(surface_selector.value) == "geometry_browser"' in geometry_frames_cell

    assert "surface_selector" in geometry_panel_cell
    assert 'str(surface_selector.value) != "geometry_browser"' in geometry_panel_cell
    assert "lazy=True" not in geometry_panel_cell
