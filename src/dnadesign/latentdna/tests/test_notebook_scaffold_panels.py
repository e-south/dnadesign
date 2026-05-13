"""
Contract tests for generated marimo browser scaffold cells.
"""

from __future__ import annotations

from dnadesign.latentdna.src.notebooks.scaffold_panels import (
    render_geometry_frames_cell,
    render_geometry_hue_selector_cell,
    render_geometry_panel_cell,
    render_plot_review_cell,
)
from dnadesign.latentdna.src.notebooks.scaffold_selectors import render_selector_cells


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


def test_projection_selector_is_only_rendered_when_it_controls_single_view_layout() -> None:
    geometry_panel_cell = render_geometry_panel_cell()

    assert (
        "_control_widgets = [layout_selector, model_selector, family_selector, context_selector]" in geometry_panel_cell
    )
    assert "_control_widgets.extend([geometry_selector, projection_selector])" in geometry_panel_cell
    assert "_control_widgets = [layout_selector]" in geometry_panel_cell
    assert "_control_widgets.append(projection_selector)" not in geometry_panel_cell


def test_geometry_selectors_only_offer_projection_backed_single_view_geometries() -> None:
    selector_cells = "\n".join(render_selector_cells())

    assert "projected_geometry_rows = [" in selector_cells
    assert 'row for row in _geometry.geometry_rows if row.get("projection_ids")' in selector_cells
    assert "_selector_rows = projected_geometry_rows or _geometry.geometry_rows" in selector_cells
    assert "for row in _selector_rows" in selector_cells
    assert 'return "Evo 2 " + _value.removeprefix("evo2_").upper()' in selector_cells
    assert 'for row in _geometry.geometry_rows\n                    if str(row.get("model"))' not in selector_cells


def test_geometry_reference_selector_preserves_user_selection_across_view_switches() -> None:
    selector_cells = "\n".join(render_selector_cells())
    geometry_hue_cell = render_geometry_hue_selector_cell()

    assert "get_requested_reference, set_requested_reference = _support.mo.state(default_reference)" in selector_cells
    assert "get_requested_reference" in geometry_hue_cell
    assert "_requested_reference" in geometry_hue_cell
    assert "if _requested_reference in _reference_values" in geometry_hue_cell
    assert "on_change=set_requested_reference" in geometry_hue_cell


def test_reference_annotation_mode_selector_controls_geometry_and_plot_label_limits() -> None:
    selector_cells = "\n".join(render_selector_cells())
    geometry_hue_cell = render_geometry_hue_selector_cell()
    geometry_panel_cell = render_geometry_panel_cell()
    plot_review_cell = render_plot_review_cell()

    assert "get_requested_reference_annotation_mode" in selector_cells
    assert "set_requested_reference_annotation_mode" in selector_cells
    assert "Reference annotations" in geometry_hue_cell
    assert "geometry_reference_annotation_selector" in geometry_panel_cell
    assert "reference_label_limit=" in geometry_panel_cell
    assert "plot_reference_annotation_selector" in plot_review_cell
    assert "_support = runtime.support" in plot_review_cell
    assert "reference_label_limit=plot_reference_label_limit" in plot_review_cell
