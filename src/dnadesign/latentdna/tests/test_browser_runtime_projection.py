"""Projection rendering regression tests for notebook geometry audit surfaces."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dnadesign.latentdna.src.notebooks import browser_runtime_projection as projection_runtime


def _panel_offsets(fig) -> list[tuple[float, float]]:
    offsets: list[tuple[float, float]] = []
    for collection in fig.axes[0].collections:
        collection_offsets = np.asarray(collection.get_offsets())
        if collection_offsets.size == 0:
            continue
        offsets.extend((float(x), float(y)) for x, y in collection_offsets.tolist())
    return sorted(offsets)


def test_render_projection_grid_keeps_point_coordinates_fixed_across_hues(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(projection_runtime, "render_matplotlib_figure", lambda fig, alt=None: fig)

    frame = pd.DataFrame(
        {
            "x": [0.0, 1.0, 2.0, 3.0],
            "y": [3.0, 2.0, 1.0, 0.0],
            "design_family": ["ethanol", "ethanol", "cipro", "cipro"],
        }
    )
    panel_specs = [{"view_id": "view_a", "projection_id": "proj_a", "title": "Anchor view"}]

    fig_without_hue = projection_runtime.render_projection_grid(
        panel_specs,
        frames=[frame],
        hue_column=None,
        hue_kinds={"design_family": "categorical"},
        joinable_tables=[],
        reference_labels=[],
        output_root=tmp_path,
        workspace_dir=tmp_path,
    )
    fig_with_hue = projection_runtime.render_projection_grid(
        panel_specs,
        frames=[frame],
        hue_column="design_family",
        hue_kinds={"design_family": "categorical"},
        joinable_tables=[],
        reference_labels=[],
        output_root=tmp_path,
        workspace_dir=tmp_path,
    )

    try:
        assert _panel_offsets(fig_without_hue) == _panel_offsets(fig_with_hue)
    finally:
        plt.close(fig_without_hue)
        plt.close(fig_with_hue)


def test_render_projection_grid_suppresses_degenerate_continuous_colorbar(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(projection_runtime, "render_matplotlib_figure", lambda fig, alt=None: fig)

    frame = pd.DataFrame(
        {
            "x": [0.0, 1.0, 2.0],
            "y": [0.0, 1.0, 2.0],
            "wildtype_margin_ethanol_vs_control": [0.25, 0.25, 0.25],
        }
    )
    panel_specs = [{"view_id": "view_a", "projection_id": "proj_a", "title": "Anchor view"}]

    fig = projection_runtime.render_projection_grid(
        panel_specs,
        frames=[frame],
        hue_column="wildtype_margin_ethanol_vs_control",
        hue_kinds={"wildtype_margin_ethanol_vs_control": "continuous"},
        joinable_tables=[],
        reference_labels=[],
        output_root=tmp_path,
        workspace_dir=tmp_path,
    )

    try:
        assert len(fig.axes) == 1
    finally:
        plt.close(fig)
