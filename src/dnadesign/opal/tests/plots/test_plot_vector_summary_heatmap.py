"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/tests/plots/test_plot_vector_summary_heatmap.py

Regression tests for plot vector summary heatmap OPAL plots.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dnadesign.opal.src.plots import vector_summary_heatmap as plot_mod
from dnadesign.opal.src.plots._context import PlotContext


class _DummyWorkspace:
    def __init__(self, outputs_dir: Path):
        self.outputs_dir = outputs_dir


def test_vector_summary_explicit_reference_does_not_require_objective_setpoint(tmp_path, monkeypatch) -> None:
    def _stub_load_events(outputs_dir, base_columns, round_selector=None, run_id=None, **_kwargs):
        assert "obj__diag__setpoint" not in base_columns
        return pd.DataFrame(
            {
                "as_of_round": [0, 0],
                "run_id": ["r0", "r0"],
                "pred__y_hat_model": [[0.25, 0.75], [0.75, 0.25]],
            }
        )

    def _fail_load_events_with_setpoint(*args, **kwargs):
        raise AssertionError("explicit reference_vector must not require objective setpoint metadata")

    monkeypatch.setattr(plot_mod, "load_events", _stub_load_events)
    monkeypatch.setattr(plot_mod, "load_events_with_setpoint", _fail_load_events_with_setpoint)

    ctx = PlotContext(
        campaign_dir=tmp_path,
        workspace=_DummyWorkspace(tmp_path),
        rounds="all",
        run_id=None,
        selection_view_id="primary",
        data_paths={},
        output_dir=tmp_path / "plots",
        filename="vector_summary.png",
        dpi=72,
        format="png",
        logger=logging.getLogger("opal.test.vector_summary"),
        save_data=True,
    )

    ctx.output_dir.mkdir(parents=True, exist_ok=True)
    plot_mod.render(
        ctx,
        params={
            "cohort": "all_pool",
            "include_reference_vector": True,
            "reference_vector": [0.0, 1.0],
            "reference_label": "target vec2",
            "channel_labels": ["a", "b"],
        },
    )

    tidy = pd.read_csv(ctx.output_dir / "vector_summary.csv")
    assert set(tidy["row_type"]) == {"reference_vector", "reference_mse", "round"}
    assert tidy.loc[tidy["row_type"] == "reference_vector", "cohort"].unique().tolist() == ["target vec2"]
    assert tidy.loc[tidy["row_type"] == "reference_mse", "channel"].unique().tolist() == ["mse"]


def test_vector_summary_heatmap_tick_size_adapts_to_channel_count() -> None:
    normal = plot_mod._adaptive_heatmap_tick_font_size(
        dim=8,
        row_count=5,
        figsize=(10.8, 5.2),
        requested=13,
        has_reference_panel=True,
    )
    dense = plot_mod._adaptive_heatmap_tick_font_size(
        dim=48,
        row_count=18,
        figsize=(10.8, 5.2),
        requested=13,
        has_reference_panel=True,
    )

    assert normal == 13
    assert 8.5 <= dense < normal


def test_vector_summary_default_layout_is_compact_for_one_round() -> None:
    width, height = plot_mod._default_heatmap_figsize(dim=8, row_count=1)

    assert width > height
    assert height <= 4.0


def test_vector_summary_centered_norm_is_symmetric_and_annotations_are_bounded() -> None:
    matrix = np.asarray([[-2.0, 0.5, 3.0]], dtype=float)
    norm = plot_mod._heatmap_norm(matrix, center=0.0)

    assert norm is not None
    assert norm.vmin == -3.0
    assert norm.vcenter == 0.0
    assert norm.vmax == 3.0

    figure, axis = plt.subplots()
    annotations = plot_mod._annotate_heatmap_values(axis, matrix, enabled=True, max_cells=8)
    assert [text.get_text() for text in annotations] == ["-2.00", "0.50", "3.00"]
    assert plot_mod._annotate_heatmap_values(axis, matrix, enabled=True, max_cells=2) == []
    plt.close(figure)
