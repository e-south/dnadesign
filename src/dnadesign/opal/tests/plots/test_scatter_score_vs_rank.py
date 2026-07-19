"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/tests/plots/test_scatter_score_vs_rank.py

Focused rendering tests for the score-versus-rank diagnostic.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from dnadesign.opal.src.plots import scatter_score_vs_rank as plot_mod
from dnadesign.opal.src.plots._context import PlotContext


class _DummyWorkspace:
    def __init__(self, outputs_dir: Path):
        self.outputs_dir = outputs_dir
        self.workdir = outputs_dir.parent


def test_score_rank_reference_boundary_is_annotated_and_included_without_loose_scale(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = pd.DataFrame(
        {
            "as_of_round": [0, 0, 0],
            "run_id": ["r0", "r0", "r0"],
            "id": ["candidate-a", "candidate-b", "candidate-c"],
            "view__rank_competition": [1, 2, 3],
            "view__is_selected": [True, False, False],
            "view__selection_score": [-3.0, -4.0, -5.0],
        }
    )
    monkeypatch.setattr(plot_mod, "load_events", lambda *_args, **_kwargs: events)
    captured: list[plt.Figure] = []

    def _capture(fig, _out, *, dpi: int, tight: bool = True) -> None:
        assert dpi == 96
        assert tight is False
        captured.append(fig)

    monkeypatch.setattr(plot_mod, "save_notebook_square_figure", _capture)
    context = PlotContext(
        campaign_dir=tmp_path,
        workspace=_DummyWorkspace(tmp_path / "outputs"),
        rounds=[0],
        run_id="r0",
        selection_view_id="ethanol",
        data_paths={},
        output_dir=tmp_path / "plots",
        filename="score-vs-rank.png",
        dpi=96,
        format="png",
        logger=logging.getLogger("opal.test.score-rank-reference"),
        save_data=False,
    )

    plot_mod.render(
        context,
        {
            "score_field": "view__selection_score",
            "rank_mode": "competition",
            "legend_location": "upper_left",
            "title": "RMF score by active-view rank",
            "title_location": "center",
            "show_selection_view": True,
            "selection_marker_label": "Allocated to this view",
            "y_axis": {
                "reference_lines": [
                    {"value": 0.0, "label": "Feasibility boundary"},
                ]
            },
        },
    )

    assert len(captured) == 1
    ax = captured[0].axes[0]
    reference_lines = [line for line in ax.lines if line.get_linestyle() == "--" and np.allclose(line.get_ydata(), 0.0)]
    assert len(reference_lines) == 1
    assert [text.get_text() for text in ax.texts] == ["Feasibility boundary"]
    lower, upper = ax.get_ylim()
    assert lower < -5.0 < upper
    assert upper > 0.0
    assert upper <= 0.30
    legend = ax.get_legend()
    assert legend is not None
    assert "Allocated to this view" in [text.get_text() for text in legend.get_texts()]
    assert ax.get_title(loc="center") == "RMF score by active view rank · Ethanol view"
    assert ax.get_title(loc="left") == ""
    assert ax.title.get_fontsize() >= 14
    assert min(text.get_fontsize() for text in legend.get_texts()) >= 9.5
    captured[0].canvas.draw()
    legend_bounds = legend.get_window_extent(captured[0].canvas.get_renderer())
    reference_y = ax.transData.transform((1.0, 0.0))[1]
    assert not (legend_bounds.y0 <= reference_y <= legend_bounds.y1)
